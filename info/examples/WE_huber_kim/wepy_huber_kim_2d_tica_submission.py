"""Wepy submission script for 2-D tICA Huber-Kim weighted ensemble.

This is adapted from the user's REVO submission script. The important change is
that the Huber-Kim progress coordinate is the *direct two-component tICA
projection* returned by DistanceTICAProjector.project(state):

    pcoord(X) = [tIC1(X), tIC2(X)]

No radial/reference distance is computed.
"""

import argparse
import glob
import os
import os.path as osp
import pickle as pkl
import re
import socket

import mdtraj as mdj
import numpy as np
import simtk.openmm as omm
import simtk.unit as unit

from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.reporter.openmm import OpenMMRunnerDashboardSection
from wepy.reporter.walker_pkl import WalkerPklReporter
from wepy.resampling.projectors.tica import DistanceTICAProjector
from wepy.resampling.projectors.tica import shorten_vecs
from wepy.resampling.resamplers.huber_kim import (
    HuberKimResampler,
    RectilinearBinMapper,
)
from wepy.runners.openmm import (
    OpenMMGPUWalkerTaskProcess,
    OpenMMRunner,
    OpenMMState,
    OpenMMWalker,
    gen_sim_state,
)
from wepy.sim_manager import Manager
from wepy.util.mdtraj import mdtraj_to_json_topology
from wepy.util.util import box_vectors_to_lengths_angles
from wepy.work_mapper.task_mapper import TaskMapper


print(f"hostname: {socket.gethostname()}")


class QuietDistanceTICAProjector(DistanceTICAProjector):
    """Distance-tICA projection without the installed class's per-call print."""

    def project(self, state):
        disp_vecs = (
            state["positions"][self.dist_idxs[:, 0]]
            - state["positions"][self.dist_idxs[:, 1]]
        )
        if self.periodic:
            box_lengths, _ = box_vectors_to_lengths_angles(state["box_vectors"])
            disp_vecs = shorten_vecs(disp_vecs, box_lengths)
        dists = np.linalg.norm(disp_vecs, axis=1)
        projection = self.model.transform(dists)
        return self.tica_weights * projection


class TICA2DProgressCoordinate:
    """Return the exact first two components of a wepy tICA projector.

    ``DistanceTICAProjector.project(state)`` in the user's projector code
    computes the pair-distance feature vector and feeds it directly through
    ``tica_model.transform``. This adapter only normalizes the returned shape;
    it does not calculate a distance to a reference or between walkers.
    """

    def __init__(self, projector, components=(0, 1)):
        self.projector = projector
        self.components = tuple(int(i) for i in components)
        if len(self.components) != 2:
            raise ValueError("TICA2DProgressCoordinate requires exactly two components")
        if min(self.components) < 0:
            raise ValueError("TICA component indices must be non-negative")

    def __call__(self, walker):
        projection = np.asarray(self.projector.project(walker.state), dtype=float).reshape(-1)

        if projection.size <= max(self.components):
            raise ValueError(
                "Projector returned {} values, but components {} were requested".format(
                    projection.size, self.components
                )
            )

        pcoord = projection[list(self.components)]
        if np.any(~np.isfinite(pcoord)):
            raise ValueError(f"Non-finite tICA progress coordinate: {pcoord}")

        return pcoord


def get_latest_walker_pkl(search_dir):
    """Find the walker pickle with the highest cycle number."""
    pkl_dir = osp.join(search_dir, "pkls")
    if not osp.exists(pkl_dir):
        raise FileNotFoundError(f"No 'pkls' directory found in {search_dir}")

    files = glob.glob(osp.join(pkl_dir, "walkers_cycle_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No walker pickle files found in {pkl_dir}")

    def extract_cycle(fname):
        match = re.search(r"walkers_cycle_(\d+).pkl", fname)
        return int(match.group(1)) if match else -1

    latest_file = max(files, key=extract_cycle)
    print(f"Resuming from: {latest_file}")
    return latest_file


def build_resampling_rng(run, sub_step, previous_output_dir=None):
    """Create the MT19937 WE RNG and restore its state between sub-steps."""
    seed = 2026000 + int(run)
    rng = np.random.Generator(np.random.MT19937(seed))

    if sub_step > 0 and previous_output_dir is not None:
        state_path = osp.join(previous_output_dir, "huber_kim_rng_state.pkl")
        if osp.exists(state_path):
            with open(state_path, "rb") as f:
                rng.bit_generator.state = pkl.load(f)
            print(f"Restored Huber-Kim RNG state from: {state_path}")
        else:
            print(
                "WARNING: no previous Huber-Kim RNG state file was found; "
                "resampling RNG will restart from the run seed."
            )

    return rng


def output_dir_for_substep(
    wepy_sim_path,
    run,
    initial_num_walkers,
    n_steps,
    n_cycles,
    sub_step,
    target_walkers_per_bin,
    n_bins,
):
    return osp.join(
        wepy_sim_path,
        (
            f"run{run}_initwalk{initial_num_walkers}_steps{n_steps}_cycs{n_cycles}_"
            f"substep{sub_step}_HK2D_target{target_walkers_per_bin}_nbins{n_bins}"
        ),
    )


if __name__ == "__main__":
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_step", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--protein_name", type=str, required=True)
    parser.add_argument("--pdb_id", type=str, required=True)
    parser.add_argument("--asam_clust_id", type=int, required=True)
    parser.add_argument("--num_excluded_clusters", type=int, required=True)
    parser.add_argument("--target_walkers_per_bin", type=int, default=4)

    args = parser.parse_args()

    sub_step = args.sub_step
    asam_clust_id = args.asam_clust_id
    protein_name = args.protein_name
    pdb_id = args.pdb_id
    n_workers = args.n_workers
    num_excluded_clusters = args.num_excluded_clusters
    target_walkers_per_bin = args.target_walkers_per_bin

    # Simulation parameters.
    # NOTE: this is the INITIAL walker count only; Huber-Kim binned WE permits
    # the total population to change as 2-D bins become occupied/empty.
    initial_num_walkers = 48
    n_steps = 50000
    run = 0
    n_cycles = 250
    tica_tau = 50
    ntica = 2

    # WESTPA/H-K controls.
    adjust_counts = True
    weight_split_threshold = 2.0
    weight_merge_cutoff = 1.0
    do_thresholds = True
    largest_allowed_weight = 1.0
    smallest_allowed_weight = 1e-310

    if target_walkers_per_bin < 1:
        raise ValueError("target_walkers_per_bin must be at least 1")

    if cuda_visible is not None:
        num_available = len([x for x in cuda_visible.split(",") if x.strip()])
        print(f"{num_available} available devices...")
        device_ids = list(range(min(n_workers, num_available)))
        print(f"Running with devices: {device_ids} on {socket.gethostname()}")
    else:
        raise RuntimeError("No GPUs available: CUDA_VISIBLE_DEVICES is not set or empty")

    # Paths.
    base_path = "/mnt/scratch/bosesami/nodes"
    inp_path = f"{base_path}/tica_training/tica_out/"
    md_path = (
        f"{base_path}/{protein_name}/state_{pdb_id}.small/clustering/"
        f"cluster_repr_{asam_clust_id}/"
    )
    wepy_sim_path = (
        f"{base_path}/wepy_sims/{protein_name}_{pdb_id}/aSAM_clust{asam_clust_id}/"
    )

    cutoff_nm = 1
    min_seq_sep = 5

    tica_model_filepath = (
        f"{inp_path}/ticamodel_{cutoff_nm}_{min_seq_sep}_{protein_name}_{pdb_id}_"
        f"{ntica}tic_{tica_tau}tau_{num_excluded_clusters}ClustersExcld.pkl"
    )
    tica_distance_idx_filepath = (
        f"{inp_path}/united_dist_feat_{protein_name}_{pdb_id}_{cutoff_nm}_{min_seq_sep}_"
        f"{num_excluded_clusters}ClustExcld_atompairs.pkl"
    )

    pdb_path = f"{md_path}/clust{asam_clust_id}_final_run1.pdb"
    rst_path = f"{md_path}/clust{asam_clust_id}_prod_run1.rst"
    system_path = f"{md_path}/system.pkl"
    topology_path = f"{md_path}/topology.pkl"

    # Fixed, uniform grid for direct tIC0/tIC1 binning. Infinite outer bins
    # ensure newly discovered projections outside [-3, 3] remain valid.
    tic0_edges = np.array(
        [-np.inf, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, np.inf],
        dtype=float,
    )
    tic1_edges = np.array(
        [-np.inf, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, np.inf],
        dtype=float,
    )
    bin_mapper = RectilinearBinMapper([tic0_edges, tic1_edges])
    n_bins = bin_mapper.nbins

    outputs_dir = output_dir_for_substep(
        wepy_sim_path,
        run,
        initial_num_walkers,
        n_steps,
        n_cycles,
        sub_step,
        target_walkers_per_bin,
        n_bins,
    )
    os.makedirs(outputs_dir, exist_ok=True)

    pdb = mdj.load_pdb(pdb_path)
    json_top = mdtraj_to_json_topology(pdb.top)

    with open(tica_model_filepath, "rb") as f:
        tica_model = pkl.load(f)
    with open(tica_distance_idx_filepath, "rb") as f:
        dist_idxs = pkl.load(f)

    with open(system_path, "rb") as f:
        system = pkl.load(f)
    with open(topology_path, "rb") as f:
        omm_top = pkl.load(f)

    with open(rst_path, "r") as f:
        simtk_state = omm.XmlSerializer.deserialize(f.read())
        bv = simtk_state.getPeriodicBoxVectors()
        pos = simtk_state.getPositions()

    system.setDefaultPeriodicBoxVectors(bv[0], bv[1], bv[2])

    integrator = omm.LangevinIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    runner = OpenMMRunner(system, omm_top, integrator, platform="CUDA")

    previous_output_dir = None

    if sub_step == 0:
        new_simtk_state = gen_sim_state(pos, system, integrator)
        walker_state = OpenMMState(new_simtk_state)
        init_weight = 1.0 / initial_num_walkers
        init_walkers = [
            OpenMMWalker(walker_state, init_weight)
            for _ in range(initial_num_walkers)
        ]
        print(
            f"Starting Huber-Kim 2-D WE with {initial_num_walkers} initial walkers; "
            f"target={target_walkers_per_bin}/occupied bin."
        )

    elif sub_step > 0:
        previous_output_dir = output_dir_for_substep(
            wepy_sim_path,
            run,
            initial_num_walkers,
            n_steps,
            n_cycles,
            sub_step - 1,
            target_walkers_per_bin,
            n_bins,
        )

        if not osp.exists(previous_output_dir):
            raise FileNotFoundError(
                f"Previous job directory {previous_output_dir} does not exist. Cannot restart."
            )

        last_pkl_path = get_latest_walker_pkl(previous_output_dir)
        with open(last_pkl_path, "rb") as f:
            init_walkers = pkl.load(f)

        print(
            f"Restarting from sub_step {sub_step - 1} with "
            f"{len(init_walkers)} walkers from the latest pickle."
        )

    else:
        raise ValueError("sub_step must be >= 0")

    # EXACT DIRECT 2-D TICA PROGRESS COORDINATE.
    # DistanceTICAProjector.project(state) -> tICA model projection.
    # We do NOT construct ProjectorDistance and do NOT calculate a radial norm.
    dist_tica_projector = QuietDistanceTICAProjector(
        dist_idxs=dist_idxs,
        tica_model=tica_model,
        periodic=True,
    )
    progress_coordinate = TICA2DProgressCoordinate(
        projector=dist_tica_projector,
        components=(0, 1),
    )

    first_pcoord = progress_coordinate(init_walkers[0])
    print(f"First walker direct 2-D tICA pcoord: {first_pcoord}")
    print(f"2-D bin grid shape: {bin_mapper.nbins_per_dim}; total bins={bin_mapper.nbins}")

    we_rng = build_resampling_rng(
        run=run,
        sub_step=sub_step,
        previous_output_dir=previous_output_dir,
    )

    resampler = HuberKimResampler(
        progress_coordinate=progress_coordinate,
        bin_mapper=bin_mapper,
        bin_target_counts=target_walkers_per_bin,
        adjust_counts=adjust_counts,
        weight_split_threshold=weight_split_threshold,
        weight_merge_cutoff=weight_merge_cutoff,
        do_thresholds=do_thresholds,
        largest_allowed_weight=largest_allowed_weight,
        smallest_allowed_weight=smallest_allowed_weight,
        rng=we_rng,
        cache_progress_coordinates=True,
        min_num_walkers=None,
        max_num_walkers=None,
    )

    print(
        "Built Huber-Kim 2-D WE resampler: "
        f"target={target_walkers_per_bin}/occupied bin, "
        f"adjust_counts={adjust_counts}."
    )

    hdf5_reporter = WepyHDF5Reporter(
        save_fields=("positions", "box_vectors"),
        file_path=osp.join(outputs_dir, "wepy.results.h5"),
        resampler=resampler,
        topology=json_top,
    )

    pkl_reporter = WalkerPklReporter(
        save_dir=osp.join(outputs_dir, "pkls"),
        freq=1,
        num_backups=2,
    )

    dashboard_reporter = DashboardReporter(
        file_path=osp.join(outputs_dir, "wepy.dash.org"),
        runner_dash=OpenMMRunnerDashboardSection(runner),
    )

    mapper = TaskMapper(
        walker_task_type=OpenMMGPUWalkerTaskProcess,
        num_workers=n_workers,
        platform="CUDA",
        device_ids=device_ids,
    )

    sim_manager = Manager(
        init_walkers,
        runner=runner,
        resampler=resampler,
        work_mapper=mapper,
        reporters=[hdf5_reporter, pkl_reporter, dashboard_reporter],
    )

    steps_list = [n_steps for _ in range(n_cycles)]
    print("Running Huber-Kim 2-D WE simulation...")
    sim_manager.run_simulation(n_cycles, steps_list)

    # Preserve resampler RNG state across completed sub-steps.
    rng_state_path = osp.join(outputs_dir, "huber_kim_rng_state.pkl")
    with open(rng_state_path, "wb") as f:
        pkl.dump(resampler.rng.bit_generator.state, f)
    print(f"Saved Huber-Kim RNG state to: {rng_state_path}")
