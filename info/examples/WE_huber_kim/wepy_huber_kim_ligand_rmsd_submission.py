"""KSL19/sEH Huber--Kim WE using receptor-aligned ligand RMSD.

This is the bin-based counterpart to the supplied REVO script. It reuses
``UnbindingDistance`` for periodic imaging, binding-site alignment, and ligand
RMSD, while ``UnbindingBC`` independently detects and recycles unbound walkers.
"""

import os
import os.path as osp
import pickle as pkl

import mdtraj as mdj
import numpy as np
import simtk.openmm as omm
import simtk.unit as unit

from wepy.boundary_conditions.receptor import UnbindingBC
from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.reporter.openmm import OpenMMRunnerDashboardSection
from wepy.reporter.walker_pkl import WalkerPklReporter
from wepy.resampling.distances.receptor import UnbindingDistance
from wepy.resampling.resamplers.huber_kim import HuberKimResampler, RectilinearBinMapper
from wepy.runners.openmm import (
    OpenMMGPUWalkerTaskProcess, OpenMMRunner, OpenMMState, OpenMMWalker,
    gen_sim_state,
)
from wepy.sim_manager import Manager
from wepy.util.mdtraj import mdtraj_to_json_topology
from wepy.work_mapper.task_mapper import TaskMapper


N_STEPS, N_CYCLES, NUM_WALKERS, RUN = 10000, 2000, 48, 2
TARGET_WALKERS_PER_BIN = 4
RMSD_EDGES_NM = np.concatenate((np.arange(0.0, 2.0 + 0.25, 0.25), [np.inf]))
OUTPUTS_DIR = (
    "/dickson/s1/bosesami/KSL_unbinding/KSL_19/"
    f"charmm-gui-4270325889/openmm/output_HK_RMSD_{N_STEPS}_{N_CYCLES}_{RUN}"
)


class ReferenceLigandRMSD:
    """Convert pairwise ``UnbindingDistance`` into RMSD from its reference."""

    def __init__(self, distance):
        self.distance = distance

    def __call__(self, walker):
        """Return a one-element protein-aligned ligand RMSD coordinate."""
        image = self.distance.image(walker.state)
        value = float(self.distance.image_distance(self.distance.ref_image, image))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid ligand RMSD: {value}")
        return np.asarray([value])


if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open("system.pkl", "rb") as stream:
        system = pkl.load(stream)
    with open("topology.pkl", "rb") as stream:
        omm_top = pkl.load(stream)
    with open("KSL19_sEH_binding_50ns.rst") as stream:
        state = omm.XmlSerializer.deserialize(stream.read())

    box_vectors, positions = state.getPeriodicBoxVectors(), state.getPositions()
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    integrator = omm.LangevinIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )
    walker_state = OpenMMState(gen_sim_state(positions, system, integrator))
    runner = OpenMMRunner(system, omm_top, integrator, platform="CUDA")

    pdb = mdj.load_pdb("KSL19_sEH_binding_50ns.pdb")
    json_top = mdtraj_to_json_topology(pdb.top)
    ligand_idxs = pdb.top.select("resname K19")
    protein_idxs = pdb.top.select("protein")
    binding_site_idxs = mdj.compute_neighbors(
        pdb, 0.5, ligand_idxs, haystack_indices=protein_idxs, periodic=True
    )[0]
    if len(ligand_idxs) == 0 or len(binding_site_idxs) < 3:
        raise ValueError("Ligand selection is empty or binding site is too small")

    walkers = [
        OpenMMWalker(walker_state, 1.0 / NUM_WALKERS)
        for _ in range(NUM_WALKERS)
    ]
    old_revo_metric = UnbindingDistance(
        ligand_idxs, binding_site_idxs, walker_state
    )
    pcoord = ReferenceLigandRMSD(old_revo_metric)
    bin_mapper = RectilinearBinMapper([RMSD_EDGES_NM])
    resampler = HuberKimResampler(
        progress_coordinate=pcoord,
        bin_mapper=bin_mapper,
        bin_target_counts=TARGET_WALKERS_PER_BIN,
        adjust_counts=True,
        weight_split_threshold=2.0,
        weight_merge_cutoff=1.0,
        do_thresholds=True,
        largest_allowed_weight=1.0,
        smallest_allowed_weight=1e-310,
        rng=2026000 + RUN,
        cache_progress_coordinates=True,
        min_num_walkers=None,
        max_num_walkers=None,
    )

    boundary_condition = UnbindingBC(
        cutoff_distance=1.0,
        initial_state=walker_state,
        topology=json_top,
        ligand_idxs=ligand_idxs,
        receptor_idxs=protein_idxs,
    )
    reporters = [
        WepyHDF5Reporter(
            save_fields=("positions", "box_vectors"),
            file_path=osp.join(OUTPUTS_DIR, "wepy.results.h5"),
            resampler=resampler,
            boundary_conditions=boundary_condition,
            topology=json_top,
        ),
        WalkerPklReporter(
            save_dir=osp.join(OUTPUTS_DIR, "pkls"), freq=1, num_backups=2
        ),
        DashboardReporter(
            file_path=osp.join(OUTPUTS_DIR, "wepy.dash.org"),
            runner_dash=OpenMMRunnerDashboardSection(runner),
        ),
    ]
    mapper = TaskMapper(
        walker_task_type=OpenMMGPUWalkerTaskProcess,
        num_workers=4,
        platform="CUDA",
        device_ids=[0, 1, 2, 3],
    )
    manager = Manager(
        walkers,
        runner=runner,
        resampler=resampler,
        boundary_conditions=boundary_condition,
        work_mapper=mapper,
        reporters=reporters,
    )
    print(
        f"Running Huber-Kim ligand-RMSD WE: {bin_mapper.nbins} bins, "
        f"{TARGET_WALKERS_PER_BIN} walkers per occupied bin"
    )
    manager.run_simulation(N_CYCLES, [N_STEPS] * N_CYCLES)
