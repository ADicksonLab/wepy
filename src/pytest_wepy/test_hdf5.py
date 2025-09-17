# Testing hdf5 functionality
#
# 1) writing to HDF5 during simulation (boundary conditions, resampling)
# 2) reading in HDF5 (tests on sensible data)
# 3) compute observable
# 4) get traces

# Standard Library
import os

# Third Party Library
import mdtraj as mdj
import numpy as np

# First Party Library
from wepy.boundary_conditions.boundary import NoBC
from wepy.boundary_conditions.randomwalk import RandomWalkBC
from wepy.hdf5 import WepyHDF5
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.resampling.distances.randomwalk import RandomWalkDistance
from wepy.resampling.resamplers.resampler import NoResampler
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.runners.randomwalk import UNIT_NAMES, RandomWalkRunner
from wepy.sim_manager import Manager
from wepy.util.mdtraj import mdtraj_to_json_topology
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.mapper import Mapper

num_walkers = 20
hdf5_filename = "test.h5"
segment_length = 1
threshold = 5


def generate_topology(N):
    """Creates an N-atom, dummy trajectory and topology for
    the randomwalk system using the mdtraj package.  Then creates a
    JSON format for the topology. This JSON string is used in making
    the WepyHDF5 reporter.

    Returns
    -------
    topology: str
        JSON string representing the topology of system being simulated.
    """
    data = []
    top = mdj.Topology()
    c = top.add_chain()
    r = top.add_residue("test", c)

    for i in range(N):
        at = top.add_atom(f"a{i}", mdj.element.argon, r, i)

    json_top_str = mdtraj_to_json_topology(top)
    return json_top_str


def test_WriteReadH5():
    cleanup = True

    # 1D random walk
    positions = np.zeros((1, 1))

    init_state = WalkerState(positions=positions, time=0.0)

    # create list of init_walkers
    initial_weight = 1 / num_walkers
    init_walkers = []

    init_walkers = [Walker(init_state, initial_weight) for i in range(num_walkers)]

    # set up runner for system
    runner = RandomWalkRunner(probability=0.5)

    units = dict(UNIT_NAMES)
    # instantiate a revo resampler and unbindingboundarycondition

    rw_distance = RandomWalkDistance()

    resampler = REVOResampler(
        merge_dist=100,
        char_dist=0.1,
        distance=rw_distance,
        init_state=init_state,
        weights=True,
        pmax=0.5,
        dist_exponent=4,
    )

    json_top = generate_topology(1)

    rw_bc = RandomWalkBC(threshold=threshold, initial_states=[init_state])

    hdf5_reporter = WepyHDF5Reporter(
        file_path=hdf5_filename,
        mode="w",
        save_fields=["positions"],
        boundary_conditions=rw_bc,
        topology=json_top,
        resampler=resampler,
        n_dims=1,
    )

    sim_manager = Manager(
        init_walkers,
        runner=runner,
        resampler=resampler,
        boundary_conditions=rw_bc,
        work_mapper=Mapper(),
        reporters=[hdf5_reporter],
    )

    n_cycles = 20
    steps = [segment_length for i in range(n_cycles)]

    sim_manager.run_simulation(n_cycles, steps)

    # ------
    # data collected! now analyze
    # ------

    we = WepyHDF5(hdf5_filename, mode="r")

    initial_trace = [(i, 0) for i in range(num_walkers)]
    final_trace = [(i, n_cycles - 1) for i in range(num_walkers)]
    with we:
        initial_pos = we.get_run_trace_fields(0, initial_trace, ["positions"])
        final_pos = we.get_run_trace_fields(0, final_trace, ["positions"])

    # test that you haven't moved more than segment_length positions in the first cycle
    assert initial_pos["positions"].max() <= segment_length
    assert initial_pos["positions"].min() >= 0

    # test that some of the trajectories have moved in the final pos
    assert final_pos["positions"].max() >= 0

    # test that no positions are further than the boundary condition
    assert final_pos["positions"].max() <= int(threshold)
    assert final_pos["positions"].min() >= 0

    if cleanup:
        os.remove(hdf5_filename)


def makeRandomWalkH5(h5name, resampling=True, warping=True):
    # 1D random walk
    positions = np.zeros((1, 1))

    init_state = WalkerState(positions=positions, time=0.0)

    # create list of init_walkers
    initial_weight = 1 / num_walkers
    init_walkers = []

    init_walkers = [Walker(init_state, initial_weight) for i in range(num_walkers)]

    # set up runner for system
    runner = RandomWalkRunner(probability=0.5)

    units = dict(UNIT_NAMES)
    # instantiate a revo resampler and unbindingboundarycondition

    rw_distance = RandomWalkDistance()

    if resampling:
        resampler = REVOResampler(
            merge_dist=100,
            char_dist=0.1,
            distance=rw_distance,
            init_state=init_state,
            weights=True,
            pmax=0.5,
            dist_exponent=4,
        )
    else:
        resampler = NoResampler()

    json_top = generate_topology(1)

    if warping:
        rw_bc = RandomWalkBC(threshold=threshold, initial_states=[init_state])
    else:
        rw_bc = NoBC()

    hdf5_reporter = WepyHDF5Reporter(
        file_path=h5name,
        mode="w",
        save_fields=["positions"],
        boundary_conditions=rw_bc,
        topology=json_top,
        resampler=resampler,
        n_dims=1,
    )

    sim_manager = Manager(
        init_walkers,
        runner=runner,
        resampler=resampler,
        boundary_conditions=rw_bc,
        work_mapper=Mapper(),
        reporters=[hdf5_reporter],
    )

    n_cycles = 20
    steps = [segment_length for i in range(n_cycles)]

    sim_manager.run_simulation(n_cycles, steps)


if __name__ == "__main__":
    makeRandomWalkH5("test_data/rw.h5", resampling=True, warping=True)
    makeRandomWalkH5("test_data/rw_noresampling.h5", resampling=False, warping=True)
    makeRandomWalkH5("test_data/rw_nowarping.h5", resampling=True, warping=False)

    print("HDF5s written")
