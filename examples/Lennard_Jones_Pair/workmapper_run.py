import sys
from copy import copy

import numpy as np

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from openmmtools.testsystems import LennardJonesPair
import mdtraj as mdj

from wepy.sim_manager import Manager

from wepy.resampling.distances.distance import Distance
from wepy.resampling.wexplore1 import WExplore1Resampler
from wepy.walker import Walker
from wepy.runners.openmm import OpenMMRunner, OpenMMState
from wepy.runners.openmm import UNIT_NAMES, GET_STATE_KWARG_DEFAULTS
from wepy.work_mapper.mapper import WorkerMapper
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.reporter.reporter import WalkersPickleReporter


from scipy.spatial.distance import euclidean

# we define a simple distance metric for this system, assuming the
# positions are in a 'positions' field
class PairDistance(Distance):

    def __init__(self, metric=euclidean):
        self.metric = metric

    def preimage(self, state):
        return state['positions']

    def preimage_distance(self, preimage_a, preimage_b):
        dist_a = self.metric(preimage_a[0], preimage_a[1])
        dist_b = self.metric(preimage_b[0], preimage_b[1])

        return np.abs(dist_a - dist_b)

def main(n_runs, n_cycles, steps, n_workers, debug_prints=False, seed=None):

    test_sys = LennardJonesPair()

    #integrator = omm.VerletIntegrator(2*unit.femtoseconds)
    integrator = omm.LangevinIntegrator(300.0*unit.kelvin, 1/unit.picosecond, 2*unit.femtoseconds)
    context = omm.Context(test_sys.system, copy(integrator))
    context.setPositions(test_sys.positions)

    get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
    init_sim_state = context.getState(**get_state_kwargs)

    thermostat = omm.AndersenThermostat(300.0 * unit.kelvin, 1/unit.picosecond)
    barostat = omm.MonteCarloBarostat(1.0*unit.atmosphere, 300.0*unit.kelvin, 50)

    runner = OpenMMRunner(test_sys.system, test_sys.topology, integrator, platform='Reference')

    num_walkers = 10
    init_weight = 1.0 / num_walkers

    init_walkers = [Walker(OpenMMState(init_sim_state), init_weight) for i in range(num_walkers)]


    # the mdtraj here is needed for the distance function
    mdtraj_topology = mdj.Topology.from_openmm(test_sys.topology)

    # make a distance object which can be used to compute the distance
    # between two walkers, for our scorer class
    distance = PairDistance()

    # make a WExplore2 resampler with default parameters and our
    # distance metric
    resampler = WExplore1Resampler(max_region_sizes=(0.5, 0.5, 0.5, 0.5),
                                   max_n_regions=(10, 10, 10, 10),
                                   distance=distance,
                                   pmin=1e-12, pmax=0.5)

    ubc = UnbindingBC(cutoff_distance=2.0,
                      initial_state=init_walkers[0].state,
                      topology=mdtraj_topology,
                      ligand_idxs=np.array(test_sys.ligand_indices),
                      binding_site_idxs=np.array(test_sys.receptor_indices))

    json_top_path = 'pair.top.json'
    with open(json_top_path, 'r') as rf:
        json_str_top = rf.read()


    # the work mapper
    work_mapper = WorkerMapper(num_workers=n_workers)

    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=ubc,
                          work_mapper=work_mapper
                         )


    for run_idx in range(n_runs):
        print("Starting run: {}".format(run_idx))
        sim_manager.run_simulation(n_cycles, steps, debug_prints=True)
        print("Finished run: {}".format(run_idx))


if __name__ == "__main__":

    import time

    n_runs = int(sys.argv[1])
    n_cycles = int(sys.argv[2])
    n_steps = int(sys.argv[3])
    n_workers = int(sys.argv[4])

    # if you pass a seed use it
    try:
        seed = int(sys.argv[5])
    except IndexError:
        seed = None

    print("Number of steps: {}".format(n_steps))
    print("Number of cycles: {}".format(n_cycles))

    steps = [n_steps for i in range(n_cycles)]

    start = time.time()
    main(n_runs, n_cycles, steps, n_workers, seed=seed, debug_prints=True)
    end = time.time()

    print("time {}".format(end-start))

