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
from wepy.resampling.scoring.scorer import AllToAllScorer
from wepy.resampling.wexplore2 import WExplore2Resampler

from wepy.runners.openmm import OpenMMRunner, OpenMMWalker
from wepy.runners.openmm import UNIT_NAMES, GET_STATE_KWARG_DEFAULTS
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.reporter.reporter import WalkersPickleReporter


from scipy.spatial.distance import euclidean

# we define a simple distance metric for this system, assuming the
# positions are in a 'positions' field
class PairDistance(Distance):
    def __init__(self, metric=euclidean):
        self.metric = metric

    def distance(self, walker_a, walker_b):

        dist_a = self.metric(walker_a['positions'][0], walker_a['positions'][1])
        dist_b = self.metric(walker_b['positions'][0], walker_b['positions'][1])

        return np.abs(dist_a - dist_b)

if __name__ == "__main__":

    n_runs = int(sys.argv[1])
    n_steps = int(sys.argv[2])
    n_cycles = int(sys.argv[3])

    test_sys = LennardJonesPair()

    #integrator = omm.VerletIntegrator(2*unit.femtoseconds)
    integrator = omm.LangevinIntegrator(300.0*unit.kelvin, 1/unit.picosecond, 2*unit.femtoseconds)
    context = omm.Context(test_sys.system, copy(integrator))
    context.setPositions(test_sys.positions)

    get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
    init_state = context.getState(**get_state_kwargs)

    thermostat = omm.AndersenThermostat(300.0 * unit.kelvin, 1/unit.picosecond)
    barostat = omm.MonteCarloBarostat(1.0*unit.atmosphere, 300.0*unit.kelvin, 50)

    runner = OpenMMRunner(test_sys.system, test_sys.topology, integrator, platform='Reference')

    num_walkers = 10
    init_weight = 1.0 / num_walkers

    init_walkers = [OpenMMWalker(init_state, init_weight) for i in range(num_walkers)]


    # the mdtraj here is needed for the distance function
    mdtraj_topology = mdj.Topology.from_openmm(test_sys.topology)

    # make a distance object which can be used to compute the distance
    # between two walkers, for our scorer class
    distance = PairDistance()

    # we need a scorer class to perform the all-to-all distances
    # using our distance class
    scorer = AllToAllScorer(distance=distance)

    # make a WExplore2 resampler with default parameters and our
    # distance metric
    resampler = WExplore2Resampler(scorer=scorer,
                                   pmax=0.5)

    ubc = UnbindingBC(cutoff_distance=0.5,
                      initial_state=init_walkers[0].state,
                      topology=mdtraj_topology,
                      ligand_idxs=np.array(test_sys.ligand_indices),
                      binding_site_idxs=np.array(test_sys.receptor_indices))

    json_top_path = 'pair.top.json'
    with open(json_top_path, 'r') as rf:
        json_str_top = rf.read()

    # make a dictionary of units for adding to the HDF5
    units = dict(UNIT_NAMES)

    report_path = 'results.wepy.h5'
    # open it in truncate mode first, then switch after first run
    hdf5_reporter = WepyHDF5Reporter(report_path, mode='w',
                                    save_fields=['positions', 'box_vectors', 'velocities'],
                                    decisions=resampler.DECISION.ENUM,
                                    instruction_dtypes=resampler.DECISION.instruction_dtypes(),
                                    warp_dtype=ubc.WARP_INSTRUCT_DTYPE,
                                    warp_aux_dtypes=ubc.WARP_AUX_DTYPES,
                                    warp_aux_shapes=ubc.WARP_AUX_SHAPES,
                                    topology=json_str_top,
                                    units=units,
                                    sparse_fields={'velocities' : 10},
                                    # sparse atoms fields
                                    main_rep_idxs=[0],
                                    all_atoms_rep_freq=10,
                                    alt_reps={'other_atom' : ([1], 2)}
    )

    pkl_reporter = WalkersPickleReporter(save_dir='./pickle_backups', freq=10, num_backups=3)

    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=ubc,
                          work_mapper=map,
                          reporters=[hdf5_reporter, pkl_reporter])

    print("Number of steps: {}".format(n_steps))
    print("Number of cycles: {}".format(n_cycles))

    steps = [n_steps for i in range(n_cycles)]
    print("Running Simulations")

    for run_idx in range(n_runs):
        print("Starting run: {}".format(run_idx))
        sim_manager.run_simulation(n_cycles, steps, debug_prints=True)
        print("Finished run: {}".format(run_idx))

    print("Finished first file")
