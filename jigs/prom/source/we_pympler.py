"""Very simple example using a pair of Lennard-Jones particles.

This script has several pieces to pay attention to:

- Importing the pieces from wepy to run a WExplore simulation.

- Definition of a distance metric for this system and process.

- Definition of the components used in the simulation: resampler,
boundary conditions, runner.

- Definition of the reporters which will write the data out.

- Create the work mapper for a non-parallel run.

- Construct the simulation manager with all the parts.

- Actually run the simulation.

"""

from pympler.asizeof import asizeof
import prometheus_client as prom

prom.start_http_server(9001)

class SimMonitor:

    def __init__(self,
                 reporter_order=(
                     'hdf5_reporter',
                     'dashboard_reporter',
                 ),
    ):

        self.reporter_order = reporter_order

        # counters
        self.cycle_counter = prom.Counter('wepy_cycle_idx', "")

        # gauges
        self.walker_size_g = prom.Gauge('wepy_walker_single_size_bytes', "")
        self.ensemble_size_g = prom.Gauge('wepy_walker_ensemble_size_bytes', "")

        self.runner_size_g = prom.Gauge('wepy_runner_size_bytes', "")
        self.resampler_size_g = prom.Gauge('wepy_resampler_size_bytes', "")
        self.bc_size_g = prom.Gauge('wepy_bc_size_bytes', "")
        self.mapper_size_g = prom.Gauge('wepy_mapper_size_bytes', "")

        self.sim_manager_size_g = prom.Gauge('wepy_sim_manager_size_bytes', "")

        self.reporter_size_g = prom.Gauge('wepy_reporters_size_bytes',
                                          "",
                                          ["name"],
        )

        # DEBUG: testing reporters another way
        # self.reporter_size_gs = {}
        # for reporter_name in self.reporter_order:
        #     self.reporter_size_gs[reporter_name] = prom.Gauge(
        #         f'wepy_reporter_size_bytes',
        #         "",
        #     )


    def cycle_monitor(self, sim_manager, walkers):

        # increment the cycle counter
        self.cycle_counter.inc()

        # get the sizes of the objects
        walker_size = asizeof(walkers[0])
        ensemble_size = asizeof(walkers)

        runner_size = asizeof(sim_manager.runner)
        resampler_size = asizeof(sim_manager.resampler)
        bc_size = asizeof(sim_manager.boundary_conditions)
        mapper_size = asizeof(sim_manager.work_mapper)
        sim_manager_size = asizeof(sim_manager)

        reporter_sizes = {}
        for idx, reporter_name in enumerate(self.reporter_order):
            reporter_sizes[reporter_name] = asizeof(sim_manager.reporters[idx])

        # then update the gauges
        self.walker_size_g.set(walker_size)
        self.ensemble_size_g.set(ensemble_size)

        self.runner_size_g.set(runner_size)
        self.resampler_size_g.set(resampler_size)
        self.bc_size_g.set(bc_size)
        self.mapper_size_g.set(mapper_size)

        self.sim_manager_size_g.set(sim_manager_size)

        for reporter_name in self.reporter_order:
            self.reporter_size_g.labels(reporter_name).set(reporter_sizes[reporter_name])

def get_size(obj):
    """get the size in units of Mb"""

    return asizeof(obj) / 1000000


import sys

from copy import copy
import os
import os.path as osp

import numpy as np

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from openmm_systems.test_systems import LennardJonesPair
import mdtraj as mdj
from wepy.util.mdtraj import mdtraj_to_json_topology

from wepy.sim_manager import Manager

from wepy.resampling.distances.distance import Distance
from wepy.resampling.resamplers.wexplore import WExploreResampler
from wepy.walker import Walker
from wepy.runners.openmm import OpenMMRunner, OpenMMState
from wepy.runners.openmm import UNIT_NAMES, GET_STATE_KWARG_DEFAULTS
from wepy.work_mapper.mapper import Mapper
from wepy.boundary_conditions.receptor import UnbindingBC
from wepy.reporter.hdf5 import WepyHDF5Reporter

from wepy.reporter.dashboard import DashboardReporter

from wepy.reporter.receptor.dashboard import ReceptorBCDashboardSection
from wepy.reporter.wexplore.dashboard import WExploreDashboardSection
from wepy.reporter.openmm import OpenMMRunnerDashboardSection

from scipy.spatial.distance import euclidean


## PARAMETERS

# Platform used for OpenMM which uses different hardware computation
# kernels. Options are: Reference, CPU, OpenCL, CUDA.

# we use the Reference platform because this is just a test
PLATFORM = 'Reference'

# Langevin Integrator
TEMPERATURE= 300.0*unit.kelvin
FRICTION_COEFFICIENT = 1/unit.picosecond
# step size of time integrations
STEP_SIZE = 0.002*unit.picoseconds

# Resampler parameters

# the maximum weight allowed for a walker
PMAX = 0.5
# the minimum weight allowed for a walker
PMIN = 1e-12

# the maximum number of regions allowed under each parent region
MAX_N_REGIONS = (10, 10, 10, 10)

# the maximum size of regions, new regions will be created if a walker
# is beyond this distance from each voronoi image unless there is an
# already maximal number of regions
MAX_REGION_SIZES = (1, 0.5, .35, .25) # nanometers

# boundary condition parameters

# maximum distance between between any atom of the ligand and any
# other atom of the protein, if the shortest such atom-atom distance
# is larger than this the ligand will be considered unbound and
# restarted in the initial state
CUTOFF_DISTANCE = 1.0 # nm

# reporting parameters

# these are the properties of the states (i.e. from OpenMM) which will
# be saved into the HDF5
SAVE_FIELDS = ('positions', 'box_vectors', 'velocities')

## INPUTS/OUTPUTS

# the inputs directory
inputs_dir = osp.realpath('input')

# the outputs path
outputs_dir = osp.realpath('_output/we')

# make the outputs dir if it doesn't exist
os.makedirs(outputs_dir, exist_ok=True)

# inputs filenames
json_top_filename = "pair.top.json"

# outputs
hdf5_filename = 'results.wepy.h5'
dashboard_filename = 'wepy.dash.org.org'

# normalize the output paths
hdf5_path = osp.join(outputs_dir, hdf5_filename)
dashboard_path = osp.join(outputs_dir, dashboard_filename)

## System and OpenMMRunner

# make the test system
test_sys = LennardJonesPair()

# make the integrator
integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

# make a context and set the positions
context = omm.Context(test_sys.system, copy(integrator))
context.setPositions(test_sys.positions)

# get the data from this context so we have a state to start the
# simulation with
get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
init_sim_state = context.getState(**get_state_kwargs)
init_state = OpenMMState(init_sim_state)

# initialize the runner
runner = OpenMMRunner(test_sys.system, test_sys.topology, integrator, platform=PLATFORM)


## Distance Metric
# we define a simple distance metric for this system, assuming the
# positions are in a 'positions' field
class PairDistance(Distance):

    def __init__(self, metric=euclidean):
        self.metric = metric

    def image(self, state):
        return state['positions']

    def image_distance(self, image_a, image_b):
        dist_a = self.metric(image_a[0], image_a[1])
        dist_b = self.metric(image_b[0], image_b[1])

        return np.abs(dist_a - dist_b)


# make a distance object which can be used to compute the distance
# between two walkers, for our scorer class
distance = PairDistance()

## Resampler
resampler = WExploreResampler(distance=distance,
                               init_state=init_state,
                               max_region_sizes=MAX_REGION_SIZES,
                               max_n_regions=MAX_N_REGIONS,
                               pmin=PMIN, pmax=PMAX)

# the mdtraj here is needed for the distance function
mdtraj_topology = mdj.Topology.from_openmm(test_sys.topology)
json_str_top = mdtraj_to_json_topology(mdtraj_topology)

# initialize the unbinding boundary conditions
ubc = UnbindingBC(cutoff_distance=CUTOFF_DISTANCE,
                  initial_state=init_state,
                  topology=json_str_top,
                  ligand_idxs=np.array(test_sys.ligand_indices),
                  receptor_idxs=np.array(test_sys.receptor_indices))

## Reporters

# make a dictionary of units for adding to the HDF5
units = dict(UNIT_NAMES)

# open it in truncate mode first, then switch after first run
hdf5_reporter = WepyHDF5Reporter(file_path=hdf5_path, mode='w',
                                 save_fields=SAVE_FIELDS,
                                 resampler=resampler,
                                 boundary_conditions=ubc,
                                 topology=json_str_top,
                                 units=units)

wexplore_dash = WExploreDashboardSection(resampler=resampler)
openmm_dash = OpenMMRunnerDashboardSection(runner=runner,
                                           step_time=STEP_SIZE)
ubc_dash = ReceptorBCDashboardSection(bc=ubc)

dashboard_reporter = DashboardReporter(
    file_path=dashboard_path,
    mode='w',
    resampler_dash=wexplore_dash,
    runner_dash=openmm_dash,
    bc_dash=ubc_dash,
)

reporter_order = ("hdf5_reporter", "dashboard_reporter")
reporters = [hdf5_reporter, dashboard_reporter]

## Work Mapper

# a simple work mapper
mapper = Mapper()

## Run the simulation


if __name__ == "__main__":

    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("arguments: n_cycles, n_steps, n_walkers")
    else:
        n_cycles = int(sys.argv[1])
        n_steps = int(sys.argv[2])
        n_walkers = int(sys.argv[3])

        print("Number of steps: {}".format(n_steps))
        print("Number of cycles: {}".format(n_cycles))

        # create the initial walkers
        init_weight = 1.0 / n_walkers
        init_walkers = [Walker(OpenMMState(init_sim_state), init_weight) for i in range(n_walkers)]


        # initialize the simulation manager
        sim_manager = Manager(init_walkers,
                              runner=runner,
                              resampler=resampler,
                              boundary_conditions=ubc,
                              work_mapper=mapper,
                              reporters=reporters)


        # make a number of steps for each cycle. In principle it could be
        # different each cycle
        steps = [n_steps for i in range(n_cycles)]

        def print_sim_objs():

            print("init_walkers: ", get_size(init_walkers))

            print("runner: ", get_size(runner))
            print("resampler: ", get_size(resampler))
            print("ubc: ", get_size(ubc))
            print("mapper: ", get_size(mapper))

            print("hdf5_reporter: ", get_size(hdf5_reporter))
            print("dashboard_reporter: ", get_size(dashboard_reporter))

            print("sim_manager: ", get_size(sim_manager))
            print("steps: ", get_size(steps))

        sim_monitor = SimMonitor(reporter_order=reporter_order)


        # actually run the simulation
        print("Starting run")
        print("----------------------------------------")
        print_sim_objs()
        print("----------------------------------------\n")

        sim_manager.run_simulation(n_cycles, steps,
                                   sim_monitor=sim_monitor)

        print("Finished run")
        print("----------------------------------------")
        print_sim_objs()
        print("----------------------------------------\n")
