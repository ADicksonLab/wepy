"""Module for generating wepy systems"""

from copy import copy, deepcopy

import numpy as np

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

from wepy.util.mdtraj import mdtraj_to_json_topology
from wepy.util.json_top import (json_top_residue_fields,
                                json_top_residue_df,
                                json_top_atom_df,
                                json_top_subset)


# OpenMM helpers
from wepy.runners.openmm import OpenMMRunner, OpenMMState
from wepy.walker import Walker
from wepy.runners.openmm import UNIT_NAMES, GET_STATE_KWARG_DEFAULTS
from wepy.orchestration.snapshot import WepySimApparatus, SimSnapshot
from wepy.orchestration.configuration import Configuration
from wepy.orchestration.orchestrator import Orchestrator

# resamplers
from wepy.resampling.resamplers.resampler import NoResampler
from wepy.resampling.resamplers.wexplore import WExploreResampler
from wepy.resampling.resamplers.revo import REVOResampler

# boundary conditions
from wepy.boundary_conditions.boundary import NoBC, RandomBC

# integrators
from simtk.openmm import LangevinIntegrator

# mappers
from wepy.work_mapper.mapper import Mapper
from wepy.work_mapper.worker import WorkerMapper, Worker
from wepy.work_mapper.task_mapper import TaskMapper, WalkerTaskProcess

from wepy.runners.openmm import (
    OpenMMCPUWorker, OpenMMGPUWorker,
    OpenMMCPUWalkerTaskProcess, OpenMMGPUWalkerTaskProcess,
)

# reporters
from wepy.reporter.hdf5 import WepyHDF5Reporter

from wepy.reporter.dashboard import (
    DashboardReporter,
    ResamplerDashboardSection,
    RunnerDashboardSection,
    BCDashboardSection)
from wepy.reporter.wexplore.dashboard import WExploreDashboardSection
from wepy.reporter.revo.dashboard import REVODashboardSection
from wepy.reporter.openmm import OpenMMRunnerDashboardSection

from wepy.reporter.restree import ResTreeReporter
from wepy.reporter.walker import WalkerReporter


# extras for reporters
from wepy.runners.openmm import UNIT_NAMES

# workers

## Resamplers

RESAMPLERS = [NoResampler, WExploreResampler, REVOResampler,]

WEXPLORE_DEFAULTS = {
    'pmax' : 0.5,
    'pmin' : 1e-12,
    'max_n_regions' : (10, 10, 10, 10),
    'max_region_sizes' : (1, 0.5, 0.35, 0.25),
}


REVO_DEFAULTS = {
    'pmax' : 0.5,
    'pmin' : 1e-12,
    'char_dist' : 1,
    'merge_dist' : 2.5,
    'dist_exponent' : 4,
    'weights' : True,
}

DEFAULT_RESAMPLER_PARAMS = {
    'NoResampler' : {},
    'WExploreResampler' : WEXPLORE_DEFAULTS,
    'REVOResampler' : REVO_DEFAULTS,
}



## BCs

BCS = [NoBC, RandomBC,]

DEFAULT_BC_PARAMS = {
    'NoBC' : {},
    'RandomBC' : {},
}


## Integrators

INTEGRATORS = [LangevinIntegrator,]

#(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)
LANGEVIN_DEFAULTS = (
    300.0*unit.kelvin,
    1/unit.picosecond,
    0.002*unit.picoseconds,
)

DEFAULT_INTEGRATOR_PARAMS = {
    'LangevinIntegrator' : LANGEVIN_DEFAULTS,
}

# these are used just to generate states
INTEGRATOR_FIXTURE = omm.LangevinIntegrator
INTEGRATOR_FIXTURE_PARAMS = DEFAULT_INTEGRATOR_PARAMS['LangevinIntegrator']

## OpenMM platforms

DEFAULT_PLATFORM_PARAMS = {
    'Reference' : {},
    'CPU' : {},
    'OpenCL' : {},
    'CUDA' : {},

}


## Work Mappers
MAPPERS = [Mapper, WorkerMapper, TaskMapper]

DEFAULT_MAPPER_PARAMS = {
    'Mapper' : {},
    'WorkerMapper' : {},
    'TaskMapper' : {},
}


## Reporters
REPORTERS = [
    WepyHDF5Reporter,
    DashboardReporter,
    ResTreeReporter,
    WalkerReporter,
]

WEPY_HDF5_REPORTER_DEFAULTS = {
                    'main_rep_idxs' : None,
                    'save_fields' : None,
                    'units' : dict(UNIT_NAMES),
                    'sparse_fields' : {'velocities' : 10},
                    'all_atoms_rep_freqs' : 10,
                    'alt_reps' : None,
                    'swmr_mode' : True,
                }

RESTREE_REPORTER_DEFAULTS = {
    'node_radius' : 3.0,
    'row_spacing' : 5.0,
    'step_spacing' : 20.0,
    'colormap_name' : 'plasma',
}

DEFAULT_REPORTER_PARAMS = {
    'WepyHDF5Reporter' : WEPY_HDF5_REPORTER_DEFAULTS,
    'DashboardReporter' : {},
    'ResTreeReporter' : RESTREE_REPORTER_DEFAULTS,
    'WalkerReporter' : {},
}

class OpenMMSimMaker():

    def __init__(self,
                 distance=None,
                 init_state=None,
                 system=None,
                 topology=None,
    ):

        self.distance = distance
        self.init_state = init_state
        self.system = system
        self.topology = topology

    @classmethod
    def make_state(cls, system, positions):

        # a temporary integrator just for this
        integrator = INTEGRATOR_FIXTURE(*INTEGRATOR_FIXTURE_PARAMS)

        # make a context and set the positions
        _context = omm.Context(system, copy(integrator))
        _context.setPositions(positions)

        # get the data from this context so we have a state to start the
        # simulation with
        _get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
        init_sim_state = _context.getState(**_get_state_kwargs)
        init_state = OpenMMState(init_sim_state)

        return init_state

    @classmethod
    def make_initial_walkers(cls, state, n_walkers):

        init_weight = 1.0 / n_walkers
        init_walkers = [Walker(deepcopy(state), init_weight) for i in range(n_walkers)]

        return init_walkers

    def make_apparatus(self,
                       platform='Reference',
                       platform_params=None,
                       integrator='LangevinIntegrator',
                       integrator_params=None,
                       resampler='WExploreResampler',
                       resampler_params=None,
                       bc=None,
                       bc_params=None,
    ):

        ## RUNNER

        # choose which integrator to use
        integrator_class = [i for i in INTEGRATORS
                            if i.__name__ == integrator][0]

        integrator_name = integrator_class.__name__


        # use either the default params or the user params
        if integrator_params is None:
            integrator_params = DEFAULT_INTEGRATOR_PARAMS[integrator_name]

        integrator = integrator_class(*integrator_params)

        # TODO: not handling the params here
        if platform_params is None:
            platform_params = DEFAULT_PLATFORM_PARAMS[platform]

        # make the runner for the test system
        runner = OpenMMRunner(self.system,
                              self.topology,
                              integrator,
                              platform=platform)


        # RESAMPLER

        # choose which resampler to use
        resampler_class = [res for res in RESAMPLERS
                           if res.__name__ == resampler][0]

        resampler_name = resampler_class.__name__

        # use either the default params or the user params
        if resampler_params is None:
            resampler_params = DEFAULT_RESAMPLER_PARAMS[resampler_name]


        resampler = resampler_class(distance=self.distance,
                                   init_state=self.init_state,
                                   **resampler_params)


        # BOUNDARY_CONDITIONS

        # you don't have to specify a boundary condition
        if bc is not None:

            # choose which bc to use
            bc_class = [res for res in BCS
                               if res.__name__ == bc][0]

            bc_name = bc_class.__name__

            # use either the default params or the user params
            if bc_params is None:
                bc_params = DEFAULT_BC_PARAMS[bc_name]


            bc = bc_class(distance=self.distance,
                                       init_state=self.init_state,
                                       **bc_params)


        # APPARATUS

        # build the apparatus
        sim_apparatus = WepySimApparatus(runner, resampler=resampler,
                                         boundary_conditions=bc)

        return sim_apparatus

    @classmethod
    def choose_work_mapper_platform_params(cls, platform, mapper_name):

        work_mapper_params = {}

        if mapper_name == 'WorkerMapper':

            if platform == 'Reference':
                worker_type = Worker
            elif platform == 'CPU':
                worker_type = OpenMMCPUWorker
            elif platform in ('CUDA', 'OpenCL',):
                worker_type = OpenMMGPUWorker
            else:
                worker_type = Worker

            work_mapper_params['worker_type'] = worker_type

        elif mapper_name == 'TaskMapper':
            if platform == 'Reference':
                worker_type = Worker
            elif platform == 'CPU':
                worker_type = OpenMMCPUWalkerTaskProcess
            elif platform in ('CUDA', 'OpenCL',):
                worker_type = OpenMMGPUWalkerTaskProcess
            else:
                worker_type = Worker

            work_mapper_params['worker_task_type'] = worker_type

        return work_mapper_params

    def resolve_reporter_params(self, apparatus, reporter_specs, reporters_kwargs=None):

        if reporters_kwargs is not None:
            raise NotImplementedError("Only the defaults are supported currently")

        if reporter_specs is None:
            # defaults to use
            reporter_specs = [
                'WepyHDF5Reporter',
                'DashboardReporter',
                # DEBUG: this isn't compatible right now, needs refactoring
                #'ResTreeReporter',
                'WalkerReporter',
            ]


        # augment the dashboard with the sections relevant to our components
        dashboard_sections = {'resampler' : ResamplerDashboardSection(apparatus.resampler),
                              'runner' : RunnerDashboardSection(apparatus.runner),
                              'bc' : BCDashboardSection(apparatus.boundary_conditions),
        }

        if 'DashboardReporter' in reporter_specs:

            ## resampler
            # WExplore
            if type(apparatus.resampler).__name__ == 'WExploreResampler':
                dashboard_sections['resampler'] = WExploreDashboardSection(apparatus.resampler)
            # REVO
            elif type(apparatus.resampler).__name__ == 'REVOResampler':
                dashboard_sections['resampler'] = REVODashboardSection(apparatus.resampler)

            ## BC
            # NoBC
            if type(apparatus.boundary_conditions).__name__ == 'NoBC':
                dashboard_sections['bc'] = BCDashboardSection(apparatus.boundary_conditions)
            # Random
            elif type(apparatus.boundary_conditions).__name__ == 'RandomBC':
                dashboard_sections['bc'] = BCDashboardSection(apparatus.boundary_conditions)

            ## Runner

            #OpenMM
            if type(apparatus.runner).__name__ == 'OpenMMRunner':
                dashboard_sections['runner'] = OpenMMRunnerDashboardSection(
                    apparatus.runner)



        # get the actual classes
        reporter_classes = []
        for reporter_spec in reporter_specs:
            match = False
            for reporter_class in REPORTERS:

                if reporter_class.__name__ == reporter_spec:
                    match = reporter_class
                    break

            if not match:
                raise ValueError("Unkown reporter for spec {}".format(reporter_spec))

            reporter_classes.append(match)


        # then get the default params for them and commensurate them
        # with the given kwargs
        reporters_params = []
        for reporter_spec in reporter_specs:
            reporter_params = {}

            if reporter_spec == 'WepyHDF5Reporter':

                # always set these ones automatically
                auto_params = {
                    'topology' : self.json_top(),
                    'resampler' : apparatus.resampler,
                    'boundary_conditions' : apparatus.boundary_conditions,
                }

                reporter_params.update(auto_params)
                reporter_params.update(deepcopy(DEFAULT_REPORTER_PARAMS[reporter_spec]))

            elif reporter_spec == 'DashboardReporter':

                # always set these ones automatically
                auto_params = {
                    # the 'getStepSize' method is an abstract one for
                    # all integrators so we can rely on it being here.
                    'step_time' : apparatus.runner.integrator.getStepSize(),

                    # the dashboard sections
                    'resampler_dash' : dashboard_sections['resampler'],
                    'runner_dash' : dashboard_sections['runner'],
                    'bc_dash' : dashboard_sections['bc'],
                }

                reporter_params.update(auto_params)
                reporter_params.update(deepcopy(DEFAULT_REPORTER_PARAMS[reporter_spec]))

            elif reporter_spec == 'WalkerReporter':

                # always set these ones automatically
                auto_params = {
                    'json_topology' : self.json_top(),
                    'init_state' : self.init_state,
                }

                reporter_params.update(auto_params)
                reporter_params.update(deepcopy(DEFAULT_REPORTER_PARAMS[reporter_spec]))

            elif reporter_spec == 'ResTreeReporter':

                # always set these ones automatically
                auto_params = {
                    'resampler' : apparatus.resampler,
                    'boundary_condition' : apparatus.boundary_conditions,
                }

                reporter_params.update(auto_params)
                reporter_params.update(deepcopy(DEFAULT_REPORTER_PARAMS[reporter_spec]))

            else:
                reporter_params.update(deepcopy(DEFAULT_REPORTER_PARAMS[reporter_spec]))

            # add them to the list for this reporter
            reporters_params.append(reporter_params)

        return reporter_classes, reporters_params


    def make_configuration(self,
                           apparatus,
                           work_mapper='TaskMapper',
                           work_mapper_params=None,
                           platform='Reference',
                           reporters=None,
                           reporter_kwargs=None,
                           work_dir=None,
    ):

        # MAPPER

        # choose which mapper to use
        work_mapper_class = [mapper for mapper in MAPPERS
                           if mapper.__name__ == work_mapper][0]

        mapper_name = work_mapper_class.__name__

        # use either the default params or the user params
        if work_mapper_params is None:
            work_mapper_params = DEFAULT_MAPPER_PARAMS[mapper_name]

        # depending on the platform and work mapper choose the worker
        # type and update the params in place
        work_mapper_params.update(
            self.choose_work_mapper_platform_params(platform, mapper_name))


        # REPORTERS

        reporter_classes, reporter_params = \
                        self.resolve_reporter_params(apparatus, reporters, reporter_kwargs)


        config = Configuration(
            work_mapper_class=work_mapper_class,
            work_mapper_partial_kwargs=work_mapper_params,
            reporter_classes=reporter_classes,
            reporter_partial_kwargs=reporter_params,
            work_dir=work_dir
        )

        return config


    def make_sim_manager(self, n_walkers, apparatus, config):

        walkers = self.make_initial_walkers(self.init_state, n_walkers)
        snapshot = SimSnapshot(walkers, apparatus)

        sim_manager = Orchestrator.gen_sim_manager(snapshot, config)

        return sim_manager



class OpenMMToolsTestSysSimMaker(OpenMMSimMaker):

    TEST_SYS = None

    @classmethod
    def num_atoms(cls):

        json_top = cls.json_top()

        # get the atom dataframe and select them from the ligand residue
        return len(json_top_atom_df(json_top))

    @classmethod
    def box_vectors(cls):

        # just munge the thing they give you to be a nice array Quantity
        bvs = cls.TEST_SYS().system.getDefaultPeriodicBoxVectors()

        return np.array([bv.value_in_unit(unit.nanometer) for bv in bvs]) * unit.nanometer

    @classmethod
    def json_top(cls):

        test_sys = cls.TEST_SYS()

        # convert to a JSON top
        json_top = mdtraj_to_json_topology(mdj.Topology.from_openmm(test_sys.topology))

        return json_top
