import pytest

from pathlib import Path
import os.path as osp
import importlib
from copy import copy
import pickle

import numpy as np
from scipy.spatial.distance import euclidean

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

from openmm_systems.test_systems import LennardJonesPair

from wepy.sim_manager import Manager

### Apparatus

## Resampler
from wepy.resampling.distances.distance import Distance
from wepy.resampling.resamplers.wexplore import WExploreResampler
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.resampling.resamplers.resampler import NoResampler

## Boundary Conditions
from wepy.boundary_conditions.unbinding import UnbindingBC

## Runner
from wepy.runners.openmm import (
    OpenMMRunner,
    OpenMMState,
    OpenMMWalker,
    UNIT_NAMES,
    GET_STATE_KWARG_DEFAULTS,
    gen_walker_state,
)

## Initial Walkers
from wepy.walker import Walker

### Configuration

## Reporters
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.reporter.restree import ResTreeReporter
from wepy.reporter.dashboard import DashboardReporter

## Work Mappers
from wepy.work_mapper.mapper import Mapper
from wepy.work_mapper.mapper import WorkerMapper
from wepy.work_mapper.task_mapper import TaskMapper

### Utilities
from wepy.util.mdtraj import mdtraj_to_json_topology

### Orchestration
from wepy.orchestration.configuration import Configuration
from wepy.orchestration.orchestrator import Orchestrator, reconcile_orchestrators
from wepy.orchestration.snapshot import WepySimApparatus, SimSnapshot

### Mock Systems

from wepy_tools.sim_makers.openmm.sim_maker import OpenMMSimMaker
from wepy_tools.systems.lennard_jones import PairDistance


### Constants

# only use the reference platform for python-only integration testing
# purposes
PLATFORM = "Reference"

### Sanity Test
@pytest.fixture(scope='class')
def lj_sanity_test():
    """Sanity test to make sure we even have the plugin fixtures installed."""
    return "sanity"

### Fixtures


## OpenMM Misc.

@pytest.fixture(scope='class')
def lj_omm_sys():
    return LennardJonesPair()

@pytest.fixture(scope='class')
def langevin_integrator():

    integrator = omm.LangevinIntegrator(
        *OpenMMSimMaker.DEFAULT_INTEGRATOR_PARAMS['LangevinIntegrator']
    )

    return integrator

integrators = [
    langevin_integrator,
]

@pytest.fixture(
    scope='class',
    params=[
        'LangevinIntegrator',
    ]
)
def lj_integrator(
        request,
        *integrators,
):

    intgr_spec = request.param
    if intgr_spec == "LangevinIntegrator":
        return langevin_integrator
    else:
        raise ValueError("Unkown integrator")


## Runner

@pytest.fixture(
    scope='class',
    params=[
        'Reference',
    ]
)
def lj_openmm_runner(
        request,
        lj_omm_sys,
        lj_integrator
):

    # parametrize the platform
    platform = request.param

    positions = test_sys.positions.value_in_unit(test_sys.positions.unit)

    init_state = gen_walker_state(
        positions,
        test_sys.system,
        integrator)

    # initialize the runner
    runner = OpenMMRunner(
        lj_omm_sys.system,
        lj_omm_sys.topology,
        lj_integrator,
        platform=platform)

    return runner

## Resampler

# @pytest.fixture(scope='class')
# def lj_distance_metric():
#     return PairDistance()


# @pytest.fixture(scope='class')
# def lj_wexplore_resampler(lj_params, lj_distance_metric, lj_init_state):
#     resampler = WExploreResampler(distance=lj_distance_metric,
#                                    init_state=lj_init_state,
#                                    max_region_sizes=lj_params['max_region_sizes'],
#                                    max_n_regions=lj_params['max_n_regions'],
#                                    pmin=lj_params['pmin'], pmax=lj_params['pmax'])

#     return resampler

# @pytest.fixture(scope='class')
# def lj_revo_resampler(lj_params, lj_distance_metric, lj_init_state):
#     resampler = REVOResampler(distance=lj_distance_metric,
#                               merge_dist=2.5,
#                               char_dist=1.0,
#                               init_state=lj_init_state,
#                               pmin=lj_params['pmin'], pmax=lj_params['pmax'])

#     return resampler

# @pytest.fixture(scope='class')
# def lj_topology(lj_omm_sys):

#     # the mdtraj here is needed for the distance function
#     mdtraj_topology = mdj.Topology.from_openmm(lj_omm_sys.topology)

#     ## Reporters if we want them
#     json_str_top = mdtraj_to_json_topology(mdtraj_topology)

#     return json_str_top


# @pytest.fixture(scope='class')
# def lj_unbinding_bc(lj_params, lj_init_state, lj_topology, lj_omm_sys):

#     # initialize the unbinding boundary condition
#     ubc = UnbindingBC(cutoff_distance=lj_params['cutoff_distance'],
#                       initial_state=lj_init_state,
#                       topology=lj_topology,
#                       ligand_idxs=np.array(lj_omm_sys.ligand_indices),
#                       receptor_idxs=np.array(lj_omm_sys.receptor_indices))

#     return ubc

# @pytest.fixture(scope='class')
# def lj_reporter_kwargs(lj_params, lj_topology, lj_wexplore_resampler, lj_unbinding_bc):
#     """Reporters that work for all of the components."""

#     # make a dictionary of units for adding to the HDF5
#     units = dict(UNIT_NAMES)

#     hdf5_reporter_kwargs = {'save_fields' : lj_params['save_fields'],
#                             'resampler' : lj_wexplore_resampler,
#                             'boundary_conditions' : lj_unbinding_bc,
#                             'topology' : lj_topology,
#                             'units' : units,
#     }

#     dashboard_reporter_kwargs = {'step_time' : lj_params['step_size'].value_in_unit(unit.second),
#                                  'bc_cutoff_distance' : lj_unbinding_bc.cutoff_distance}

#     # Resampling Tree
#     restree_reporter_kwargs = {'resampler' : lj_wexplore_resampler,
#                                'boundary_condition' : lj_unbinding_bc,
#                                'node_radius' : 3.0,
#                                'row_spacing' : 5.0,
#                                'step_spacing' : 20.0,
#                                'progress_key' : 'min_distances',
#                                'max_progress_value' : lj_unbinding_bc.cutoff_distance,
#                                'colormap_name' : 'plasma'}


#     reporter_kwargs = [hdf5_reporter_kwargs, dashboard_reporter_kwargs,
#                        restree_reporter_kwargs]

#     return reporter_kwargs


# @pytest.fixture(scope='class')
# def lj_reporter_classes():
#     reporter_classes = [WepyHDF5Reporter, DashboardReporter,
#                         ResTreeReporter]

#     return reporter_classes

# @pytest.fixture(scope='class')
# def lj_init_walkers(lj_params, lj_init_sim_state):
#     init_weight = 1.0 / lj_params['n_walkers']
#     init_walkers = [OpenMMWalker(OpenMMState(lj_init_sim_state), init_weight)
#                     for i in range(lj_params['n_walkers'])]

#     return init_walkers


# @pytest.fixture(scope='class')
# def lj_apparatus(lj_openmm_runner, lj_wexplore_resampler, lj_unbinding_bc):

#     sim_apparatus = WepySimApparatus(lj_openmm_runner, resampler=lj_wexplore_resampler,
#                                      boundary_conditions=lj_unbinding_bc)

#     return sim_apparatus

# @pytest.fixture(scope='class')
# def lj_null_apparatus(lj_openmm_runner):

#     sim_apparatus = WepySimApparatus(lj_openmm_runner, resampler=NoResampler())

#     return sim_apparatus

# @pytest.fixture(scope='class')
# def lj_snapshot(lj_init_walkers, lj_apparatus):

#     return SimSnapshot(lj_init_walkers, lj_apparatus)



# @pytest.fixture(scope='class')
# def lj_configuration(tmp_path_factory, lj_reporter_classes, lj_reporter_kwargs):

#     # make a temporary directory for this configuration to work with
#     tmpdir = str(tmp_path_factory.mktemp('lj_fixture'))
#     # tmpdir = tmp_path_factory.mktemp('lj_fixture/work_dir')

#     configuration = Configuration(work_dir=tmpdir,
#                                   reporter_classes=lj_reporter_classes,
#                                   reporter_partial_kwargs=lj_reporter_kwargs)

#     return configuration

# @pytest.fixture(scope='class')
# def lj_null_configuration(tmp_path_factory, lj_reporter_classes, lj_reporter_kwargs,
#                           lj_params, lj_wexplore_resampler, lj_topology):

#     reporter_classes = [WepyHDF5Reporter]

#     # make a dictionary of units for adding to the HDF5
#     units = dict(UNIT_NAMES)

#     hdf5_reporter_kwargs = {'save_fields' : lj_params['save_fields'],
#                             'resampler' : lj_wexplore_resampler,
#                             'topology' : lj_topology,
#                             'units' : units,
#     }

#     reporter_kwargs = [hdf5_reporter_kwargs]



#     # make a temporary directory for this configuration to work with
#     tmpdir = str(tmp_path_factory.mktemp('lj_fixture'))
#     # tmpdir = tmp_path_factory.mktemp('lj_fixture/work_dir')

#     configuration = Configuration(work_dir=tmpdir,
#                                   reporter_classes=reporter_classes,
#                                   reporter_partial_kwargs=reporter_kwargs)

#     return configuration

# @pytest.fixture(scope='class')
# def lj_inmem_configuration(tmp_path_factory):

#     # make a temporary directory for this configuration to work with
#     tmpdir = str(tmp_path_factory.mktemp('lj_fixture'))
#     # tmpdir = tmp_path_factory.mktemp('lj_fixture/work_dir')

#     configuration = Configuration(work_dir=tmpdir)

#     return configuration


# @pytest.fixture(scope='class')
# def lj_work_mapper(lj_configuration):

#     work_mapper = Mapper()

#     return work_mapper


# @pytest.fixture(scope='class')
# def lj_work_mapper_worker():

#     work_mapper = WorkerMapper(num_workers=1)

#     return work_mapper

# @pytest.fixture(scope='class')
# def lj_work_mapper_task():

#     work_mapper = TaskMapper(num_workers=1)

#     return work_mapper


# @pytest.fixture(scope='class')
# def lj_reporters(tmp_path_factory, lj_reporter_classes, lj_reporter_kwargs):


#     # make a temporary directory for this configuration to work with
#     tmpdir = str(tmp_path_factory.mktemp('lj_fixture'))

#     # make a config so that the reporters get parametrized properly
#     config = Configuration(work_dir=tmpdir,
#                            reporter_classes=lj_reporter_classes,
#                            reporter_partial_kwargs=lj_reporter_kwargs)

#     return config.reporters




# @pytest.fixture(scope='class')
# def lj_orchestrator(lj_apparatus, lj_init_walkers, lj_configuration):

#     # use an in memory database with sqlite

#     # make a path to the temporary directory for this orchestrator
#     # orch_path = tmp_path_factory.mktemp('lj_fixture/lj.orch.sqlite')

#     # then create the seed/root/master orchestrator which will be used
#     # from here on out
#     orch = Orchestrator()

#     return orch

# @pytest.fixture(scope='class')
# def lj_orchestrator_defaults(lj_orchestrator,
#                                  lj_apparatus, lj_init_walkers, lj_configuration):


#     lj_orchestrator.set_default_sim_apparatus(lj_apparatus)
#     lj_orchestrator.set_default_init_walkers(lj_init_walkers)
#     lj_orchestrator.set_default_configuration(lj_configuration)

#     lj_orchestrator.gen_default_snapshot()


#     return lj_orchestrator

# @pytest.fixture(scope='class')
# def lj_orchestrator_defaults_inmem(lj_orchestrator,
#                                        lj_apparatus, lj_init_walkers, lj_inmem_configuration):


#     lj_orchestrator.set_default_sim_apparatus(lj_apparatus)
#     lj_orchestrator.set_default_init_walkers(lj_init_walkers)
#     lj_orchestrator.set_default_configuration(lj_configuration)

#     lj_orchestrator.gen_default_snapshot()


#     return lj_orchestrator

# @pytest.fixture(scope='class')
# def lj_orchestrator_defaults_null(lj_orchestrator,
#                                   lj_null_apparatus, lj_init_walkers,
#                                   lj_null_configuration):

#     lj_orchestrator.set_default_sim_apparatus(lj_null_apparatus)
#     lj_orchestrator.set_default_init_walkers(lj_init_walkers)
#     lj_orchestrator.set_default_configuration(lj_null_configuration)

#     lj_orchestrator.gen_default_snapshot()


#     return lj_orchestrator



# @pytest.fixture(scope='class')
# def lj_orchestrator_file(tmp_path_factory, lj_apparatus, lj_init_walkers, lj_configuration):

#     # use an in memory database with sqlite

#     # make a path to the temporary directory for this orchestrator
#     orch_path = str(tmp_path_factory.mktemp('lj_fixture') / "lj.orch.sqlite")

#     # then create the seed/root/master orchestrator which will be used
#     # from here on out
#     orch = Orchestrator(orch_path)

#     return orch

# @pytest.fixture(scope='class')
# def lj_orchestrator_file_other(tmp_path_factory,
#                                    lj_apparatus, lj_init_walkers, lj_configuration):

#     # use an in memory database with sqlite

#     # make a path to the temporary directory for this orchestrator
#     orch_path = str(tmp_path_factory.mktemp('lj_fixture') / "lj_other.orch.sqlite")

#     # then create the seed/root/master orchestrator which will be used
#     # from here on out
#     orch = Orchestrator(orch_path)

#     return orch

# @pytest.fixture(scope='class')
# def lj_orchestrator_defaults_file(lj_orchestrator_file,
#                                       lj_apparatus, lj_init_walkers, lj_configuration):


#     lj_orchestrator_file.set_default_sim_apparatus(lj_apparatus)
#     lj_orchestrator_file.set_default_init_walkers(lj_init_walkers)
#     lj_orchestrator_file.set_default_configuration(lj_configuration)

#     lj_orchestrator_file.gen_default_snapshot()


#     return lj_orchestrator_file

# @pytest.fixture(scope='class')
# def lj_orchestrator_defaults_file_other(lj_orchestrator_file_other,
#                                             lj_apparatus, lj_init_walkers, lj_configuration):

#     lj_orchestrator_file_other.set_default_sim_apparatus(lj_apparatus)
#     lj_orchestrator_file_other.set_default_init_walkers(lj_init_walkers)
#     lj_orchestrator_file_other.set_default_configuration(lj_configuration)

#     lj_orchestrator_file_other.gen_default_snapshot()


#     return lj_orchestrator_file_other


# @pytest.fixture(scope='class')
# def lj_sim_manager(tmp_path_factory, lj_orchestrator_defaults):

#     start_snapshot = lj_orchestrator_defaults.get_default_snapshot()
#     configuration = lj_orchestrator_defaults.get_default_configuration()

#     # make a new temp dir for this configuration
#     tempdir = str(tmp_path_factory.mktemp('lj_sim_manager'))
#     configuration = configuration.reparametrize(work_dir=tempdir)

#     sim_manager = lj_orchestrator_defaults.gen_sim_manager(start_snapshot,
#                                                                configuration=configuration)

#     return sim_manager


# @pytest.fixture(scope='class')
# def lj_sim_manager_inmem(tmp_path_factory, lj_orchestrator_defaults_inmem):

#     start_snapshot = lj_orchestrator_defaults_inmem.get_default_snapshot()
#     configuration = lj_orchestrator_defaults_inmem.get_default_configuration()

#     # make a new temp dir for this configuration
#     tempdir = str(tmp_path_factory.mktemp('lj_sim_manager'))
#     configuration = configuration.reparametrize(work_dir=tempdir)

#     sim_manager = lj_orchestrator_defaults.gen_sim_manager(start_snapshot,
#                                                            configuration=configuration)

#     return sim_manager

# @pytest.fixture(scope='class')
# def lj_sim_manager_null(tmp_path_factory, lj_orchestrator_defaults_null):

#     start_snapshot = lj_orchestrator_defaults_null.get_default_snapshot()
#     configuration = lj_orchestrator_defaults_null.get_default_configuration()

#     # make a new temp dir for this configuration
#     tempdir = str(tmp_path_factory.mktemp('lj_sim_manager'))
#     configuration = configuration.reparametrize(work_dir=tempdir)

#     sim_manager = lj_orchestrator_defaults_null.gen_sim_manager(start_snapshot,
#                                                                 configuration=configuration)

#     return sim_manager


# @pytest.fixture(scope='class')
# def lj_sim_manager_run_results(lj_sim_manager):

#     n_cycles = 10
#     n_steps = 100

#     steps = [n_steps for _ in range(n_cycles)]

#     return lj_sim_manager.run_simulation(n_cycles, steps)

# @pytest.fixture(scope='class')
# def lj_sim_manager_null_run_results(lj_sim_manager_null):

#     n_cycles = 10
#     n_steps = 100

#     steps = [n_steps for _ in range(n_cycles)]

#     return lj_sim_manager_null.run_simulation(n_cycles, steps)

# @pytest.fixture(scope='class')
# def lj_orch_run_by_time_results(tmp_path_factory, lj_orchestrator_defaults):

#     runtime = 20 # seconds
#     n_steps = 100

#     start_snaphash = lj_orchestrator_defaults.get_default_snapshot_hash()

#     # make a new temp dir for this configuration
#     configuration = lj_orchestrator_defaults.get_default_configuration()
#     tempdir = str(tmp_path_factory.mktemp('lj_sim_manager'))
#     configuration = configuration.reparametrize(work_dir=tempdir)


#     return lj_orchestrator_defaults.run_snapshot_by_time(start_snaphash,
#                                                              runtime, n_steps,
#                                                              configuration=configuration)

# @pytest.fixture(scope='class')
# def lj_orch_run_end_snapshot(lj_orch_run_by_time_results):

#     end_snapshot, _, _, _ = lj_orch_run_by_time_results

#     return end_snapshot

# @pytest.fixture(scope='class')
# def lj_orch_orchestrated_run(tmp_path_factory, lj_orchestrator_defaults):

#     run_time = 20 # seconds
#     n_steps = 100

#     start_snaphash = lj_orchestrator_defaults.get_default_snapshot_hash()

#     tempdir = str(tmp_path_factory.mktemp('orchestrate_run'))

#     run_orch = lj_orchestrator_defaults.orchestrate_snapshot_run_by_time(start_snaphash,
#                                                                          run_time, n_steps,
#                                                                          work_dir=tempdir)

#     return run_orch

# @pytest.fixture(scope='class')
# def lj_orch_file_orchestrated_run(tmp_path_factory, lj_orchestrator_defaults_file):

#     run_time = 20 # seconds
#     n_steps = 100

#     start_snaphash = lj_orchestrator_defaults_file.get_default_snapshot_hash()

#     tempdir = str(tmp_path_factory.mktemp('orchestrate_run'))

#     run_orch = lj_orchestrator_defaults_file.orchestrate_snapshot_run_by_time(start_snaphash,
#                                                                          run_time, n_steps,
#                                                                          work_dir=tempdir)

#     return run_orch

# @pytest.fixture(scope='class')
# def lj_orch_file_other_orchestrated_run(tmp_path_factory,
#                                         lj_orchestrator_defaults_file_other):

#     run_time = 20 # seconds
#     n_steps = 100

#     start_snaphash = lj_orchestrator_defaults_file_other.get_default_snapshot_hash()

#     tempdir = str(tmp_path_factory.mktemp('orchestrate_run_other'))

#     run_orch = lj_orchestrator_defaults_file_other.orchestrate_snapshot_run_by_time(start_snaphash,
#                                                                          run_time, n_steps,
#                                                                          work_dir=tempdir)

#     return run_orch


# @pytest.fixture(scope='class')
# def lj_orch_reconciled_orchs(tmp_path_factory, lj_apparatus, lj_init_walkers, lj_configuration):

#     run_time = 20 # seconds
#     n_steps = 100

#     # tempdirs for the orchestrators and configuration output
#     first_tempdir = str(tmp_path_factory.mktemp('reconcile_first_run'))
#     second_tempdir = str(tmp_path_factory.mktemp('reconcile_second_run'))

#     first_orch_path = osp.join(first_tempdir, "first.orch.sqlite")
#     second_orch_path = osp.join(second_tempdir, "second.orch.sqlite")

#     # make two orchestrators in their directories
#     first_orch = Orchestrator(orch_path=first_orch_path)
#     second_orch = Orchestrator(orch_path=second_orch_path)

#     # configure them
#     # 1
#     first_orch.set_default_sim_apparatus(lj_apparatus)
#     first_orch.set_default_init_walkers(lj_init_walkers)
#     first_orch.set_default_configuration(lj_configuration)
#     first_orch.gen_default_snapshot()
#     # 2
#     second_orch.set_default_sim_apparatus(lj_apparatus)
#     second_orch.set_default_init_walkers(lj_init_walkers)
#     second_orch.set_default_configuration(lj_configuration)
#     second_orch.gen_default_snapshot()

#     # do independent runs for each of them

#     # start snapshot hashes
#     first_starthash = first_orch.get_default_snapshot_hash()
#     second_starthash = second_orch.get_default_snapshot_hash()

#     # then orchestrate the runs
#     first_run_orch = first_orch.orchestrate_snapshot_run_by_time(first_starthash,
#                                                        run_time, n_steps,
#                                                        work_dir=first_tempdir)

#     second_run_orch = second_orch.orchestrate_snapshot_run_by_time(second_starthash,
#                                                        run_time, n_steps,
#                                                        work_dir=second_tempdir)

#     # then reconcile them
#     reconciled_orch = reconcile_orchestrators(first_run_orch.orch_path, second_run_orch.orch_path)

#     return first_run_orch, second_run_orch, reconciled_orch
