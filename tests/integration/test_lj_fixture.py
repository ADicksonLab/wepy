# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import multiprocessing as mp
import pdb

# Third Party Library
import pytest

# testing helpers
from multiprocessing_logging import install_mp_handler

# First Party Library
from wepy.sim_manager import Manager

# we define a fixture for a fixture for all the components of a
# simulation of the openmmtools Lennard-Jones pair. We test the
# fixtures by requiring them one by one.

# the fixtures are class scoped so we make a class for this

lj_fixtures = [
    "lj_params",
    "lj_omm_sys",
    "lj_integrator",
    "lj_init_sim_state",
    "lj_init_state",
    "lj_openmm_runner",
    "lj_distance_metric",
    "lj_resampler",
    "lj_topology",
    "lj_boundary_condition",
    "lj_reporter_kwargs",
    "lj_reporter_classes",
    "lj_init_walkers",
    "lj_apparatus",
    "lj_snapshot",
    "lj_configuration",
    "lj_work_mapper",
    "lj_reporters",
    "lj_orchestrator",
    "lj_orchestrator_defaults",
    "lj_orchestrator_file",
    "lj_orchestrator_file_other",
    "lj_orchestrator_defaults_file",
    "lj_sim_manager",
    "lj_sim_manager_run_results",
    "lj_orch_run_by_time_results",
    "lj_orch_run_end_snapshot",
    "lj_orch_orchestrated_run",
    "lj_orch_file_orchestrated_run",
    "lj_orch_file_other_orchestrated_run",
    "lj_orch_reconciled_orchs",
    "lj_sim_manager_null_run_results",
]


@pytest.mark.interactive
def test_init_state(lj_init_state):
    pdb.set_trace()
    pass


@pytest.mark.usefixtures(*lj_fixtures)
class TestLJPairNewOrch:
    # just an empty thing to get the fixtures made and catch errors
    # there
    def test_fixtures(self):
        pass

    @pytest.mark.interactive
    def test_orch_interactive(self, lj_orchestrator_defaults):
        pdb.set_trace()

        pass

    @pytest.mark.interactive
    def test_reconciled_orch(self, lj_orch_reconciled_orchs):
        host_orch, other_orch, reconciled_orch = lj_orch_reconciled_orchs
        pdb.set_trace()

        pass


@pytest.mark.usefixtures(
    "lj_reporters",
    "lj_init_walkers",
    "lj_openmm_runner",
    "lj_unbinding_bc",
    "lj_wexplore_resampler",
    "lj_revo_resampler",
    "lj_work_mapper",
    "lj_work_mapper_worker",
    "lj_work_mapper_task",
)
class TestLJSimIntegration:
    # TODO: add revo back in after all combinations with WExplore are passing
    # @pytest.mark.parametrize('resampler_class', ['WExploreResampler', 'REVOResampler',])

    # NOTE: CUDA has issues but OpenCL tests the code path that we
    # need so we will just use it here

    # order matters here for the platforms and the work mapper classes
    # since there is issues with that and typically aren't being all
    # used in the same place like we do here. Basically do the
    # 'Mapper' last since it doesn't use it's own multiprocessing
    # context.
    @pytest.mark.parametrize(
        "boundary_condition_class",
        [
            "UnbindingBC",
        ],
    )
    @pytest.mark.parametrize(
        "resampler_class",
        [
            "WExploreResampler",
        ],
    )
    @pytest.mark.parametrize(
        "platform",
        [
            "Reference",
            "CPU",
            "OpenCL",
        ],
    )  # 'CUDA'
    @pytest.mark.parametrize(
        "work_mapper_class",
        [
            "WorkerMapper",
            "TaskMapper",
            "Mapper",
        ],
    )
    def test_lj_sim_manager_openmm_integration_run(
        self,
        class_tmp_path_factory,
        boundary_condition_class,
        resampler_class,
        work_mapper_class,
        platform,
        lj_params,
        lj_omm_sys,
        lj_integrator,
        lj_reporter_classes,
        lj_reporter_kwargs,
        lj_init_walkers,
        lj_openmm_runner,
        lj_unbinding_bc,
        lj_wexplore_resampler,
        lj_revo_resampler,
    ):
        """Run all combinations of components in the fixtures for the smallest
        amount of time, just to make sure they all work together and don't give errors.
        """

        logger = logging.getLogger("testing").setLevel(logging.DEBUG)
        install_mp_handler()
        logger.debug("Starting the test")

        print("starting the test")

        # the configuration class gives us a convenient way to
        # parametrize our reporters for the locale
        # First Party Library
        from wepy.orchestration.configuration import Configuration

        # the runner
        from wepy.runners.openmm import (
            OpenMMCPUWalkerTaskProcess,
            OpenMMCPUWorker,
            OpenMMGPUWalkerTaskProcess,
            OpenMMGPUWorker,
            OpenMMRunner,
        )

        # mappers
        from wepy.work_mapper.mapper import Mapper

        # the walker task types for the TaskMapper
        from wepy.work_mapper.task_mapper import TaskMapper, WalkerTaskProcess

        # the worker types for the WorkerMapper
        from wepy.work_mapper.worker import Worker, WorkerMapper

        n_cycles = 1
        n_steps = 2
        num_workers = 2

        # generate the reporters and temporary directory for this test
        # combination

        tmpdir_template = "lj_fixture_{plat}-{wm}-{res}-{bc}"
        tmpdir_name = tmpdir_template.format(
            plat=platform,
            wm=work_mapper_class,
            res=resampler_class,
            bc=boundary_condition_class,
        )

        # make a temporary directory for this configuration to work with
        tmpdir = str(class_tmp_path_factory.mktemp(tmpdir_name))

        # make a config so that the reporters get parametrized properly
        reporters = Configuration(
            work_dir=tmpdir,
            reporter_classes=lj_reporter_classes,
            reporter_partial_kwargs=lj_reporter_kwargs,
        ).reporters

        steps = [n_steps for _ in range(n_cycles)]

        # choose the components based on the parametrization
        boundary_condition = None
        resampler = None

        walker_fixtures = [lj_init_walkers]
        runner_fixtures = [lj_openmm_runner]
        boundary_condition_fixtures = [lj_unbinding_bc]
        resampler_fixtures = [lj_wexplore_resampler, lj_revo_resampler]

        walkers = lj_init_walkers

        boundary_condition = [
            boundary_condition
            for boundary_condition in boundary_condition_fixtures
            if type(boundary_condition).__name__ == boundary_condition_class
        ][0]
        resampler = [
            resampler
            for resampler in resampler_fixtures
            if type(resampler).__name__ == resampler_class
        ][0]

        assert boundary_condition is not None
        assert resampler is not None

        # generate the work mapper given the type and the platform

        work_mapper_classes = {
            mapper_class.__name__: mapper_class
            for mapper_class in [Mapper, WorkerMapper, TaskMapper]
        }

        # # select the right one given the option
        # work_mapper_type = [mapper_type for mapper_type in work_mapper_classes
        #                     if type(mapper_type).__name__ == work_mapper_class][0]

        # decide based on the platform and the work mapper which
        # platform dependent components to build
        if work_mapper_class == "Mapper":
            # then there is no settings
            work_mapper = Mapper()

        elif work_mapper_class == "WorkerMapper":
            if platform == "CUDA" or platform == "OpenCL":
                work_mapper = WorkerMapper(
                    num_workers=num_workers,
                    worker_type=OpenMMGPUWorker,
                    device_ids={"0": 0, "1": 1},
                    proc_start_method="spawn",
                )
            if platform == "OpenCL":
                work_mapper = WorkerMapper(
                    num_workers=num_workers,
                    worker_type=OpenMMGPUWorker,
                    device_ids={"0": 0, "1": 1},
                )

            elif platform == "CPU":
                work_mapper = WorkerMapper(
                    num_workers=num_workers,
                    worker_type=OpenMMCPUWorker,
                    worker_attributes={"num_threads": 1},
                )

            elif platform == "Reference":
                work_mapper = WorkerMapper(
                    num_workers=num_workers,
                    worker_type=Worker,
                )

        elif work_mapper_class == "TaskMapper":
            if platform == "CUDA":
                work_mapper = TaskMapper(
                    num_workers=num_workers,
                    walker_task_type=OpenMMGPUWalkerTaskProcess,
                    device_ids={"0": 0, "1": 1},
                    proc_start_method="spawn",
                )

            elif platform == "OpenCL":
                work_mapper = TaskMapper(
                    num_workers=num_workers,
                    walker_task_type=OpenMMGPUWalkerTaskProcess,
                    device_ids={"0": 0, "1": 1},
                )

            elif platform == "CPU":
                work_mapper = TaskMapper(
                    num_workers=num_workers,
                    walker_task_type=OpenMMCPUWalkerTaskProcess,
                    worker_attributes={"num_threads": 1},
                )

            elif platform == "Reference":
                work_mapper = TaskMapper(
                    num_workers=num_workers,
                    worker_type=WalkerTaskProcess,
                )

        else:
            raise ValueError("Platform {} not recognized".format(platform))

        # initialize the runner with the platform
        runner = OpenMMRunner(
            lj_omm_sys.system, lj_omm_sys.topology, lj_integrator, platform=platform
        )

        logger.debug("Constructing the manager")

        manager = Manager(
            walkers,
            runner=runner,
            boundary_conditions=boundary_condition,
            resampler=resampler,
            work_mapper=work_mapper,
            reporters=reporters,
        )

        # since different work mappers need different process start
        # methods for different platforms i.e. CUDA and linux fork
        # vs. spawn we choose the appropriate one for each method.

        logger.debug("Starting the simulation")

        walkers, filters = manager.run_simulation(
            n_cycles, steps, num_workers=num_workers
        )

        # no assert if it runs we are happy for now
