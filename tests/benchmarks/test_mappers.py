# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import multiprocessing as mp
import time
from copy import deepcopy

# Third Party Library
import pytest

# First Party Library
from wepy.resampling.resamplers.resampler import NoResampler
from wepy.runners.openmm import (
    OpenMMCPUWalkerTaskProcess,
    OpenMMCPUWorker,
    OpenMMGPUWalkerTaskProcess,
    OpenMMGPUWorker,
    OpenMMRunner,
    OpenMMState,
    OpenMMWalker,
)
from wepy.sim_manager import Manager
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.mapper import Mapper, TaskException
from wepy.work_mapper.task_mapper import (
    TaskMapper,
    TaskProcessException,
    WalkerTaskProcess,
)
from wepy.work_mapper.worker import Worker, WorkerException, WorkerMapper
from wepy_tools.sim_makers.openmm.lennard_jones import LennardJonesPairOpenMMSimMaker
from wepy_tools.sim_makers.openmm.lysozyme import LysozymeImplicitOpenMMSimMaker

N_WALKER_TESTS = [8, 16, 48, 96]
N_WORKER_TESTS = [2, 4, 8]
# 5 ps, 10 ps, 20 ps
N_STEPS_TEST = [5000, 10000, 20000]
N_CYCLES_TEST = [100]
SYSTEMS_TEST = [
    "LysozymeImplicit",
]  # 'LennardJonesPair',
PLATFORMS_TEST = ["OpenCL"]  # ['Reference', 'CPU', 'OpenCL']


def get_sim_maker(spec):
    if spec == "LennardJonesPair":
        sim_maker = LennardJonesPairOpenMMSimMaker()
    elif spec == "LysozymeImplicit":
        sim_maker = LysozymeImplicitOpenMMSimMaker()
    else:
        raise ValueError("Unknown system spec: {}".format(spec))

    return sim_maker


class TestBenchmark:
    @pytest.mark.parametrize("n_walkers", N_WALKER_TESTS)
    @pytest.mark.parametrize("n_cycles", N_CYCLES_TEST)
    @pytest.mark.parametrize("n_workers", N_WORKER_TESTS)
    @pytest.mark.parametrize("n_steps", N_STEPS_TEST)
    @pytest.mark.parametrize("platform", PLATFORMS_TEST)
    @pytest.mark.parametrize("system", SYSTEMS_TEST)
    @pytest.mark.parametrize("mapper", ["TaskMapper", "WorkerMapper"])
    def test_mappers(
        self,
        n_walkers,
        n_cycles,
        n_workers,
        n_steps,
        platform,
        system,
        mapper,
        benchmark,
    ):
        # certain combinations don't make sense so we skip them
        if mapper == "Mapper" and n_workers != 1:
            pytest.skip("The Mapper work mapper only works with a single worker.")

        sim_maker = get_sim_maker(system)

        apparatus = sim_maker.make_apparatus(platform=platform, resampler="NoResampler")

        config = sim_maker.make_configuration(work_mapper=mapper, platform=platform)

        sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)

        def thunk():
            return sim_manager.run_simulation(n_cycles, n_steps, num_workers=n_workers)

        result = benchmark(thunk)


# TODO make this basically the same as the others so we can have
# uniformity and just do all of the benchmarks in one
# - [ ] SimpleRunner
# - [ ] SimpleSimMaker


def gen_walkers(n_args):
    args = range(n_args)

    return [Walker(WalkerState(**{"num": arg}), 1 / len(args)) for arg in args]


# test basic functionality
def task_pass(walker):
    # simulate it actually taking some time
    time.sleep(3)
    n = walker.state["num"]
    return Walker(WalkerState(**{"num": n + 1}), walker.weight)


class TestSimpleBenchmark:
    @pytest.mark.parametrize("n_walkers", N_WALKER_TESTS)
    def test_Mapper(self, n_walkers, benchmark):
        mapper = Mapper(segment_func=task_pass)

        mapper.init()

        def thunk():
            return mapper.map(gen_walkers(n_walkers))

        result = benchmark(thunk)

        mapper.cleanup()

    @pytest.mark.parametrize("n_walkers", N_WALKER_TESTS)
    @pytest.mark.parametrize("n_workers", N_WORKER_TESTS)
    def test_WorkerMapper(self, n_walkers, n_workers, benchmark):
        mapper = WorkerMapper(
            segment_func=task_pass, num_workers=n_workers, worker_type=Worker
        )

        mapper.init()

        def thunk():
            return mapper.map(gen_walkers(n_walkers))

        result = benchmark(thunk)

        mapper.cleanup()

    @pytest.mark.parametrize("n_walkers", N_WALKER_TESTS)
    @pytest.mark.parametrize("n_workers", N_WORKER_TESTS)
    def test_TaskMapper(self, n_walkers, n_workers, benchmark):
        mapper = TaskMapper(
            segment_func=task_pass,
            num_workers=n_workers,
            walker_task_type=WalkerTaskProcess,
        )

        mapper.init()

        def thunk():
            return mapper.map(gen_walkers(n_walkers))

        result = benchmark(thunk)

        mapper.cleanup()
