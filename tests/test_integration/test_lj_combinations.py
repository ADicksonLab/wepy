import multiprocessing as mp
import logging
from copy import deepcopy
import time

import pytest

from wepy.sim_manager import Manager
from wepy.walker import Walker, WalkerState
from wepy.resampling.resamplers.resampler import NoResampler
from wepy.runners.openmm import (
    OpenMMState, OpenMMWalker, OpenMMRunner,
    OpenMMCPUWorker, OpenMMGPUWorker,
    OpenMMCPUWalkerTaskProcess, OpenMMGPUWalkerTaskProcess,
)

from wepy.work_mapper.mapper import Mapper, TaskException
from wepy.work_mapper.worker import Worker, WorkerMapper, WorkerException
from wepy.work_mapper.task_mapper import TaskMapper, WalkerTaskProcess, TaskProcessException

from wepy_tools.sim_makers.openmm.lennard_jones import LennardJonesPairOpenMMSimMaker
from wepy_tools.sim_makers.openmm.lysozyme import LysozymeImplicitOpenMMSimMaker



def get_sim_maker(spec):

    if spec == 'LennardJonesPair':
        sim_maker = LennardJonesPairOpenMMSimMaker()
    elif spec == 'LysozymeImplicit':
        sim_maker = LysozymeImplicitOpenMMSimMaker()
    else:
        raise ValueError("Unknown system spec: {}".format(spec))

    return sim_maker

# testing on a node in HPCC means we will have 8 GPUS
BIG_NODE_N_WORKERS = 8
DEV_NODE_N_WORKERS = 1

# number of walkers in multiples of 8 since that is how many GPUs we
# have
BIG_NODE_N_WALKER_TESTS = [i*BIG_NODE_N_WORKERS for i in (1,2,4,8)]
DEV_NODE_N_WALKER_TESTS = [i*DEV_NODE_N_WORKERS for i in (5,10,20)]
DEV_NODE_N_WALKER_TESTS = [i for i in (10,)]

# 1 ps, 5 ps, 10 ps, #20 ps
N_STEPS_TEST = [10, 50, 100, 200]#[1000, 5000, 10000]
N_CYCLES_TEST = [1, 10, 100]
SYSTEMS_TEST = ['LennardJonesPair', 'LysozymeImplicit',]
PLATFORMS_TEST = ['OpenCL']
RESAMPLERS_TEST = ['NoResampler', 'REVOResampler', 'WExploreResampler']
WORK_MAPPERS_TEST = ['WorkerMapper', 'TaskMapper']

@pytest.mark.node_minor
class TestCombinationsMinorNode():

    @pytest.mark.parametrize('n_walkers', [5,])
    @pytest.mark.parametrize('n_cycles', [3,])
    @pytest.mark.parametrize('n_steps', [10,])
    @pytest.mark.parametrize('platform', ['CPU',])
    @pytest.mark.parametrize('system', SYSTEMS_TEST)
    @pytest.mark.parametrize('resampler', RESAMPLERS_TEST)
    @pytest.mark.parametrize('work_mapper', WORK_MAPPERS_TEST)
    def test_combinations(self, n_walkers, n_cycles, n_steps,
                          platform, system, resampler, work_mapper):

        sim_maker = get_sim_maker(system)

        apparatus = sim_maker.make_apparatus(platform=platform,
                                             resampler=resampler)

        config = sim_maker.make_configuration(apparatus,
                                              work_mapper=work_mapper,
                                              platform=platform,
                                              reporters=None)

        sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)

        result = sim_manager.run_simulation(n_cycles, n_steps,
                                              num_workers=DEV_NODE_N_WORKERS)


@pytest.mark.node_dev
class TestCombinationsDevNode():

    @pytest.mark.parametrize('n_walkers', [5,])
    @pytest.mark.parametrize('n_cycles', [3,])
    @pytest.mark.parametrize('n_steps', [10,])
    @pytest.mark.parametrize('platform', ['OpenCL',])
    @pytest.mark.parametrize('system', SYSTEMS_TEST)
    @pytest.mark.parametrize('resampler', RESAMPLERS_TEST)
    @pytest.mark.parametrize('work_mapper', WORK_MAPPERS_TEST)
    def test_combinations(self, n_walkers, n_cycles, n_steps,
                          platform, system, resampler, work_mapper):

        sim_maker = get_sim_maker(system)

        apparatus = sim_maker.make_apparatus(platform=platform,
                                             resampler=resampler)

        config = sim_maker.make_configuration(apparatus,
                                              work_mapper='TaskMapper',
                                              platform=platform,
                                              reporters=None)

        sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)

        result = sim_manager.run_simulation(n_cycles, n_steps,
                                              num_workers=DEV_NODE_N_WORKERS)
@pytest.mark.node_production
class TestCombinationsBigNode():

    @pytest.mark.parametrize('n_walkers', BIG_NODE_N_WALKER_TESTS)
    @pytest.mark.parametrize('n_cycles', N_CYCLES_TEST)
    @pytest.mark.parametrize('n_steps', N_STEPS_TEST)
    @pytest.mark.parametrize('platform', PLATFORMS_TEST)
    @pytest.mark.parametrize('system', SYSTEMS_TEST)
    @pytest.mark.parametrize('resampler', RESAMPLERS_TEST)
    def test_combinations(self, n_walkers, n_cycles, n_steps, platform, system, resampler):

        sim_maker = get_sim_maker(system)

        apparatus = sim_maker.make_apparatus(platform=platform,
                                             resampler=resampler)

        config = sim_maker.make_configuration(work_mapper='TaskMapper',
                                              platform=platform)

        sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)

        result = sim_manager.run_simulation(n_cycles, n_steps,
                                              num_workers=BIG_NODE_N_WORKERS)
