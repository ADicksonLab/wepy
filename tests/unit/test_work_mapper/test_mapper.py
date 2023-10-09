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
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.mapper import Mapper, TaskException
from wepy.work_mapper.task_mapper import (
    TaskMapper,
    TaskProcessException,
    WalkerTaskProcess,
)
from wepy.work_mapper.worker import Worker, WorkerException, WorkerMapper

ARGS = (0, 1, 2)


def gen_walkers():
    return [Walker(WalkerState(**{"num": arg}), 1 / len(ARGS)) for arg in ARGS]


# test basic functionality
def task_pass(walker):
    # simulate it actually taking some time
    n = walker.state["num"]
    return Walker(WalkerState(**{"num": n + 1}), walker.weight)


TASK_PASS_ANSWER = [n + 1 for n in ARGS]


class TestWorkMappers:
    def test_mapper(self):
        mapper = Mapper(segment_func=task_pass)

        mapper.init()

        results = mapper.map(gen_walkers())

        assert all(
            [res.state["num"] == TASK_PASS_ANSWER[i] for i, res in enumerate(results)]
        )

        mapper.cleanup()

    def test_worker_mapper(self):
        mapper = WorkerMapper(segment_func=task_pass, num_workers=3, worker_type=Worker)

        mapper.init()

        results = mapper.map(gen_walkers())

        assert all(
            [res.state["num"] == TASK_PASS_ANSWER[i] for i, res in enumerate(results)]
        )

        mapper.cleanup()

    def test_task_mapper(self):
        mapper = TaskMapper(
            segment_func=task_pass, num_workers=3, walker_task_type=WalkerTaskProcess
        )

        mapper.init()

        results = mapper.map(gen_walkers())

        assert all(
            [res.state["num"] == TASK_PASS_ANSWER[i] for i, res in enumerate(results)]
        )

        mapper.cleanup()

        time.sleep(1)


# test that task failures are passed up properly
def task_fail(walker):
    n = walker.state["num"]
    if n == 1:
        raise ValueError("No soup for you!!")
    else:
        return Walker(WalkerState(**{"num": n + 1}), walker.weight)


class TestTaskFail:
    ARGS = ((0, 1, 2),)

    def test_mapper(self):
        mapper = Mapper(segment_func=task_fail)

        mapper.init()

        with pytest.raises(TaskException) as task_exc_info:
            results = mapper.map(gen_walkers())

        mapper.cleanup()

    def test_worker_mapper(self):
        mapper = WorkerMapper(segment_func=task_fail, num_workers=3, worker_type=Worker)

        mapper.init()

        with pytest.raises(TaskException) as task_exc_info:
            results = mapper.map(gen_walkers())

        mapper.cleanup()

    def test_task_mapper(self):
        mapper = TaskMapper(
            segment_func=task_fail, num_workers=3, walker_task_type=WalkerTaskProcess
        )

        mapper.init()

        with pytest.raises(TaskException) as task_exc_info:
            results = mapper.map(gen_walkers())

        mapper.cleanup()
