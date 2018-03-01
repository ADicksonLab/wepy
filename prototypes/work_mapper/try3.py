import time
import asyncio
from collections import deque


class WorkerQueue():

    def __init__(self, worker_idxs):
        self.worker_idxs = deque(worker_idxs)


class Worker():

    def __init__(worker_idx):
        self.worker_idx = worker_idx

    def work(func, *args):
        return func(*args)

async def func(job_idx, *args):
    print("Start IN FUNC job {} at time: {}".format(job_idx, time.time()))
    time.sleep(10)
    print("End IN FUNC job {} at time: {}".format(job_idx, time.time()))
    yield args

async def map(_call_, args):
    results = []
    print(args)
    for i, arg in enumerate(args):
        result = await _call_(i, *arg)
        print(result)
    return results

n_jobs = 6
job_args = [(i,) for i in range(n_jobs)]

loop = asyncio.get_event_loop()
loop.run_until_complete(map(func, job_args))
loop.close()
