# Not working code, just a sketch of an idea of using asyncio and
# queues and an event loop to check for finished things.

import time
import asyncio

class MyProcess():

    def __init__(self, func, args):
        self.args = args
        self.func = func

    def work(self, worker_idx):

        try:
            result = self.func(self.args.pop(), worker_idx)
        except IndexError:
            # no more jobs
            break

        return result, worker_idx

async def myfunc(future, job_idx, *args):
    print("Start IN FUNC job {} at time: {}".format(job_idx, time.time()))
    time.sleep(10)
    print("End IN FUNC job {} at time: {}".format(job_idx, time.time()))

    future.set_result((job_idx, args))

n_jobs = 6
job_args = [(i,) for i in range(n_jobs)]
n_workers = 2

# make the futures
job_queue = [asyncio.Future() for job in job_args]
result_queue = []
worker_queue = [worker_idx for worker_idx in range(n_workers)]
while True:


    # if there are any workers available schedule some work in the
    # event loop
    if len(worker_queue) > 0 and len(job_queue) > 0:
        while True:
            try:
                worker_idx = worker_queue.pop()
            except IndexError:
                # no more workers in the queue
                break

            # schedule a job
            job = myfunc(job_queue.pop())
            result_queue.append(job)
            asyncio.ensure_future(job)

    # if all the jobs are done quit. Do this last so we can save if we
    # need to
    if len(job_queue) < 1:
        break
