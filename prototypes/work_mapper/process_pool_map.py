# easy implementation of a process pool for just getting processors

import time
from concurrent.futures import ProcessPoolExecutor
from collections import deque


def func(job_idx, *args):
    print("Starting job {}".format(job_idx))
    time.sleep(20)
    print("finished job {}".format(job_idx))
    return args


def main(n_jobs, n_workers):

    job_args = [(i,) for i in range(n_jobs)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = job_args, executor.map(func, job_args)

    return results

if __name__ == "__main__":

    results = main(10, 2)
