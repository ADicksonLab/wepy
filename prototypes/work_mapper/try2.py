# starts two processes then blocks until the first one started is done

import time
from multiprocessing import Pool, Queue




if __name__ == '__main__':

    def func(job_idx, *args):
        print("Starting job {}".format(job_idx))
        time.sleep(20)
        print("finished job {}".format(job_idx))
        return args

    n_jobs = 6
    job_args = [(i,) for i in range(n_jobs)]
    n_workers = 2

    pool = Pool(n_workers)

    futures = []
    done_idx = 0
    results = []
    for job_idx, job_arg in enumerate(job_args):
        futures.append(pool.apply_async(func, (job_idx, *job_arg)))

        if len(futures) > 2:
            print("JOB idx {}".format(job_idx))
            print("done_idx {}".format(done_idx))
            futures[done_idx].wait()
            results.append(futures[done_idx].get())
            done_idx += 1
