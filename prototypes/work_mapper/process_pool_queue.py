import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque




def main(_call_, n_jobs, n_workers):


    # queue for the workers
    workers = deque([i for i in range(n_workers)])


    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        job_idxs = deque([i for i in range(n_jobs)])
        job_args = deque([(i,) for i in range(n_jobs)])

        def start_new_job(_future_):
            try:
                executor.submit(_call_, job_idxs.popleft(), job_args.popleft())
            except IndexError:
                # no more jobs left
                pass


        # start the first jobs given the number of workers we have to
        # use
        job_futures = []
        for worker_idx in range(n_workers):
            # submit all the jobs and get their futures
            future = executor.submit(_call_, job_idxs.popleft(), job_args.popleft())

            # save the future
            job_futures.append(future)

        # iterate over them as they are completed
        for future in as_completed(job_futures)

    return results

if __name__ == "__main__":

    def func(job_idx, *args):
        print("Starting job {}".format(job_idx))
        time.sleep(20)
        print("finished job {}".format(job_idx))
        return args

    results = main(func, 10, 2)
