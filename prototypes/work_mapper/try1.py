# don't know came from python docs example
import time
from multiprocessing import Process, Queue, current_process

def func(job_idx, *args):
    print("Starting job {}".format(job_idx))
    time.sleep(20)
    print("finished job {}".format(job_idx))
    return args

def worker(input, out):
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)

def main(tasks, n_workers):
    task_queue = Queue()
    done_queue = Queue()

    for task_idx, task in enumerate(tasks):
        task_queue.put((task_idx, *task))

        for i in range(n_workers):
            Process(target=worker,
                    args=(task_queue, done_queue)).start()

    return done_queue

if __name__ == '__main__':

    n_jobs = 6
    job_args = ([i for i in range(n_jobs)],)
    n_workers = 2

    results = main(job_args, n_workers)
