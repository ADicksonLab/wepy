import time
from multiprocessing import Queue, Process, Semaphore, Manager

class Mapper(object):

    def __init__(self, n_walkers, gpu_indices):

        print("initializing mapper")
        self.gpu_indices = gpu_indices
        self.n_workers = len(gpu_indices)
        self.free_workers = Queue()

        self.lock = Semaphore(self.n_workers)

        self.results_list = Manager().list()
        for i in range(n_walkers):
            self.results_list.append(None)

        for i in range(self.n_workers):
            self.free_workers.put(i)

        print("done initializing mapper")

    def exec_call(self, _call_, index, *args):


        self.lock.acquire()
        worker_idx = self.free_workers.get()

        gpu_idx = self.gpu_indices[worker_idx]
        print("Starting exec_call job {} on worker {}".format(index, gpu_idx))
        result = _call_(*args, gpu_idx)

        self.free_workers.put(worker_idx)
        self.results_list[index] = result
        self.lock.release()
        print("Ending exec_call job {}".format(index))



    def map(self, _call_, *iterables):

        walkers_pool = []

        print("Creating processes")
        for index, args in enumerate(zip(*iterables)):
            walkers_pool.append(Process(target=self.exec_call,
                                        args=(_call_, index, *args)))

        print("Finished creating processes, {} made".format(len(walkers_pool)))

        print("Starting processes")
        for i, p in enumerate(walkers_pool):
            print("Starting process {}".format(i))
            p.start()
            print("Ending process {}".format(i))


        for i, p in enumerate(walkers_pool):
            print("starting JOIN process {}".format(i))
            p.join()
            print("Ending JOIN process {}".format(i))

        print("Finished")
        return self.results_list


if __name__ == "__main__":

    def func(job_idx, *args):
        print("Start IN FUNC job {} at time: {}".format(job_idx, time.time()))
        time.sleep(20)
        print("End IN FUNC job {} at time: {}".format(job_idx, time.time()))
        return args

    n_workers = 2
    worker_idxs = [i for i in range(n_workers)]

    n_jobs = 6
    job_args = ([i for i in range(n_jobs)],)

    mapper = Mapper(n_jobs, worker_idxs)

    print("mapping")
    results = mapper.map(func, *job_args)
    print("finished mapping")

