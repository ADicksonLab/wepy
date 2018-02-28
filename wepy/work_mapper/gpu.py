import multiprocessing as mulproc

from multiprocessing import Process

class GPUMapperFutures(object):

    def __init__(self, gpu_idxs):
        self.worker_idxs = gpu_idxs

class GPUMapperMultiprocessing(object):

    def __init__(self, gpu_idxs):
        self.worker_idxs = gpu_idxs
        self.free_workers = 

    @property
    def n_workers(self):
        return len(self.gpu_idxs)

    def get_worker_idx(self):
        pass

    def release_worker(self, worker_idx):
        pass

    def exec_call(self, func, walker_idx, *args):
        raise NotImplementedError

    def map(self, func, *iterables):

        # structure for the results to go into
        results = mulproc.Manager().list()

        # make the pool of work to be done
        walkers_pool = []
        for walker_idx, args in enumerate(zip(*iterables)):
            walkers_pool.append(Process(target=self.exec_call,
                                        args=(func, walker_idx, *args)))


        # 


class GPUMapper(object):

    def __init__(self, n_walkers, n_workers=None, gpu_indices=None):

        if gpu_indices is not None:
            self.gpu_indices = gpu_indices
            self.n_workers = len(gpu_indices)
        else:
            assert n_workers, "If gpu_indices are not given the n_workers must be given"
            self.n_workers = n_workers
            self.gpu_indices = range(n_workers)

        # make a Queue for free workers, when one is being used it is
        # popped off and locked
        self.free_workers = mulproc.Queue()

        # the semaphore provides the locks on the workers
        self.lock = mulproc.Semaphore(self.n_workers)

        # initialize a list to put results in
        self.results_list = mulproc.Manager().list()
        for i in range(n_walkers):
            self.results_list.append(None)

        # add the free worker indices (not device/gpu indices) to the
        # free workers queue
        for i in range(self.n_workers):
            self.free_workers.put(i)

    def exec_call(self, _call_, index, *args):

        # lock a worker, then pop it off the queue so no other process
        # tries to use it
        self.lock.acquire()
        worker_idx = self.free_workers.get()

        # convert this to an available GPU index
        gpu_idx = self.gpu_indices[worker_idx]

        # call the function which is accepted as kwargs
        result = _call_(*args, DeviceIndex=str(gpu_idx))

        self.free_workers.put(worker_idx)
        self.results_list[index] = result
        self.lock.release()


    def map(self, _call_, *iterables):

        walkers_pool = []
        # create processes and start to run
        index = 0

        for args in zip(*iterables):
            walkers_pool.append(mulproc.Process(target=self.exec_call,
                                                args=(_call_, index, *args)))
            index += 1

        for p in walkers_pool:
            p.start()

        for p in walkers_pool:
            p.join()


        return self.results_list
