import multiprocessing as mulproc

class GPUMapper(object):

    def __init__(self, n_walkers, n_workers):

        # TODO add comments describing what is going on
        self.free_workers= mulproc.Queue()
        self.lock = mulproc.Semaphore(n_workers)
        self.results_list = mulproc.Manager().list()
        for i in range(n_walkers):
            self.results_list.append(None)
        for i in range (n_workers):
            self.free_workers.put(i)

    def exec_call(self, _call_, index, *args):
        # gets a free GPU and calls the runnable function
        self.lock.acquire()
        gpu_index = self.free_workers.get()
        result = _call_(*args, DeviceIndex=str(gpu_index))

        self.free_workers.put(gpu_index)
        self.results_list[index] = result
        self.lock.release()


    def map(self, _call_, *iterables):

        walkers_pool = []
        # create processes and start to run
        index = 0

        for args in zip(*iterables):
            walkers_pool.append(mulproc.Process(target=self.exec_call,
                                                args=(_call_,index,*args)))
            index += 1

        for p in walkers_pool:
            p.start()

        for p in walkers_pool:
            p.join()


        return self.results_list
