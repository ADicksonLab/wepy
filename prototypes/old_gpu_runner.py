import multiprocessing as mulproc

from multiprocessing import Process

class GPUProcess(Process):
    def __init__(self, task_queue, result_queue):
        Process.__init__(self)

        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:

            # get the next task
            task_idx, next_task = self.task_queue.get()

            # check for the poison pill which is the signal to stop
            if next_task is None:


                print('{}: Exiting'.format(self.name))

                # mark the task as done
                self.task_queue.task_done()

                # and exit the loop
                break

            print('{}: {}'.format(self.name, next_task))

            # run the task
            answer = next_task()

            # (for joinable queue) tell the queue that the formerly
            # enqued task is complete
            self.task_queue.task_done()

            # put the results into the results queue with it's task
            # index so we can sort them later
            self.result_queue.put((task_idx, answer))

class GPUQueueMapper(object):

    def __init__(self, gpu_indices):
        self.gpu_indices = gpu_indices

        # initialize the func to None, this must be set a
        self.func = None


    def map(self, tasks):
        


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
