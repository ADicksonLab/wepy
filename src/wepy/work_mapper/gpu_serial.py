import multiprocessing as mp
import time
import logging

from wepy.work_mapper.mapper import Mapper
from wepy.runners.openmm import OpenMMState, OpenMMWalker

# TODO remove this so it isn't coupled to OpenMM
# import simtk.openmm as omm

class TaskMapper(Mapper):
    """Process-per-task mapper.

    This method of work mapper starts new processes for each runner
    segment task that needs to be run. This allows cheap copying of
    shared state using the operating system primitives. On linux this
    would be either 'fork' (default) or 'spawn'. Fork is cheap but
    doesn't initialize certain process namespace things, whereas spawn
    is much more expensive but properly cleans things up. Fork should
    be sufficient in most cases, however spawn may be needed when you
    have some special contexts in the parent process. This is the case
    with starting CUDA contexts in the main parent process and then
    forking new processes from it. We suggest using fork and avoiding
    making these kinds of contexts in the main process.

    This method avoids using shared memory or sending objects through
    interprocess communication (that has a serialization and
    deserialization cost associated with them) by using OS copying
    mechanism. However, a new process will be created each cycle for
    each walker in the simulation. So if you want a large number of
    walkers you may experience a large overhead. If your walker states
    are very small or a very fast serializer is available you may also
    not benefit from full process address space copies. Instead the
    WorkerMapper may be better suited.

    """

    def __init__(self, n_walkers, gpu_indices=None, num_workers=None, **kwargs):

        self.n_walkers = n_walkers
        if gpu_indices is not None:
            self.gpu_indices = gpu_indices
            self.n_workers = len(gpu_indices)
        else:
            assert num_workers, "If gpu_indices are not given the n_workers must be given"
            self.n_workers = num_workers
            self.gpu_indices = range(num_workers)



    def init(self, **kwargs):

        super().init(**kwargs)

        # make a Queue for free workers, when one is being used it is
        # popped off and locked
        self.free_workers = mp.Queue()
        # the semaphore provides the locks on the workers
        self.lock = mp.Semaphore(self.n_workers)
        # initialize a list to put results in
        self.results_list = mp.Manager().list()
        for i in range(self.n_walkers):
            self.results_list.append(None)

        self._worker_segment_times = mp.Manager().dict()
        self._worker_segment_times = {i : [] for i in range(self.n_workers)}
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

        # call the function setting the appropriate properties of the
        # platform
        start = time.time()
        new_walker = _call_(*args, DeviceIndex=str(gpu_idx))
        self.results_list[index] = new_walker.state.sim_state.__getstate__()
        end = time.time()

        self.free_workers.put(worker_idx)
        seg_time_dic[worker_idx] = seg_time_dic[worker_idx] + [end - start]

        self.lock.release()

    def map(self, *iterables):

        # initialize segment times and results list for workers to
        # fill in
        manager = mp.Manager()

        worker_segment_times = manager.dict()

        for i in range(self.n_workers):
            worker_segment_times.update({i:[]})
            self.results_list[i] = None


        walkers_pool = []
        weights = []
        # create processes and start to run
        index = 0

        for args in zip(*iterables):
            weights.append(args[0].weight)
            walkers_pool.append(mp.Process(target=self.exec_call,
                                                args=(self._func, index, *args)))
            index += 1

        for p in walkers_pool:
            p.start()

        for p in walkers_pool:
            p.join()

        self._worker_segment_times = worker_segment_times

        # rebuild walkers
        new_walkers = []
        for i,w in enumerate(self.results_list):
            new_omm_state = omm.State()
            new_omm_state.__setstate__(w)
            new_walkers.append(OpenMMWalker(OpenMMState(new_omm_state),weights[i]))

        return new_walkers
