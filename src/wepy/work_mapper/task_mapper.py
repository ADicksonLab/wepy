import multiprocessing as mp
import time
import logging
from warnings import warn

from wepy.work_mapper.mapper import ABCWorkerMapper

class TaskMapper(ABCWorkerMapper):
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

    def __init__(self,
                 walker_task_type=None,
                 num_workers=None,
                 segment_func=None,
                 **kwargs):

        super().__init__(num_workers=num_workers,
                         segment_func=segment_func,
                         **kwargs)
        # choose the type of the worker
        if walker_task_type is None:
            self._walker_task_type_type = WalkerTaskProcess
            warn("walker_task_type not given using the default base class")
            logging.warn("walker_task_type not given using the default base class")
        else:
            self._walker_task_type = walker_task_type

        # initialize a list to put results in
        self.results = None


    @property
    def walker_task_type(self):
        """The callable that generates a worker object.

        Typically this is just the type from the class definition of
        the Worker where the constructor is called.

        """
        return self._walker_task_type



    def map(self, *iterables):

        # run computations in a Manager context
        with mp.Manager() as manager:

            # to manage access to worker resources we use a queue with
            # the index of the worker
            worker_queue = manager.Queue()

            # put the workers onto the queue
            for worker_idx in range(self.num_workers):
                worker_queue.put(worker_idx)

            # initialize segment times for workers to
            # fill in
            worker_segment_times = manager.dict()

            # initialize for the number of workers, since these will be
            # the slots to put timing results in
            for i in range(self.num_workers):
                worker_segment_times[i] = []

            # make a shared list for the walker results
            results = manager.list()

            # since this will be indexed by walker index initialize the
            # length of the array
            for walker in range(len(args)):
                results.append(None)

            # create the task based processes
            walker_processes = []
            for walker_idx, task_args in enumerate(zip(*args)):

                # start a process for this walker
                walker_process = self.walker_task_type(walker_idx, self._attributes,
                                                       self._func, task_args,
                                                       worker_queue,
                                                       results,
                                                       worker_segment_times)

                walker_process.start()

                walker_processes.append(walker_process)

            # then block until they have all finished
            for p in walker_processes:
                p.join()

            # save the managed list of the recorded worker times locally
            for key, val in worker_segment_times.items():
                self._worker_segment_times[key] = val

            new_walkers = [result for result in results]

        return new_walkers


class WalkerTaskProcess(mp.Process):

    NAME_TEMPLATE = "Walker-{}"

    def __init__(self, walker_idx, mapper_attributes,
                 func, task_args,
                 worker_queue, results_list, worker_segment_times,
                 **kwargs
                 ):

        # initialize the process customizing the name
        mp.Process.__init__(self, name=self.NAME_TEMPLATE.format(walker_idx))

        # the idea with this TaskProcess thing is that we pass in all
        # the data to the constructor to create a "thunk" (a closure
        # that is ready to be run without arguments) and then when run
        # is called there will be no arguments to be passed. This
        # simplifies the flow of data and underscores that the task is
        # the process.

        # task arguments
        self.func = func
        self.task_args = task_args

        # set the managed datastructure proxies as an attribute so we
        self.walker_idx = walker_idx
        self.mapper_attributes = mapper_attributes
        self.free_workers = free_workers
        self.worker_segment_times = worker_segment_times

        # the variable for the current worker index this is supposed
        # to be using, transient
        self._worker_idx = None


        self._attributes = kwargs

    @property
    def attributes(self, key):
        return self._attributes

    @attributes.getter
    def attributes(self, key):
        return self._attributes[key]

    def run_task(self):

        # for the vanilla work mapper we just use the raw results from
        # the func. In practice this is the result of run segment,
        # which is going to be a walker. This can be customized for
        # performance reasons if you want.

        return self.func(self.task_args)


    def run(self):

        walker_process = mp.current_process()
        logging.info("Walker process started as name: {}; PID: {}".format(walker_process.name,
                                                                          walker_process.pid))

        # lock a worker, then pop it off the queue so no other process
        # tries to use it

        # pop off a worker to use it
        self._worker_idx = self.free_workers.get()

        logging.info("{}: acquired worker {}".format(walker_process.name,
                                                    worker_idx))

        # generate the task thunk
        task = self._make_task(*self.task_args)

        # run the task
        start = time.time()
        logging.info("{}: running function. Time: {}".format(walker_process.name,
                                                      time.time()))


        result = self.run_task(task)
        logging.info("{}: finished running function. Time {}".format(walker_process.name,
                                                                     time.time()))
        end = time.time()

        # we separately set the result to the results list. This is so
        # we can debug performance problems, and dissect what is due
        # to computation time and what is due to communication
        logging.info("{}: Setting value to results list. Time {}".format(walker_process.name,
                                                                     time.time()))
        self.results[self.walker_idx] = result
        logging.info("{}: Finished setting value to results list. Time {}".format(walker_process.name,
                                                                                  time.time()))

        # put the worker back onto the queue since we are done using it
        self.free_workers.put(self._worker_idx)
        self._worker_idx = None

        logging.info("{}: released worker {}".format(walker_process.name,
                                                     worker_idx))

        # add the time for this segment to the collection of the worker times
        segment_time = end - start
        seg_times = self.worker_segment_times[worker_idx] + segment_time

        # we must explicitly set the new value in total to trigger an
        # update of the real dictionary. In place modification of
        # proxy objects has no effect
        self.worker_segment_times[worker_idx] = seg_times

