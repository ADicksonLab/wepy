"""Reference implementations, abstract base classes, and a production
ready worker style mapper for mapping runner dynamics to walkers for
wepy simulation cycles.

"""

import multiprocessing as mp
from multiprocessing import Queue, JoinableQueue
import queue
import time
import logging
from warnings import warn

from wepy.work_mapper.worker import Worker, Task

PY_MAP = map

class ABCMapper(object):
    """Abstract base class for a Mapper. Useful only for showing the
    interface stubs."""

    def __init__(self, **kwargs):
        raise NotImplementedError

    def init(self, **kwargs):
        raise NotImplementedError

    def cleanup(self, **kwargs):
        raise NotImplementedError

    def map(self, **kwargs):
        raise NotImplementedError


class Mapper(object):
    """Basic non-parallel reference implementation of a mapper."""

    def __init__(self, segment_func=None, *args, **kwargs):
        """Constructor for the Mapper class. No arguments are required.

        Parameters
        ----------
        segment_func : callable, optional
            Set a default segment_func. Typically set at runtime.

        """

        self._segment_func = segment_func
        self._worker_segment_times = {0 : []}

    def init(self, segment_func=None, **kwargs):
        """Runtime initialization and setting of function to map over walkers.

        Parameters
        ----------
        segment_func : callable implementing the Runner.run_segment interface

        """

        if segment_func is None:
            ValueError("segment_func must be given")

        self._func = segment_func

    @property
    def segment_func(self):
        """The function that will be called for new data in the `map` method."""
        return self._func

    def cleanup(self, **kwargs):
        """Runtime post-simulation tasks.

        This is run either at the end of a successful simulation or
        upon an error in the main process of the simulation manager
        call to `run_cycle`.

        The Mapper class performs no actions here and all arguments
        are ignored.

        """

        # nothing to do
        pass

    def map(self, *args):
        """Map the 'segment_func' to args.

        Parameters
        ----------
        *args : list of list
            Each element is the argument to one call of 'segment_func'.

        Returns
        -------
        results : list
            The results of each call to 'segment_func' in the same order as input.

        Examples
        --------

        >>> Mapper(segment_func=sum).map([(0,1,2), (3,4,5)])
        [3, 12]

        """

        args = [list(arg) for arg in args]

        segment_times = []
        results = []
        for arg_idx in range(len(args[0])):
            start = time.time()
            result = self._func(*[arg[arg_idx] for arg in args])
            end = time.time()
            segment_time = end - start
            segment_times.append(segment_time)

            results.append(result)

        self._worker_segment_times[0] = segment_times

        return results

    @property
    def worker_segment_times(self):
        """The run timings for each segment for each walker.

        Returns
        -------
        worker_seg_times : dict of int : list of float
            Dictionary mapping worker indices to a list of times in
            seconds for each segment run.

        """
        return self._worker_segment_times

class WorkerMapper(Mapper):
    """Work mapper implementation using multiple worker processes and task
    queue.

    Uses the python multiprocessing module to spawn multiple worker
    processes which watch a task queue of walker segments.
    """

    def __init__(self, num_workers=None, worker_type=None,
                 worker_attributes=None, **kwargs):
        """Constructor for WorkerMapper.

        kwargs are ignored.


        Parameters
        ----------
        num_workers : int
            The number of worker processes to spawn.

        worker_type : callable, optional
            Callable that generates an object implementing the Worker
            interface, typically a type from a Worker class.

        worker_attributes : dictionary
            A dictionary of values that are passed to the worker
            constructor as key-word arguments.

        """

        if worker_attributes is not None:
            self._worker_attributes = worker_attributes
        else:
            self._worker_attributes = {}

        self._num_workers = num_workers
        self._worker_segment_times = {i : [] for i in range(self.num_workers)}

        # choose the type of the worker
        if worker_type is None:
            self._worker_type = Worker
            warn("worker_type not given using the default base class")
            logging.warn("worker_type not given using the default base class")
        else:
            self._worker_type = worker_type

    @property
    def num_workers(self):
        """The number of worker processes."""
        return self._num_workers

    # TODO remove after testing
    # @num_workers.setter
    # def num_workers(self, num_workers):
    #     """Setter for the number of workers

    #     Parameters
    #     ----------
    #     num_workers : int

    #     """
    #     self._num_workers = num_workers

    @property
    def worker_type(self):
        """The callable that generates a worker object.

        Typically this is just the type from the class definition of
        the Worker where the constructor is called.

        """
        return self._worker_type

    # TODO remove after testing
    # @worker_type.setter
    # def worker_type(self, worker_type):
    #     """

    #     Parameters
    #     ----------
    #     worker_type :

    #     Returns
    #     -------

    #     """
    #     self._worker_type = worker_type

    def init(self, num_workers=None, **kwargs):
        """Runtime initialization and setting of function to map over walkers.

        Parameters
        ----------
        num_workers : int
            The number of worker processes to spawn

        segment_func : callable implementing the Runner.run_segment interface

        """

        super().init(**kwargs)

        # the number of workers must be given here or set as an object attribute
        if num_workers is None and self.num_workers is None:
            raise ValueError("The number of workers must be given, received {}".format(num_workers))

        # if the number of walkers was given for this init() call use
        # that, otherwise we use the default that was specified when
        # the object was created
        elif num_workers is None and self.num_workers is not None:
            num_workers = self.num_workers

        # Establish communication queues
        self._task_queue = JoinableQueue()
        self._result_queue = Queue()

        # Start workers, giving them all the queues
        self._workers = []
        for i in range(num_workers):
            worker = self.worker_type(i, self._task_queue, self._result_queue,
                                      **self._worker_attributes)
            self._workers.append(worker)

        # start the worker processes
        for worker in self._workers:
            worker.start()

            logging.info("Worker process started as name: {}; PID: {}".format(worker.name,
                                                                              worker.pid))

    def cleanup(self, **kwargs):
        """Runtime post-simulation tasks.

        This is run either at the end of a successful simulation or
        upon an error in the main process of the simulation manager
        call to `run_cycle`.

        The Mapper class performs no actions here and all arguments
        are ignored.

        """

        # send poison pills (Stop signals) to the queues to stop them in a nice way
        # and let them finish up
        for i in range(self.num_workers):
            self._task_queue.put((None, None))

        # delete the queues and workers
        self._task_queue = None
        self._result_queue = None
        self._workers = None

    def _make_task(self, *args, **kwargs):
        """Generate a task from 'segment_func' attribute.

        Similar to partial evaluation (or currying).

        Args will be eventually used as the arguments to the call of
        'segment_func' by the worker processes when they receive the
        task from the queue.

        Returns
        -------
        task : Task object

        """
        return Task(self._func, *args, **kwargs)

    def map(self, *args):
        # docstring in superclass

        map_process = mp.current_process()
        logging.info("Mapping from process {}; PID {}".format(map_process.name, map_process.pid))

        # make tuples for the arguments to each function call
        task_args = zip(*args)

        num_tasks = len(args[0])
        # Enqueue the jobs
        for task_idx, task_arg in enumerate(task_args):

            # a task will be the actual task and its task idx so we can
            # sort them later
            self._task_queue.put((task_idx, self._make_task(*task_arg)))


        logging.info("Waiting for tasks to be run")

        # Wait for all of the tasks to finish
        self._task_queue.join()

        # workers_done = [worker.done for worker in self._workers]

        # if all(workers_done):

        # get the results out in an unordered way. We rely on the
        # number of tasks we know we put out because if you just try
        # to get from the queue until it is empty it will just wait
        # forever, since nothing is there. ALternatively it is risky
        # to implement a wait timeout or no wait in case there is a
        # small wait time.
        logging.info("Retrieving results")

        n_results = num_tasks
        results = []
        while n_results > 0:

            logging.info("trying to retrieve result: {}".format(n_results))

            result = self._result_queue.get()
            results.append(result)

            logging.info("Retrieved result {}: {}".format(n_results, result))

            n_results -= 1

        logging.info("No more results")

        logging.info("Retrieved results")

        # sort the results according to their task_idx
        results.sort()

        # save the task run times, so they can be accessed if desired,
        # after clearing the task times from the last mapping
        self._worker_segment_times = {i : [] for i in range(self.num_workers)}
        for task_idx, worker_idx, task_time, result in results:
            self._worker_segment_times[worker_idx].append(task_time)

        # then just return the values of the function
        return [result for task_idx, worker_idx, task_time, result in results]
