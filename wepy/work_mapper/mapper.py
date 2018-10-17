import multiprocessing as mp
from multiprocessing import Queue, JoinableQueue
import queue
import time
import logging

from wepy.work_mapper.worker import Worker, Task

PY_MAP = map

class ABCMapper(object):

    def __init__(self, **kwargs):
        pass

    def init(self, **kwargs):
        pass

    def cleanup(self, **kwargs):
        pass

    def map(self, **kwargs):
        pass

class Mapper(object):

    def __init__(self, *args, **kwargs):
        self._worker_segment_times = {0 : []}

    def init(self, segment_func=None, **kwargs):

        if segment_func is None:
            ValueError("segment_func must be given")

        self._func = segment_func


    def cleanup(self, **kwargs):
        # nothing to do
        pass

    def map(self, *args, **kwargs):

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
        return self._worker_segment_times

class WorkerMapper(Mapper):

    def __init__(self, num_workers=None, worker_type=None,
                 **kwargs):

        self._num_workers = num_workers
        self._worker_segment_times = {i : [] for i in range(self.num_workers)}

        # choose the type of the worker
        if worker_type is None:
            self._worker_type = Worker
        else:
            self._worker_type = worker_type

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, num_workers):
        self._num_workers = num_workers

    @property
    def worker_type(self):
        return self._worker_type

    @worker_type.setter
    def worker_type(self, worker_type):
        self._worker_type = worker_type

    def init(self, num_workers=None, **kwargs):

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
        self._workers = [self.worker_type(i, self._task_queue, self._result_queue)
                         for i in range(num_workers)]

        # start the worker processes
        for worker in self._workers:
            worker.start()

            logging.info("Worker process started as name: {}; PID: {}".format(worker.name,
                                                                              worker.pid))

    def cleanup(self):

        # send poison pills (Stop signals) to the queues to stop them in a nice way
        # and let them finish up
        for i in range(self.num_workers):
            self._task_queue.put((None, None))

        # delete the queues and workers
        self._task_queue = None
        self._result_queue = None
        self._workers = None

    def make_task(self, *args, **kwargs):
        return Task(self._func, *args, **kwargs)

    def map(self, *args):

        map_process = mp.current_process()
        logging.info("Mapping from process {}; PID {}".format(map_process.name, map_process.pid))

        # make tuples for the arguments to each function call
        task_args = zip(*args)

        num_tasks = len(args[0])
        # Enqueue the jobs
        for task_idx, task_arg in enumerate(task_args):

            # a task will be the actual task and its task idx so we can
            # sort them later
            self._task_queue.put((task_idx, self.make_task(*task_arg)))


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
