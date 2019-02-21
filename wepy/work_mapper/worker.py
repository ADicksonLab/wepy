"""Classes for workers and tasks for use with WorkerMapper."""

from multiprocessing import Process
import multiprocessing as mp
import time
import logging

class Worker(Process):
    """Worker process.

    This is a subclass of process with an overriden `__init__`
    constructor that will automatically generate the Process.

    When this class is constructed a new process will be formed.

    """

    NAME_TEMPLATE = "Worker-{}"
    """A string formatting template to identify worker processes in
    logs. The field will be filled with the worker index."""

    def __init__(self, worker_idx, task_queue, result_queue, **kwargs):
        """Constructor for the Worker class.

        Parameters
        ----------
        worker_idx : int
            The index of the worker. Should be unique.

        task_queue : multiprocessing.JoinableQueue
            The shared task queue the worker will watch for new tasks to complete.

        result_queue : multiprocessing.Queue
            The shared queue that completed task results will be placed on.

        """

        # call the Process constructor
        Process.__init__(self, name=self.NAME_TEMPLATE.format(worker_idx))

        self.worker_idx = worker_idx

        # set all the kwargs into an attributes dictionary
        self._attributes = kwargs

        # the queues for work to be done and work done
        self.task_queue = task_queue
        self.result_queue = result_queue

    @property
    def attributes(self):
        """Dictionary of attributes of the worker."""
        return self._attributes

    def run_task(self, task):
        """Runs the given task and returns the results.

        Parameters
        ----------
        task : Task object
            The partially evaluated task; function plus arguments

        Returns
        -------
        task_result
            Results of running the task.

        """

        return task()

    def run(self):
        """Overriding method for Process. Starts this process."""

        worker_process = mp.current_process()
        logging.info("Worker process started as name: {}; PID: {}".format(worker_process.name,
                                                                          worker_process.pid))

        while True:

            # get the next task
            task_idx, next_task = self.task_queue.get()

            # # check for the poison pill which is the signal to stop
            if next_task is None:

                logging.info('Worker: {}; received {} {}: FINISHED'.format(
                    self.name, task_idx, next_task))

                # mark the poison pill task as done
                self.task_queue.task_done()

                # and exit the loop
                break

            logging.info('Worker: {}; task_idx : {}; args : {} '.format(
                self.name, task_idx, next_task.args))

            # run the task
            start = time.time()
            answer = self.run_task(next_task)
            end = time.time()
            task_time = end - start

            logging.info('Worker: {}; task_idx : {}; COMPLETED in {} s'.format(
                self.name, task_idx, task_time))

            # (for joinable queue) tell the queue that the formerly
            # enqued task is complete
            self.task_queue.task_done()

            # put the results into the results queue with it's task
            # index so we can sort them later
            self.result_queue.put((task_idx, self.worker_idx, task_time, answer))


class Task(object):
    """Class that composes a function and arguments."""

    def __init__(self, func, *args):
        """Constructor for Task.

        Parameters
        ----------
        func : callable
            Function to be called on the arguments.

        *args
            The arguments to pass to func

        """
        self.args = args
        self.func = func

    def __call__(self, **kwargs):
        """Makes the Task itself callable."""

        # run the function passing in the args for running it and any
        # worker information in the kwargs
        return self.func(*self.args, **kwargs)
