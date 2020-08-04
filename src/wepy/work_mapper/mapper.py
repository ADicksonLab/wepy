"""Reference implementations, abstract base classes, and a production
ready worker style mapper for mapping runner dynamics to walkers for
wepy simulation cycles.

"""
import sys
import multiprocessing as mp
import queue as pyq
import traceback
import time
from warnings import warn
import signal

import logging
from eliot import start_action, log_call

PY_MAP = map

class ABCMapper(object):
    """Abstract base class for a Mapper."""

    def __init__(self, segment_func=None, **kwargs):
        """Constructor for the Mapper class. No arguments are required.

        Parameters
        ----------
        segment_func : callable, optional
            Set a default segment_func. Typically set at runtime.

        """

        self._func = segment_func

        self._attributes = kwargs

    @property
    def attributes(self):
        return self._attributes

    @attributes.getter
    def attributes(self, key):
        return self._attributes[key]

    @log_call(include_args=[],
              include_result=False)
    def init(self, segment_func=None, **kwargs):
        """Runtime initialization and setting of function to map over walkers.

        Parameters
        ----------
        segment_func : callable implementing the Runner.run_segment interface

        """


        if self.segment_func is not None and segment_func is not None:
            logging.info("overriding default segment_func {} with {}".format(self._func, segment_func))
            self._func = segment_func

        elif self.segment_func is None and segment_func is None:
            ValueError("segment_func must be given since no default specified")

        elif self.segment_func is None and segment_func is not None:
            self._func = segment_func


    @property
    def segment_func(self):
        """The function that will be called for new data in the `map` method."""
        return self._func

    @log_call(include_args=[],
              include_result=False)
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

    @log_call(include_args=[],
              include_result=False)
    def map(self, *args, **kwargs):
        raise NotImplementedError



class Mapper(ABCMapper):
    """Basic non-parallel reference implementation of a mapper."""

    def __init__(self, segment_func=None, **kwargs):
        """Constructor for the Mapper class. No arguments are required.

        Parameters
        ----------
        segment_func : callable, optional
            Set a default segment_func. Typically set at runtime.

        """

        super().__init__(segment_func=segment_func, **kwargs)


        self._worker_segment_times = {0 : []}

    @log_call(include_args=[],
              include_result=False)
    def map(self, *args, **kwargs):
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

        # expand the generators for the args and kwargs
        args = [list(arg) for arg in args]
        kwargs = {key : list(kwarg) for key, kwarg in kwargs.items()}

        segment_times = []
        results = []
        for arg_idx in range(len(args[0])):
            start = time.time()

            # get just the args for this call to func
            call_args = [arg[arg_idx] for arg in args]
            call_kwargs = {key : value[arg_idx] for key, value in kwargs.items()}

            # run the task, catch any errors and reraise as a
            # TaskException to satisfy the pattern
            try:

                result = self._func(*call_args, **call_kwargs)

            except Exception as task_exception:

                # get the traceback for the exception
                tb = sys.exc_info()[2]

                msg = "Exception '{}({})' caught in a task.".format(
                                       type(task_exception).__name__, task_exception)
                traceback_log_msg = \
                    """Traceback:
--------------------------------------------------------------------------------
{}
--------------------------------------------------------------------------------
                    """.format(''.join(traceback.format_exception(
                        type(task_exception), task_exception, tb)),
                    )

                logging.critical(msg + '\n' + traceback_log_msg)

                # raise a TaskException to distinguish it from the worker
                # errors with the metadata about the original exception

                raise TaskException("Error occured during task execution, recovery not possible.",
                                wrapped_exception=task_exception,
                                tb=tb)

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


class Task(object):
    """Class that composes a function and arguments."""

    def __init__(self, func, *args, **kwargs):
        """Constructor for Task.

        Parameters
        ----------
        func : callable
            Function to be called on the arguments.

        *args
            The arguments to pass to func

        """
        self.args = args
        self.kwargs = kwargs
        self.func = func

    def __call__(self, **worker_kwargs):
        """Makes the Task itself callable."""

        # run the function passing in the args for running it and any
        # worker information in the worker kwargs.
        return self.func(*self.args, **self.kwargs, **worker_kwargs)

class WrapperException(Exception):
    """Exception used for wrapping another exception.

    Since tracebacks can't be pickled we format it and save that
    instead.

    """

    def __init__(self, message,
                 # must be kwargs so we can pickle it (I know weird...)
                 wrapped_exception=None,
                 tb=None):
        super().__init__(message)

        # save the exception with the traceback
        self.wrapped_exception = wrapped_exception
        self.formatted_tb = traceback.format_tb(tb)


class TaskException(WrapperException):
    pass

class ABCWorkerMapper(ABCMapper):

    def __init__(self,
                 num_workers=None,
                 segment_func=None,
                 proc_start_method='fork',
                 **kwargs):
        """Constructor for WorkerMapper.


        Parameters
        ----------
        num_workers : int
            The number of worker processes to spawn.

        segment_func : callable, optional
            Set a default segment_func. Typically set at runtime.

        proc_start_method : str or None
            A string indicating the type of process start method to
            use from python multiprocessing typically 'fork', 'spawn',
            or 'forkserver', or the platform default for None. See
            documentation. Generates a context with the method
            multiprocessing.get_context(proc_start_method) on `init`.

        """

        super().__init__(segment_func=segment_func, **kwargs)


        self._proc_start_method = proc_start_method

        self._num_workers = num_workers
        self._worker_segment_times = None

        if num_workers is not None:
            self._worker_segment_times = {i : [] for i in range(self.num_workers)}

    def init(self, num_workers=None, segment_func=None,
             **kwargs):
        """Runtime initialization and setting of function to map over walkers.

        Parameters
        ----------
        num_workers : int
            The number of worker processes to spawn

        segment_func : callable implementing the Runner.run_segment interface

        """

        super().init(segment_func=segment_func)

        # create the multiprocessing context to use for spawning
        # processes here
        self._mp_ctx = mp.get_context(method=self._proc_start_method)

        # the number of workers must be given here or set as an object attribute
        if num_workers is None and self.num_workers is None:
            raise ValueError("The number of workers must be given, received {}".format(num_workers))

        # if the number of walkers was given for this init() call use
        # that, otherwise we use the default that was specified when
        # the object was created
        elif num_workers is not None and self.num_workers is None:
            self._num_workers = num_workers

        # update the worker segment times
        self._worker_segment_times = {i : [] for i in range(self.num_workers)}


    def cleanup(self, **kwargs):

        # ALERT: is this all we need to do? I have a hunch there is
        # more caveats, but these context objects are not really
        # documented

        # make sure the context for this work mapper is destroyed
        del self._mp_ctx

    @property
    def num_workers(self):
        """The number of worker processes."""
        return self._num_workers

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


# ----------------------------------
# everything below this logically belongs in worker.py and should be imported from there


class WorkerException(WrapperException):
    pass

class WorkerKilledError(ChildProcessError):
    pass


# TODO: move this class to the wepy.work_mapper.worker class where it
# belongs. It shouldn't be in this namespace, but we will leave it
# here. Furthermore I would like to rename it since we now have
# different worker mapper implementations with different concurrency
# models
class WorkerMapper(ABCWorkerMapper):
    """Work mapper implementation using multiple worker processes and task
    queue.

    Uses the python multiprocessing module to spawn multiple worker
    processes which watch a task queue of walker segments.
    """

    def __init__(self,
                 num_workers=None,
                 worker_type=None,
                 worker_attributes=None,
                 segment_func=None,
                 **kwargs):
        """Constructor for WorkerMapper.


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

        segment_func : callable, optional
            Set a default segment_func. Typically set at runtime.

        """

        super().__init__(num_workers=num_workers,
                         segment_func=segment_func,
                         **kwargs)

        # since the workers will be their own process classes we
        # handle this data

        # attributes that will be passed to the worker constructors
        if worker_attributes is not None:
            self._worker_attributes = worker_attributes
        else:
            self._worker_attributes = {}

        # choose the type of the worker
        if worker_type is None:
            self._worker_type = Worker
            warn("worker_type not given using the default base class")
            logging.warn("worker_type not given using the default base class")
        else:
            self._worker_type = worker_type

    @property
    def worker_type(self):
        """The callable that generates a worker object.

        Typically this is just the type from the class definition of
        the Worker where the constructor is called.

        """
        return self._worker_type


    def init(self, num_workers=None, segment_func=None,
             **kwargs):
        """Runtime initialization and setting of function to map over walkers.

        Parameters
        ----------
        num_workers : int
            The number of worker processes to spawn

        segment_func : callable implementing the Runner.run_segment interface

        """

        super().init(num_workers=num_workers,
                     segment_func=segment_func,
                     **kwargs)

        manager = self._mp_ctx.Manager()

        # Establish communication queues

        # A queue for errors
        self._exception_queue = manager.Queue()

        # queue for the tasks we know the batch size so we don't need
        # a JoinableQueue
        self._task_queue = manager.Queue()

        # results queue
        self._result_queue = manager.Queue()

        # use pipes for communication channels between this parent
        # process and the children for sending specific interrupts
        # such as the signal to kill them. Note that the clean way to
        # end the process is to send poison pills on the task queue,
        # this is for other stuff. IRQ is a common abbreviation for
        # interrupts
        self._irq_parent_conns = []

        # Start workers, giving them all the queues
        self._workers = []
        for i in range(self.num_workers):

            # make a pipe to communicate with this worker for the int
            parent_conn, child_conn = self._mp_ctx.Pipe()
            self._irq_parent_conns.append(parent_conn)

            # create the worker giving it all of the communication
            # channels
            worker = self.worker_type(i,
                                      self._task_queue,
                                      self._result_queue,
                                      self._exception_queue,
                                      child_conn,
                                      mapper_attributes=self._attributes,
                                      **self._worker_attributes)
            self._workers.append(worker)

        # start the worker processes
        for worker in self._workers:
            worker.start()

            logging.info("Worker process started as name: {}; PID: {}".format(worker.name,
                                                                              worker.pid))

        # now that we have started the processes register the handler
        # for SIGTERM signals that will clean up our children cleanly
        signal.signal(signal.SIGTERM, self._sigterm_shutdown)

    def _sigterm_shutdown(self, signum, frame):

        logging.critical("Received external SIGTERM, forcing shutdown.")

        self.force_shutdown()

    def force_shutdown(self, **kwargs):

        logging.critical("Forcing shutdown")

        # our primary job is to shut down all of the running processes
        # without just shutting down the queues and breaking the pipes

        # to do this we send the kill signals to them on the kill
        # channel.

        for worker_idx, worker in enumerate(self._workers):

            logging.critical("Sending SIGTERM message on {} to worker {}".format(
                self._irq_parent_conns[worker_idx].fileno(), worker_idx))

            # send a kill message to the worker
            self._irq_parent_conns[worker_idx].send(signal.SIGTERM)

        logging.critical("All kill messages sent to workers")



        # check that all have exited
        alive_workers = [worker.is_alive() for worker in self._workers]
        worker_acks = {}
        worker_exitcodes = {}
        premature_exit = False
        while any(alive_workers) and not premature_exit:

            for worker_idx, worker in enumerate(self._workers):

                # ignore already known dead workers
                if not alive_workers[worker_idx]:
                    continue

                if worker.is_alive():

                    # if it is still alive and we have an ack from it
                    # just terminate. There is a bug in the code and
                    # is out of our control
                    if worker_idx in worker_acks:
                        logging.debug("Ack received from {} but has not shut down".format(worker.name))
                        premature_exit = True

                    # otherwise we need to try and receive the ack
                    elif self._irq_parent_conns[worker_idx].poll(1):

                        # receive the acknowledgement
                        ack = self._irq_parent_conns[worker_idx].recv()

                        logging.debug("Received {} acknowledgement from {}".format(ack,
                                                                                   worker.name))

                        # make sure the ack is affirmative
                        if ack is True:
                            worker_acks[worker_idx] = ack

                        # if it is an exeption wrap it as a worker
                        # error and use the os to kill the process
                        elif issubclass(type(ack), Exception):

                            # wrap it as a worker exception
                            exception = WorkerException(wrapped_exception=ack)
                            worker_acks[worker_idx] = exception

                            logging.critical("{} not responding, terminating with SIGTERM".format(
                                worker.name))

                            worker.terminate()


                else:
                    alive_workers[worker_idx] = False
                    worker_exitcodes[worker_idx] = worker.exitcode

        if any(alive_workers):
            logging.critical("Terminating main process with running workers {}".format(
                ','.join([str(worker_idx) for worker_idx in range(len(self._workers))
                          if alive_workers[worker_idx]])))




    def cleanup(self, **kwargs):
        """Runtime post-simulation tasks.

        This is run either at the end of a successful simulation or
        upon an error in the main process of the simulation manager
        call to `run_cycle`.

        The Mapper class performs no actions here and all arguments
        are ignored.

        """

        super().cleanup(**kwargs)

        # send poison pills (Stop signals) to the queues to stop them in a nice way
        # and let them finish up
        for i in range(self.num_workers):
            self._task_queue.put((None, None))

        # delete the queues and workers
        self._task_queue = None
        self._result_queue = None
        self._workers = None


    def map(self, *args, **kwargs):
        # docstring in superclass

        map_process = self._mp_ctx.current_process()
        logging.info("Mapping from process {}; PID {}".format(map_process.name, map_process.pid))

        # make tuples for the arguments to each function call
        task_args = zip(*args)
        kwargs = {key : list(kwarg) for key, kwarg in kwargs.items()}

        num_tasks = len(args[0])
        # Enqueue the jobs
        for task_idx, task_arg in enumerate(task_args):

            task_kwargs = {key : value[task_idx] for key, value in kwargs.items()}

            # a task will be the actual task and its task idx so we can
            # sort them later
            self._task_queue.put((task_idx, self._make_task(*task_arg, **task_kwargs)))


        logging.info("Waiting for tasks to be run")

        # poll the exception and result queues for results
        n_results_left = num_tasks
        results = []
        while n_results_left > 0:

            # first check if any errors came back but don't wait,
            # since the methods for querying whether it is empty or
            # not are not reliable we just try and if we don't get
            # anything we will come back around
            try:
                proc_name, pid, exception = self._exception_queue.get_nowait()
            except pyq.Empty:
                pass

            else:

                logging.error("Exception occured in process {}; pid {}.".format(
                    proc_name, pid))

                # we can handle Task and Worker exceptions differently
                if type(exception) == TaskException:

                    logging.critical("Exception encountered in a task which is unrecoverable."
                                "You will need to reconfigure your components in a stable manner.")


                    self.force_shutdown()

                    logging.critical("Shutdown complete.")
                    raise exception

                elif type(exception) == WorkerException:

                    # we make just an error message to say that errors
                    # in the worker may be due to the network or
                    # something and could recover
                    logging.error("Exception encountered in the work mapper worker process."
                                  "Recovery possible, see further messages.")

                    # However, the current implementation doesn't
                    # support retries or whatever so we issue a
                    # critical log informing that it has been elevated
                    # to critical and will force shutdown
                    logging.critical("Worker error mode resiliency not supported at this time."
                                     "Performing force shutdown and simulation ending.")

                    self.force_shutdown()

                    logging.critical("Shutdown complete.")
                    raise exception

                else:
                    logging.critical("Unknown exception encountered.")

                    self.force_shutdown()

                    logging.critical("Shutdown complete.")

                    raise exception


            # attempt to get something off of the results queue
            try:
                result = self._result_queue.get_nowait()
            except pyq.Empty:
                pass

            # if we get something handle it
            else:

                logging.info("Retrieved result: {}".format(result))
                results.append(result)

                # reduce the counter so we know when we are done
                n_results_left -= 1

        # sort the results according to their task_idx
        results.sort()

        # save the task run times, so they can be accessed if desired,
        # after clearing the task times from the last mapping

        # DEBUG: removing this because it should be set on init()
        #self._worker_segment_times = {i : [] for i in range(self.num_workers)}

        for task_idx, worker_idx, task_time, result in results:
            self._worker_segment_times[worker_idx].append(task_time)

        # then just return the values of the function
        return [result for task_idx, worker_idx, task_time, result in results]


# same for the worker in terms of refactoring
class Worker(mp.Process):
    """Worker process.

    This is a subclass of process with an overriden `__init__`
    constructor that will automatically generate the Process.

    When this class is constructed a new process will be formed.

    """

    NAME_TEMPLATE = "Worker-{}"
    """A string formatting template to identify worker processes in
    logs. The field will be filled with the worker index."""

    def __init__(self, worker_idx,
                 task_queue,
                 result_queue,
                 exception_queue,
                 interrupt_connection,
                 mapper_attributes=None,
                 **kwargs):
        """Constructor for the Worker class.

        Parameters
        ----------
        worker_idx : int
            The index of the worker. Should be unique.

        task_queue : multiprocessing.JoinableQueue
            The shared task queue the worker will watch for new tasks to complete.

        result_queue : multiprocessing.Queue
            The shared queue that completed task results will be placed on.

        interrupt_connection : multiprocessing.Connection
            One end of a pipe to listen for messages specific to this worker.

        mapper_attributes : None or dict
            A dictionary of the attributes of the mapper for reference in workers.

        kwargs :
            The worker specific attributes

        """

        # call the Process constructor
        mp.Process.__init__(self, name=self.NAME_TEMPLATE.format(worker_idx))

        self._exception_queue = exception_queue
        self._exception = None
        self._traceback = None

        # the queue that will trigger a shutdown in the event of failure
        self._irq_channel = interrupt_connection

        # also register the SIGTERM signal handler for graceful
        # shutdown with reporting to mapper
        signal.signal(signal.SIGTERM, self._sigterm_shutdown)

        self._worker_idx = worker_idx

        self._mapper_attributes = mapper_attributes

        # set all the kwargs into an attributes dictionary
        self._attributes = kwargs

        # the queues for work to be done and work done
        self._task_queue = task_queue
        self._result_queue = result_queue

        logging.debug("{} process created".format(self.name))


    @property
    def worker_idx(self):
        """Dictionary of attributes of the worker."""
        return self._worker_idx

    @property
    def attributes(self):
        """Dictionary of attributes of the worker."""
        return self._attributes

    @property
    def mapper_attributes(self):
        """Dictionary of attributes of the worker."""
        return self._mapper_attributes

    def run(self):

        logging.debug("{}: starting to run".format(self.name))

        # try to run the worker and it's task, except either class of
        # error that can come from it either from the worker
        # (WorkerException) or the task (TaskException) and communicate it
        # back to the main process

        # if we get an exception there is some cleanup logic
        run_exception = None

        try:

            # run the worker, which will retrieve its task from the
            # queue attempt to run the task, and if it succeeds will
            # put the results on the result queue, if the task fails
            # it will catch it and wrap it as a task exception
            self._run_worker()

        except TaskException as task_exception:

            logging.error("{}: TaskException caught".format(self.name))

            run_exception = task_exception

        # anything else is considered a WorkerException so take the
        # original exception and generate a worker exception from that
        except Exception as exception:

            logging.debug("{}: WorkerError caught".format(self.name))


            # get the traceback
            tb = sys.exc_info()[2]

            msg = "Exception '{}({})' caught in a worker.".format(
                                   type(exception).__name__, exception)
            traceback_log_msg = \
                """Traceback:
--------------------------------------------------------------------------------
{}
--------------------------------------------------------------------------------
                """.format(''.join(traceback.format_exception(
                    type(exception), exception, tb)),
                )

            logging.error("{}:".format(self.name) + msg + '\n' + traceback_log_msg)

            # raise a TaskError to distinguish it from the worker
            # errors with the metadata about the original exception

            worker_exception = WorkerException(
                "Error occured during worker execution.",
                wrapped_exception=exception,
                tb=tb)

            run_exception = worker_exception

        # raise worker_exception
        if run_exception is not None:

            logging.debug("{}: Putting exception on exception queue".format(self.name))

            # then put the exception and the traceback onto the queue
            # so we can communicate back to the parent process
            try:
                self._exception_queue.put((self.name, self.pid, run_exception))
            except BrokenPipeError as exc:
                logging.error(
                    "Pipe is broken indicating the root process has already exited:\n{}".format(
                        exc))

            # TODO: not sure if this is good or not
            # then reraise the exception so it can be caught
            # raise run_exception

    def _sigterm_shutdown(self, signum, frame):

        logging.debug("Received external SIGTERM kill command.")

        logging.debug("Alerting mapper that this will be honored.")

        # send an error to the mapper that the worker has been killed
        self._irq_channel.send(WorkerKilledError(
            "{} (pid: {}) killed by external SIGTERM signal".format(self.name, self.pid)))

        logging.debug("Acknowledgment sent")

        logging.debug("Shutting down process")

    def _shutdown(self):

        logging.debug("Received SIGTERM kill command from mapper")

        logging.debug("Acknowledging kill request will be honored")

        # report back that we are shutting down with a True
        self._irq_channel.send(True)

        logging.debug("Acknowledgment sent")

        logging.debug("Shutting down process")


    def _run_worker(self):

        # run the logic associated with communication and liveness of
        # the worker process itself, this is not necessarily a fatal
        # (critical) error and restarting a worker might resolve the
        # problem. This calls the _run_task method though which is
        # always critical since the logic in the code cannot be
        # disputed

        # TODO remove when confirmed that this works
        # worker_process = mp.current_process()
        logging.info("{}: Worker process started as name: {}; PID: {}".format(
            self.name, self.name, self.pid))

        while True:

            # check to see if there is any signals in the interrupt channel
            if self._irq_channel.poll():
                # get the message
                message = self._irq_channel.recv()

                logging.debug("{}: Received message from mapper on filehandle {}: {}".format(
                    self.name, self._irq_channel.fileno(), message))

                # handle the message

                # a SIGTERM is a signal to kill the process
                # unconditionally
                if message is signal.SIGTERM:

                    self._shutdown()

                    # break from the event (while) loop and shut down
                    break

                # anything is not recognized and we will continue and
                # report back that we don't recognize the message with
                # a ValueError object
                else:
                    logging.error("{}: Message not recognized, continuing operations and"
                                  " sending error to mapper".format(self.name))
                    self._irq_channel.send(
                        ValueError(
                            "Message: {} not recognized continuing operations".format(
                                message)))

            # get the next task
            try:
                 task_idx, next_task = self._task_queue.get(block=False, timeout=None)

                 logging.debug("{}: Got task {}".format(self.name, task_idx))

            except pyq.Empty:

                task_idx = None
                next_task = Ellipsis


            # # check for the poison pill which is the signal to stop
            if next_task is None:

                logging.info('{}: received {} {}: FINISHED'.format(
                    self.name, task_idx, next_task))

                # TODO remove since we aren't using joinble queue anymore
                # mark the poison pill task as done
                #self.task_queue.task_done()

                # and exit the loop
                break

            # only execute this if a task was actually receieved from
            # the queue; an Ellipsis indicates continue the loop
            elif next_task is not Ellipsis:

                logging.info('{}; task_idx : {}; args : {} '.format(
                    self.name, task_idx, next_task.args))

                # run the task
                start = time.time()

                answer = self._run_task(next_task)

                end = time.time()
                task_time = end - start

                logging.info('{}: task_idx : {}; COMPLETED in {} s'.format(
                    self.name, task_idx, task_time))

                # put the results into the results queue with it's task
                # index so we can sort them later
                self._result_queue.put((task_idx, self.worker_idx, task_time, answer))

    def run_task(self, task):
        """Actually executes the task.

        This default runner simply executes the task thunk.

        This can be customized by subclasses in order to allow for
        injection of worker specific data.

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


    def _run_task(self, task):
        """Runs the given task and returns the results.

        This manages handling exceptions and tracebacks from the
        actual `run_task` function which is intended to be specialized
        by different workers to inject worker specific arguments to
        tasks. Such as node and device identification.

        Parameters
        ----------
        task : Task object
            The partially evaluated task; function plus arguments

        Returns
        -------
        task_result
            Results of running the task.

        """

        logging.info("Running task")
        try:
            return self.run_task(task)

        except Exception as task_exception:

            # get the traceback for the exception
            tb = sys.exc_info()[2]

            msg = "Exception '{}({})' caught in a task.".format(
                                   type(task_exception).__name__, task_exception)
            traceback_log_msg = \
                """Traceback:
--------------------------------------------------------------------------------
{}
--------------------------------------------------------------------------------
                """.format(''.join(traceback.format_exception(
                    type(task_exception), task_exception, tb)),
                )

            logging.critical(msg + '\n' + traceback_log_msg)

            # raise a TaskException to distinguish it from the worker
            # errors with the metadata about the original exception

            raise TaskException("Error occured during task execution, recovery not possible.",
                            wrapped_exception=task_exception,
                            tb=tb)
