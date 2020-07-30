import multiprocessing as mp
import queue as pyq
import time
import logging
from warnings import warn
import sys
import traceback
import signal
import pickle

from wepy.work_mapper.mapper import ABCWorkerMapper, WrapperException, TaskException, Task

from wepy.walker import Walker

class TaskProcessException(WrapperException):
    pass

class TaskProcessKilledError(ChildProcessError):
    pass

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
            self._walker_task_type = WalkerTaskProcess
            warn("walker_task_type not given using the default base class")
            logging.warn("walker_task_type not given using the default base class")
        else:
            self._walker_task_type = walker_task_type

        # initialize a list to put results in
        self.results = None

        # this is meant to be a transient variable, will be initialized and deinitialized
        self._walker_processes = None

    def init(self, **kwargs):

        super().init(**kwargs)

        # now that we have started the processes register the handler
        # for SIGTERM signals that will clean up our children cleanly
        signal.signal(signal.SIGTERM, self._sigterm_shutdown)

    def _sigterm_shutdown(self, signum, frame):

        logging.critical("Received external SIGTERM, forcing shutdown.")

        self.force_shutdown()

        logging.critical("Shutdown complete.")



    @property
    def walker_task_type(self):
        """The callable that generates a worker object.

        Typically this is just the type from the class definition of
        the Worker where the constructor is called.

        """
        return self._walker_task_type


    def force_shutdown(self):

        # send sigterm signals to processes to kill them
        for walker_idx, walker_process in enumerate(self._walker_processes):

            logging.critical("Sending SIGTERM message on {} to worker {}".format(
                self._irq_parent_conns[walker_idx].fileno(), walker_idx))

            # send a kill message to the worker
            self._irq_parent_conns[walker_idx].send(signal.SIGTERM)

        logging.critical("All kill messages sent to workers")


        # wait for the walkers to finish and handle errors in them
        # appropriately
        alive_walkers = [walker.is_alive() for walker in self._walker_processes]
        walker_exitcodes = {}
        premature_exit = False
        while any(alive_walkers):

            for walker_idx, walker in enumerate(self._walker_processes):

                if not alive_walkers[walker_idx]:
                    continue

                if walker.is_alive():
                    pass

                # otherwise the walker is done
                else:
                    alive_walkers[walker_idx] = False
                    walker_exitcodes[walker_idx] = walker.exitcode



    def map(self, *args, **kwargs):

        # run computations in a Manager context
        with self._mp_ctx.Manager() as manager:

            num_walkers = len(args[0])

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
            for walker in range(num_walkers):
                results.append(None)


            # use pipes for communication channels between this parent
            # process and the children for sending specific interrupts
            # such as the signal to kill them. Note that the clean way to
            # end the process is to send poison pills on the task queue,
            # this is for other stuff. IRQ is a common abbreviation for
            # interrupts
            self._irq_parent_conns = []

            # unpack the generator for the kwargs
            kwargs = {key : list(kwarg) for key, kwarg in kwargs.items()}

            # create the task based processes
            self._walker_processes = []
            for walker_idx, task_args in enumerate(zip(*args)):

                task_kwargs = {key : value[walker_idx] for key, value in kwargs.items()}

                # make the interrupt pipe
                parent_conn, child_conn = self._mp_ctx.Pipe()
                self._irq_parent_conns.append(parent_conn)

                # start a process for this walker
                walker_process = self.walker_task_type(walker_idx,
                                                       self._attributes,
                                                       self._func,
                                                       task_args,
                                                       task_kwargs,
                                                       worker_queue,
                                                       results,
                                                       worker_segment_times,
                                                       child_conn,
                )

                walker_process.start()

                self._walker_processes.append(walker_process)

            new_walkers = [None for _ in range(num_walkers)]
            results_found = [False for _ in range(num_walkers)]
            while not all(results_found):

                # go through the results list and handle the values that may be there
                for walker_idx, result in enumerate(results):

                    if results_found[walker_idx]:
                        continue

                    # logging.info("Checking for walker {}".format(walker_idx))

                    # first check to see if any of the task processes were
                    # terminated from the system
                    if self._irq_parent_conns[walker_idx].poll():

                        irq = self._irq_parent_conns[walker_idx].recv()

                        if issubclass(type(irq), TaskProcessKilledError):

                            # just terminate if a worker goes down. We
                            # could handle this better but it is not implemented now
                            logging.critical(
                                "Process {} was killed by sigterm, shutting down.".format(
                                walker_process[walker_idx].name))

                            logging.info(
                                "Recovery is possible here, but is not implemented "
                                "so we opt to fail fast and let you know a problem exists."
                                "Please use checkpointing to avoid lost data.")

                            self.force_shutdown()
                            logging.critical("Shutdown complete.")

                        logging.debug("Received {} acknowledgement from {}".format(ack,
                                                                                   worker.name))

                    # if no interrupts were handled we continue

                    # if it is None no response has been made at all
                    # yet, this is the initialized value
                    if result is None:
                        pass


                    # walker results are returned serialized as
                    # pickles, they are packed into a tuple so that we
                    # can associate them with an explicit marker, if
                    # we have a tuple then we can handle that
                    # appropriately
                    elif type(result) == tuple:

                        logging.debug("Received a results tuple")

                        assert len(result) == 2, "Result tuples should be only be (ID, pickle)"

                        result_id, payload = result

                        # there was a walker successfully returned
                        if result_id == 'Walker':

                            logging.debug("Received a serialized results walker")

                            # deserialize
                            logging.debug("deserializing")
                            new_walker = pickle.loads(payload)

                            logging.info("Got result for walker {}".format(walker_idx))

                            new_walkers[walker_idx] = new_walker
                            results_found[walker_idx] = True

                        else:
                            raise ValueError("Unkown result ID: {}".format(result_id))

                    elif issubclass(type(result), TaskException):

                        logging.critical(
                            "Exception encountered in a task which is unrecoverable."
                            "You will need to reconfigure your components in a stable manner.")

                        self.force_shutdown()

                        logging.critical("Shutdown complete.")
                        raise result

                    elif issubclass(type(result), TaskProcessException):

                        # we make just an error message to say that errors
                        # in the worker may be due to the network or
                        # something and could recover
                        logging.error("Exception encountered in the work mapper task process."
                                      "Recovery possible, see further messages.")

                        # However, the current implementation doesn't
                        # support retries or whatever so we issue a
                        # critical log informing that it has been elevated
                        # to critical and will force shutdown
                        logging.critical(
                            "Task process error mode resiliency not supported at this time."
                            "Performing force shutdown and simulation ending.")

                        self.force_shutdown()

                        logging.critical("Shutdown complete.")
                        raise result

                    elif issubclass(type(result), Exception):
                        logging.critical("Unknown exception {} encountered.".format(result))

                        self.force_shutdown()

                        logging.critical("Shutdown complete.")

                        raise result

                    else:
                        logging.critical("Unknown result value {} encountered.".format(result))

                        self.force_shutdown()

                        logging.critical("Shutdown complete.")


            # save the managed list of the recorded worker times locally
            for key, val in worker_segment_times.items():
                self._worker_segment_times[key] = val

            # wait for the processes to end
            # for walker in self._walker_processes:
            #     walker.join()
            #     logging.info("Joined {}".format(walker.name))

        # deinitialize the current walker processes
        self._walker_processes = None

        return new_walkers


class WalkerTaskProcess(mp.Process):

    NAME_TEMPLATE = "Walker-{}"

    def __init__(self,
                 walker_idx,
                 mapper_attributes,
                 func,
                 task_args,
                 task_kwargs,
                 worker_queue,
                 results_list,
                 worker_segment_times,
                 interrupt_connection,
                 **kwargs
                 ):

        # initialize the process customizing the name
        mp.Process.__init__(self, name=self.NAME_TEMPLATE.format(walker_idx), **kwargs)

        # the idea with this TaskProcess thing is that we pass in all
        # the data to the constructor to create a "thunk" (a closure
        # that is ready to be run without arguments) and then when run
        # is called there will be no arguments to be passed. This
        # simplifies the flow of data and underscores that the task is
        # the process.

        # task arguments
        self._func = func
        self._task_args = task_args
        self._task_kwargs = task_kwargs


        self.walker_idx = walker_idx
        self._worker_idx = None
        self.mapper_attributes = mapper_attributes

        # set the managed datastructure proxies as an attribute so we
        self._worker_queue = worker_queue
        self._results_list = results_list
        self._worker_segment_times = worker_segment_times
        self._irq_channel = interrupt_connection

        # also register the SIGTERM signal handler for graceful
        # shutdown with reporting to mapper
        signal.signal(signal.SIGTERM, self._external_sigterm_shutdown)


    def _external_sigterm_shutdown(self, signum, frame):

        logging.debug("Received external SIGTERM kill command.")

        logging.debug("Alerting mapper that this will be honored.")

        # send an error to the mapper that the worker has been killed
        self._irq_channel.send(TaskProcessKilledError(
            "{} (pid: {}) killed by external SIGTERM signal".format(self.name, self.pid)))

        logging.debug("Acknowledgment sent")

        logging.debug("Shutting down process")


    def _shutdown(self):
        """The normal shutdown which can be ordered by the work mapper."""

        logging.debug("Received SIGTERM kill command from mapper")

        logging.debug("Acknowledging kill request will be honored")

        # report back that we are shutting down with a True
        self._irq_channel.send(True)

        logging.debug("Acknowledgment sent")

        logging.debug("Shutting down process")



    @property
    def attributes(self, key):
        return self._attributes

    @attributes.getter
    def attributes(self, key):
        return self._attributes[key]

    def _run_task(self, task):

        # run the task thunk
        logging.info("{}: Running task".format(self.name))
        try:
            result = self.run_task(task)
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

            logging.critical("{}: ".format(self.name) + msg + '\n' + traceback_log_msg)

            # raise a TaskException to distinguish it from the worker
            # errors with the metadata about the original exception

            raise TaskException("Error occured during task execution, recovery not possible.",
                            wrapped_exception=task_exception,
                            tb=tb)

        return result

    def run_task(self, task):

        logging.info("Running an unspecialized task")

        return task()

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
            self._run_walker()

        except TaskException as task_exception:

            logging.error("{}: TaskException caught".format(self.name))

            run_exception = task_exception

        # anything else is considered a WorkerException so take the
        # original exception and generate a worker exception from that
        except Exception as exception:

            logging.debug("{}: TaskProcessException Error caught".format(self.name))


            # get the traceback
            tb = sys.exc_info()[2]

            msg = "Exception '{}({})' caught in a task process.".format(
                                   type(exception).__name__, exception)
            traceback_log_msg = \
                """Traceback:
--------------------------------------------------------------------------------
{}
--------------------------------------------------------------------------------
                """.format(''.join(traceback.format_exception(
                    type(exception), exception, tb)),
                )

            logging.error(msg + '\n' + traceback_log_msg)

            # raise a TaskError to distinguish it from the worker
            # errors with the metadata about the original exception

            walker_exception = TaskProcessException(
                "Error occured during task process execution.",
                wrapped_exception=exception,
                tb=tb)

            run_exception = walker_exception

        # raise worker_exception
        if run_exception is not None:

            logging.debug("{}: Putting exception in managed results list".format(self.name))

            # then put the exception and the traceback onto the queue
            # so we can communicate back to the parent process
            try:
                self._results_list[self.walker_idx] = run_exception
            except BrokenPipeError as exc:
                logging.error(
                    "{}: Pipe is broken indicating the root process has already exited:\n{}".format(
                        self.name, exc))

    def _run_walker(self):

        logging.info("Walker process started as name: {}; PID: {}".format(self.name,
                                                                          self.pid))

        # lock a worker, then pop it off the queue so no other process
        # tries to use it
        worker_received = False
        while not worker_received:

            # pop off a worker to use it, this will block until it
            # receives a worker
            try:
                worker_idx = self._worker_queue.get_nowait()
            except pyq.Empty:
                pass

            # always do on a successful get
            else:

                if type(worker_idx) == int:
                    worker_received = True
                    logging.info("{}: acquired worker {}".format(self.name,
                                                                 worker_idx))

                # if it is a shutdown signal we do so
                elif worker_idx is signal.SIGTERM:
                    logging.info(
                        "{}: SIGTERM signal received from mapper. Shutting down.".format(
                        self.name))

                    self._shutdown()

                    return None


            # check to see if there is any signals on the interrupt channel
            if self._irq_channel.poll():

                # get the message
                message = self._irq_channel.recv()

                logging.debug("{}: Received message from mapper on filehandle {}: {}".format(
                    self.name, self._irq_channel.fileno(), message))

                # handle the message


                # check for signals to die
                if message is signal.SIGTERM:

                    logging.critical(
                        f"{self.name}: SIGTERM signal received from mapper. Shutting down."
                    )

                    self._shutdown()

                    return None

                else:
                    logging.error("{}: Message not recognized, continuing operations and"
                                  " sending error to mapper".format(self.name))
                    self._irq_channel.send(
                        ValueError(
                            "Message: {} not recognized continuing operations".format(
                                message)))

        # after the wait loop we can now perform our work

        self._worker_idx = worker_idx

        # generate the task thunk
        task = Task(self._func, *self._task_args, **self._task_kwargs)

        # run the task
        start = time.time()
        logging.info("{}: running function. Time: {}".format(self.name,
                                                      time.time()))

        # run the task doing the proper handling of the task
        # exception, this can raise a task exception
        result = self._run_task(task)

        logging.info("{}: finished running function. Time {}".format(self.name,
                                                                     time.time()))
        end = time.time()

        # we separately set the result to the results list. This is so
        # we can debug performance problems, and dissect what is due
        # to computation time and what is due to communication
        logging.info("{}: Setting value to results list. Time {}".format(self.name,
                                                                     time.time()))

        logging.debug("Serializing the result walker")

        serial_result = pickle.dumps(result)
        logging.debug("Putting tagged serialized walker tuple on managed results list")
        self._results_list[self.walker_idx] = ('Walker', serial_result)

        logging.info("{}: Finished setting value to results list. Time {}".format(self.name,
                                                                                  time.time()))

        # put the worker back onto the queue since we are done using it
        self._worker_queue.put(worker_idx)

        logging.info("{}: released worker {}".format(self.name,
                                                     worker_idx))

        # add the time for this segment to the collection of the worker times
        segment_time = end - start
        seg_times = self._worker_segment_times[worker_idx] + [segment_time]

        # we must explicitly set the new value in total to trigger an
        # update of the real dictionary. In place modification of
        # proxy objects has no effect
        self._worker_segment_times[worker_idx] = seg_times

        logging.info("{}: Exiting normally having completed the task".format(self.name))
