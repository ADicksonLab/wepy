
import trio

from wepy.work_mapper.mapper import (
    ABCMapper,
    Worker,
    Task,
)


class TrioThreadWorker():

    NAME_TEMPLATE = "TrioThreadWorker-{}"

    def __init__(self,
                 worker_idx,
                 task_recv_chan,
                 result_send_chan,
                 **kwargs
    ):

        self._worker_idx = worker_idx
        self._name = self.NAME_TEMPLATE.format(self.worker_idx)

        self._attributes = kwargs

        self._task_recv_chan = task_recv_chan
        self._result_send_chan = result_send_chan

    @property
    def worker_idx(self):
        """Dictionary of attributes of the worker."""
        return self._worker_idx

    @property
    def attributes(self):
        """Dictionary of attributes of the worker."""
        return self._attributes


    def run_task(self, task):

        task()

class OpenMMGPUTrioThreadWorker(TrioThreadWorker):

    NAME_TEMPLATE = "OpenMMGPUTrioThreadWorker-{}"


    def run_task(self, task):

        # documented in superclass

        device_id = self.mapper_attributes['device_ids'][self._worker_idx]

        # make the platform kwargs dictionary
        platform_options = {'DeviceIndex' : str(device_id)}

        # run the task and pass in the DeviceIndex for OpenMM to
        # assign work to the correct GPU
        return task(platform_kwargs=platform_options)

class TrioMapper(ABCMapper):

    def __init__(self,
                 num_workers=None,
                 worker_type=None,
                 worker_attributes=None,
                 segment_func=None,
                 **kwargs):

        super().__init__(
            segment_func=segment_func,
            **kwargs
        )

        self._num_workers = num_workers
        self._worker_segment_times = None

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


    def init(self,
             num_workers=None,
             segment_func=None,
    ):

        super().init(segment_func=segment_func)

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


        # the task channel
        task_send_chan, task_recv_chan = trio.open_memory_channel(0)

        result_send_chan, result_recv_chan = trio.open_memory_channel(0)

        


    def cleanup(self, **kwargs):

        pass

    def map(self, *args, **kwargs):

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



        # make the worker threads
