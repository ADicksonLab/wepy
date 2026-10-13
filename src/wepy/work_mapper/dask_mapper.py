"""
Dask is a Python library for parallel computing. 
It splits work into many tasks and runs them at the same time across multiple processes or machines. 

DaskMapper lets Wepy's Manager use a Dask cluster without knowing Dask exists. 
The Manager only ever calls mapper.init(), mapper.map(...) and mapper.cleanup(). 
DaskMapper implements those three methods by sending work to the cluster.

dask.distributed package runs three kinds of process:
    Scheduler: one process that keeps track of every task and every worker, 
               and decides which worker runs which task.
    Workers: long-lived processes that run the tasks. Here each worker drives one GPU. 
             A worker stays alive between tasks, so anything stored on it survives from one cycle to the next.
    Clients: the programs that submit work. Here the client lives inside DaskMapper, 
             in the process running the wepy sim manager.

Usage
-----
    with slurm_dask_cluster(rendezvous_dir) as (address, n_workers):
        mapper = DaskMapper(runner_factory=factory, address=address,
                            num_workers=n_workers)
        mapper.init()
        manager = Manager(init_walkers, runner=factory(), work_mapper=mapper,
                          resampler=resampler, reporters=reporters)
        manager.run_simulation(n_cycles, segment_lengths)
"""

import time
import logging

from dask.distributed import Client, WorkerPlugin, get_worker

logger = logging.getLogger(__name__)

GPU_PLATFORMS = ("CUDA", "OpenCL", "HIP")

class WepyRunnerPlugin(WorkerPlugin):
    """Builds the OpenMM runner once and attaches it to the worker as worker.wepy_runner. 

    A Dask WorkerPlugin is an object whose setup method Dask runs on every
    worker: on each worker already connected when the plugin is registered,
    and on any worker that joins later. DaskMapper.init registers this
    plugin, so each worker gets its own runner before the first cycle.
    """

    name = "wepy-runner"

    def __init__(self, runner_factory, device_index="0"):
        """Store the settings.

        Parameters
        ----------
        runner_factory : callable
            Called with no arguments on each worker to build its runner.
        device_index : str or None, default "0"
            GPU index the worker's runner should use.
        """
        self.runner_factory = runner_factory
        self.device_index = device_index


    def setup(self, worker):
        """Build current worker's runner and store it on the worker.

        Called by Dask, afterwards the worker has two
        attributes that _run_segment reads on every task:

        - worker.wepy_runner: the runner returned by runner_factory().
        - worker.wepy_device_index: the GPU index as a string.

        Parameters
        ----------
        worker : distributed.Worker
            The worker being set up, passed in by Dask.
        """
        logger.info("building wepy runner on %s", worker.address)
        worker.wepy_runner = self.runner_factory()
        if self.device_index is None:
            worker.wepy_device_index = str(worker.name)
        else:
            worker.wepy_device_index = self.device_index


    def teardown(self, worker):
        """Release the worker's runner when the worker shuts down.

        Called by Dask when the worker closes.
        Calls the runner's cleanup method, if it has one, which for
        DaskOpenMMRunner drops the Context and frees its GPU memory.

        Parameters
        ----------
        worker : distributed.Worker
            The worker being shut down, passed in by Dask.
        """
        runner = getattr(worker, "wepy_runner", None)
        if runner is not None and hasattr(runner, "cleanup"):
            runner.cleanup()
        worker.wepy_runner = None


def _worker_ready(dask_worker):
    return getattr(dask_worker, "wepy_runner", None) is not None


def _run_segment(walker, segment_length, call_kwargs):
    """The task body. It looks up the worker's runner, picks the platform and device, 
    runs one segment, and returns the new walker, the elapsed seconds and the worker name.

    Parameters
    ----------
    walker : object implementing the Walker interface
        The walker for which dynamics will be propagated.
    segment_length : int or float
        The numerical value that specifies how much dynamics are to be run.

    Returns
    -------
    new_walker : object implementing the Walker interface
        Walker after dynamics was run, only the state should be modified.
    elapsed_time: float
        Time it took for a runner to run one segment.
    worker_name: str
        Name of the worker.
    """
    worker = get_worker()
    worker_name = worker.name
    runner = worker.wepy_runner

    platform = runner.platform_name
    platform_kwargs = dict(runner.platform_kwargs or {})
    if platform in GPU_PLATFORMS:
        platform_kwargs.setdefault("DeviceIndex", worker.wepy_device_index)

    start = time.time()
    new_walker = runner.run_segment(walker, segment_length, platform=platform,
                                    platform_kwargs=platform_kwargs, **call_kwargs)
    elapsed_time = time.time() - start
    return new_walker, elapsed_time, worker_name


class DaskMapper:
    """The driver-side object. It connects to the scheduler,
      initializes the runner on every worker, and turns one cycle into one batch of Dask tasks.
    """

    def __init__(self, runner_factory, scheduler_file=None, address=None,
                 num_workers=None, retries=2, worker_timeout=600,
                 resources=None, device_index="0"):
        """Constructor for DaskMapper

        Parameters
        ----------
        resources : dict, optional
            Dask resource request per task, e.g. {"GPU": 1}.
        device_index : str or None
            "0" (default) when every worker sees exactly one GPU
            (srun --gpus-per-task=1). None when all workers share a node
            and see every GPU (LocalCluster): worker i then uses GPU i, so
            n_workers must not exceed the number of visible GPUs.
        """
        if scheduler_file is None and address is None:
            raise ValueError("need one of scheduler_file or address")

        self.runner_factory = runner_factory
        self.scheduler_file = scheduler_file
        self.address = address
        self.num_workers = num_workers
        self.retries = retries
        self.worker_timeout = worker_timeout
        self.resources = resources
        self.device_index = device_index

        self.client = None
        self._worker_ids = {}
        self._worker_segment_times = {}


    @property
    def worker_segment_times(self):
        """{worker_id: [segment seconds, ...]} for the last cycle. The sim
        manager reads this for its timing reports."""
        return self._worker_segment_times


    def init(self, num_workers=None, segment_func=None, **kwargs):
        """Called by the sim manager before cycle 0.

        segment_func is accepted for interface compatibility and ignored: the
        runner lives on the workers.

        Parameters
        ----------
        num_workers: int
            Number of workers assined to the job.
        """
        if self.client is not None:
            logger.info("DaskMapper already initialized; reusing client and warm runners")
            return

        num_workers = num_workers or self.num_workers

        self.client = Client(address=self.address, scheduler_file=self.scheduler_file)

        if num_workers:
            logger.info("waiting for %d workers", num_workers)
            self.client.wait_for_workers(n_workers=num_workers,
                                         timeout=self.worker_timeout)

        plugin = WepyRunnerPlugin(self.runner_factory, device_index=self.device_index)
        register = getattr(self.client, "register_plugin", None) \
            or self.client.register_worker_plugin
        register(plugin)

        ready = self.client.run(_worker_ready)
        broken = [addr for addr, ok in ready.items() if not ok]
        if broken:
            raise RuntimeError("runner setup failed on workers {}; see their logs"
                               .format(broken))

        logger.info("mapper ready on %d workers: %s", len(ready), self.client.dashboard_link)


    def cleanup(self, **kwargs):
        """ Closes up and clears the client."""
        if self.client is not None:
            self.client.close()
            self.client = None


    def map(self, *args, **kwargs):
        """One cycle.

        args and kwargs are both per-walker sequences (generators, in the
        sim manager's case).
        """
        args = [list(arg) for arg in args]
        kwargs = {key: list(value) for key, value in kwargs.items()}
        n_calls = len(args[0])
        per_call_kwargs = [{key: value[i] for key, value in kwargs.items()}
                           for i in range(n_calls)]

        options = {}
        if self.resources:
            options["resources"] = self.resources

        futures = self.client.map(_run_segment, *args, per_call_kwargs,
                                  pure=False, retries=self.retries, **options)
        results = self.client.gather(futures)

        walkers = []
        self._worker_segment_times = {}
        for new_walker, seconds, worker_name in results:
            walkers.append(new_walker)
            worker_id = self._worker_ids.setdefault(worker_name, len(self._worker_ids))
            self._worker_segment_times.setdefault(worker_id, []).append(seconds)
        return walkers