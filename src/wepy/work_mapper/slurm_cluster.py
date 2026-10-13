"""
Turn the tasks of a Slurm job step into a Dask cluster.

When a job runs ``srun python script.py``, Slurm starts one copy of the
script per task. This module makes those copies
find each other and form one Dask cluster:

- Task 0 (the "boss") starts the Dask scheduler plus one worker for its own
  GPU, then runs the rest of the script (the simulation).
- Every other task (a "helper") becomes a single Dask worker and does nothing
  else.

Task 0 writes the scheduler's address to JSON file, in a directory every node can see. 
Helpers wait for that file, read the address, and connect.
"""

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

from distributed import Client, LocalCluster, Worker
from distributed.utils import get_ip, get_ip_interface

logger = logging.getLogger(__name__)


def slurm_rank():
    """Return this task's position in the current srun step.
 
    Returns
    -------
    rank : int
        SLURM_PROCID, the index of this task within the step (0-based).
    ntasks : int
        SLURM_NTASKS, the number of tasks in the step.
 
    Outside SLURM both variables are missing and this returns (0, 1), so a
    script run directly with ``python`` behaves as a one-task job: task 0
    with a single local worker.
    """
    return (int(os.environ.get("SLURM_PROCID", 0)),
            int(os.environ.get("SLURM_NTASKS", 1)))


def _rendezvous_path(rendezvous_dir):
    """Path of the JSON file through which task 0 publishes the scheduler address.
    
    Parameters
    ----------
    rendezvous_dir : str
        Directory on a filesystem shared by all nodes in the job.

    Returns
    -------
    str
        Path to the JSON file that contains teh schedular address.
    """
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    step = os.environ.get("SLURM_STEP_ID", "0")
    return os.path.join(rendezvous_dir, "dask-{}.{}.json".format(job, step))


def _host_ip(interface):
    """IP address this node should bind Dask to.
 
    Parameters
    ----------
    interface : str or None
        Network interface name, e.g. "ib0". If None, the address of the
        interface holding the default route is used.
 
    Returns
    -------
    str
        An IPv4 address that processes on other nodes can connect to.
    """
    return get_ip_interface(interface) if interface else get_ip()


def _scratch(rank):
    """Node-local directory for a worker's put files and other scratch data."""

    base = os.environ.get("TMPDIR") or "/tmp"
    return os.path.join(base, "dask-{}-{}".format(os.environ.get("SLURM_JOB_ID", "nojob"), rank))


def _wait_for_address(path, timeout):
    """Block until task 0 has published the scheduler address, then return it.
 
    A missing file, unparseable JSON or a
    missing "address" key all count as "not ready yet". 
 
    Parameters
    ----------
    path : str
        Rendezvous file written by task 0.
    timeout : float
        Seconds to keep polling before giving up.
 
    Returns
    -------
    str
        Scheduler address, e.g. "tcp://10.1.2.3:41235".
 
    Raises
    ------
    TimeoutError
        If no address appears within ``timeout`` seconds.
    """

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(path) as f:
                return json.load(f)["address"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            time.sleep(1)
    raise TimeoutError("no scheduler address at {} after {}s".format(path, timeout))


async def _serve(address, rank, host, death_timeout):
    """Run one Dask worker connected to ``address`` until it closes.
 
    The worker has a single thread, so it runs one segment at a time and
    owns one Context. It has no memory limit, so Dask never pauses or
    restarts it over memory use. Its name, "wepy-w<rank>", is what task 0
    uses at shutdown to tell remote workers apart from its own local one.
 
    The worker closes, and this coroutine returns, when the scheduler
    retires it (normal shutdown) or when the scheduler has been unreachable
    for ``death_timeout`` seconds (the driver on task 0 crashed or was killed).
 
    Parameters
    ----------
    address : str
        Scheduler address read from the rendezvous file.
    rank : int
        This task's SLURM_PROCID, used for the worker name and scratch path.
    host : str
        IP address to bind the worker to; see _host_ip.
    death_timeout : float
        Seconds to keep trying to reach a missing scheduler before closing.
    """
    async with Worker(
        address,
        nthreads=1,                  # one Context, one segment at a time
        name="wepy-w{}".format(rank),
        memory_limit=0,
        host=host,
        death_timeout=death_timeout, # exit if the scheduler vanishes (driver crashed)
        local_directory=_scratch(rank),
    ) as worker:
        await worker.finished()


def _run_worker_and_exit(path, rank, interface, timeout, death_timeout):
    """Entire life of a task with rank > 0: wait, serve, exit.
 
    Logs which GPU(s) SLURM gave this task, waits for the scheduler address,
    runs a worker until it closes, and then terminates the process with
    sys.exit(0).
    """

    logger.info("task %d on %s: CUDA_VISIBLE_DEVICES=%s", rank, socket.gethostname(),
                os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    address = _wait_for_address(path, timeout)
    asyncio.run(_serve(address, rank, _host_ip(interface), death_timeout))
    logger.info("task %d: scheduler closed, exiting", rank)
    sys.exit(0)


@contextmanager
def slurm_dask_cluster(rendezvous_dir, interface=None, timeout=600, death_timeout=120):
    """Turn the tasks of the current slurm run step into a Dask cluster.
 
    Every task in the step must enter this context manager. On task 0 it
    starts the scheduler and a local worker, publishes the scheduler address,
    and yields. On every other task it becomes a worker and exits the process
    when the cluster shuts down.
 
    Parameters
    ----------
    rendezvous_dir : str
        Directory on a filesystem shared by all nodes in the job.
    interface : str, optional
        Network interface for Dask traffic, e.g. "ib0". If None, uses
        $DASK_INTERFACE when set, otherwise the default-route interface.
    timeout : float, default 600
        Seconds a worker task waits for task 0 to publish the address.
    death_timeout : float, default 120
        Seconds a worker keeps trying to reach the scheduler after losing
        it (or before first reaching it) before it closes and its task exits.
 
    Yields
    ------
    address : str
        Scheduler address. Pass it to DaskMapper(address=...).
    n_workers : int
        Total number of workers that will join
 
    Raises
    ------
    TimeoutError
        On worker tasks only, if task 0 does not publish an address within
        ``timeout`` seconds.
 
    Notes
    -----
    Task 0's own worker is created by LocalCluster in a separate process
    that inherits task 0's environment, including CUDA_VISIBLE_DEVICES, so
    it gets task 0's GPU. LocalCluster names it 0 rather than "wepy-w0".
    """
    if interface is None:
        interface = os.environ.get("DASK_INTERFACE") or None
    rank, ntasks = slurm_rank()
    path = _rendezvous_path(rendezvous_dir)

    if rank != 0:
        _run_worker_and_exit(path, rank, interface, timeout, death_timeout)

    os.makedirs(rendezvous_dir, exist_ok=True)
    host = _host_ip(interface)
    logger.info("task 0 on %s (%s): CUDA_VISIBLE_DEVICES=%s", socket.gethostname(), host,
                os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))

    # Scheduler, plus one worker process for task 0's own GPU. Binding to the
    # node's real address (not the LocalCluster default of 127.0.0.1) lets
    # workers on other nodes connect.
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        memory_limit=0,
        host=host,
        scheduler_port=0,
        dashboard_address=":0",
        local_directory=_scratch(0),
    )
    address = cluster.scheduler_address

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"address": address, "ntasks": ntasks}, f)
    os.replace(tmp, path)  # atomic: readers never see a half-written file
    logger.info("scheduler at %s, dashboard %s", address, cluster.dashboard_link)

    try:
        yield address, ntasks
    finally:
        # Close the remote workers first so they exit cleanly instead of
        # logging connection errors, then let LocalCluster close its own
        # worker and the scheduler.
        try:
            with Client(address, timeout=30) as client:
                remote = [addr for addr, info in client.scheduler_info()["workers"].items()
                          if str(info.get("name", "")).startswith("wepy-w")]
                if remote:
                    client.retire_workers(workers=remote, close_workers=True)
        except Exception as err:
            logger.warning("closing remote workers: %s", err)
        try:
            cluster.close(timeout=30)
        except Exception as err:
            logger.warning("closing scheduler: %s", err)
        try:
            os.remove(path)
        except OSError:
            pass


def gpu_report(dask_worker=None):
    """Report which GPUs the calling worker process can see.
 
    Returns
    -------
    hostname : str
        Node the worker runs on.
    uuids : list of str
        UUIDs of the visible GPUs, e.g. ["GPU-3f2a..."]. Empty if
        nvidia-smi is missing, fails, or takes longer than 30 seconds.
    """
    cmd = ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        cmd += ["-i", visible]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        uuids = [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        uuids = []
    return socket.gethostname(), uuids


def check_one_gpu_per_worker(client):
    """Verify that every worker has exactly one GPU and that no GPU is shared.
 
    Call it after DaskMapper.init(), passing ``mapper.client``, to catch a
    wrong slurm run layout (for example, a missing --gpus-per-task=1, which gives
    every task all of the node's GPUs) before any simulation time is spent.
 
    Parameters
    ----------
    client : distributed.Client
        Client connected to the cluster to check.
 
    Returns
    -------
    dict
        {worker address: (hostname, [GPU UUIDs])}, as returned by gpu_report
        on each worker.
 
    Raises
    ------
    RuntimeError
        If any worker sees zero or more than one GPU (zero includes a worker
        where nvidia-smi failed while others succeeded), or if two workers
        see the same GPU.
    """
    report = client.run(gpu_report)
    if all(not uuids for _, uuids in report.values()):
        logger.warning("nvidia-smi unavailable; GPU placement not checked")
        return report
    seen = {}
    for addr, (host, uuids) in report.items():
        if len(uuids) != 1:
            raise RuntimeError("worker {} on {} sees {} GPUs: {}".format(addr, host, len(uuids), uuids))
        if uuids[0] in seen:
            raise RuntimeError("workers {} and {} share GPU {}".format(seen[uuids[0]], addr, uuids[0]))
        seen[uuids[0]] = addr
    logger.info("GPU placement ok: %s", {a: "{}:{}".format(h, u[0][:12]) for a, (h, u) in report.items()})
    return report