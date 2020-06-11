from openmm_systems.test_systems import (
    LennardJonesPair,
    LysozymeImplicit,
)
import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
from wepy.runners.openmm import gen_sim_state

import time
import functools

import trio

def mysleep(seconds):

    time.sleep(seconds)

    return f"Finished my task sir of sleeping for {seconds}"

def create_sim(device_id):

    test_sys = LysozymeImplicit()

    integrator = omm.LangevinIntegrator(300.0*unit.kelvin,
                                        1/unit.picosecond,
                                        0.002*unit.picoseconds)

    init_state = gen_sim_state(test_sys.positions, test_sys.system, integrator)

    platform = omm.Platform.getPlatformByName('OpenCL')
    if device_id is not None:
        platform.setPropertyDefaultValue('DeviceIndex', device_id)

    simulation = omma.Simulation(
        test_sys.topology,
        test_sys.system,
        integrator,
        platform=platform,
    )

    simulation.context.setState(init_state)

    return simulation


def run_sim(device_id=None):

    start = time.time()

    steps = 10000
    sim = create_sim(device_id)

    print("Starting Simulation")

    # sim.step(steps)

    # SNIPPET: using the raw integrator which is uninterruptible

    sim.integrator.step(steps)

    # SNIPPET: getting our own interruptible thing
    # for step in range(steps):
    #     sim.integrator.step(1)

    end = time.time()

    print(f"Simulation took {end - start}")

    return sim


class Task():
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


async def worker(
        task_recv_chan,
        result_send_chan,
        worker_idx,
        worker_kwargs,
):

    async with task_recv_chan, result_send_chan:

        print(f"Worker {worker_idx} created with kwargs: {worker_kwargs}")

        async for task_idx, task in task_recv_chan:

            # partially apply the worker kwargs
            task_part = functools.partial(task, **worker_kwargs)

            print(f"Running task {task_idx} in worker {worker_idx}")

            result = await trio.to_thread.run_sync(task_part)

            print(f"Sending result for task {task_idx} from worker {worker_idx}")

            await result_send_chan.send((task_idx, result))

            print(f"Sent result for task {task_idx} from worker {worker_idx}")



async def main():

    num_sims = 6
    num_workers = 2

    print(f"{num_sims} tasks on {num_workers} workers")

    async with trio.open_nursery() as nrs:

        task_send_chan, task_recv_chan = trio.open_memory_channel(num_sims)
        result_send_chan, result_recv_chan = trio.open_memory_channel(num_sims)

        async with task_send_chan, task_recv_chan, result_send_chan, result_recv_chan:

            # start the worker threads
            for worker_idx in range(num_workers):

                print(f"Starting worker {worker_idx}")

                # nrs.start_soon(worker,
                #                task_recv_chan.clone(),
                #                result_send_chan.clone(),
                #                worker_idx,
                # )

                nrs.start_soon(worker,
                               task_recv_chan.clone(),
                               result_send_chan.clone(),
                               worker_idx,
                               {"device_id" : str(worker_idx)},
                )


            results = [None for _ in range(num_sims)]

            # then send the tasks to the workers
            for sim_idx in range(num_sims):

                # task = Task(mysleep, sim_idx + 5)
                task = Task(run_sim)

                print(f"Sending task {sim_idx}")
                await task_send_chan.send((sim_idx, task))

            print("Finished submitting tasks to workers")

            # fetching the results

            # here we just wait for them in order, but we could scan
            # each of them to get which ones arrive first etc.
            for result_idx in range(num_sims):

                print(f"Waiting results of task {result_idx}")

                task_idx, task_result = await result_recv_chan.receive()
                print(f"Received results for task {task_idx}")

                results[task_idx] = task_result

print("Starting things")
start = time.time()
trio.run(main)
end = time.time()

print(f"Took {end - start} seconds")
