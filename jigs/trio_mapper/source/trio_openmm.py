from openmm_systems.test_systems import LennardJonesPair
import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
from wepy.runners.openmm import gen_sim_state

import time

import trio


def create_sim():

    test_sys = LennardJonesPair()

    integrator = omm.LangevinIntegrator(300.0*unit.kelvin,
                                        1/unit.picosecond,
                                        0.002*unit.picoseconds)

    init_state = gen_sim_state(test_sys.positions, test_sys.system, integrator)

    platform = omm.Platform.getPlatformByName('CPU')

    simulation = omma.Simulation(
        test_sys.topology,
        test_sys.system,
        integrator,
        platform=platform,
    )

    simulation.context.setState(init_state)

    return simulation

async def run_sim(sim, steps):

    start = time.time()

    await sim.step(steps)
    # for step in range(steps):
    #     sim.step(1)

    end = time.time()

    print(f"Simulation took {end - start}")

    return sim

async def main():

    num_sims = 3

    simulations = []
    for idx in range(num_sims):

        simulations.append(create_sim())

    async with trio.open_nursery() as nrs:
        for i, sim in enumerate(simulations):

            print(f"starting sim {i}")
            nrs.start_soon(run_sim, sim, 10000)

start = time.time()
trio.run(main)
end = time.time()

print(f"Took {end - start} seconds")
