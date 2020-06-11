from openmm_systems.test_systems import (
    LennardJonesPair,
    LysozymeImplicit,
)
import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
from wepy.runners.openmm import gen_sim_state

import time


def create_sim():

    test_sys = LysozymeImplicit()

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

def run_sim(sim, steps):

    sim.integrator.step(steps)

    return sim

def main():

    num_sims = 2
    steps = 5000

    simulations = []
    for idx in range(num_sims):

        simulations.append(create_sim())

    for i, sim in enumerate(simulations):

        start = time.time()
        run_sim(sim, steps)
        end = time.time()

        print(f"Sim {i} took: {end - start}")

start = time.time()
main()
end = time.time()

print(f"Took {end - start} seconds")
