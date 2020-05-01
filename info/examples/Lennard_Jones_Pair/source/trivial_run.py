"""Very basic example of a simulation without a resampler or boundary
conditions"""

from copy import copy

import simtk.openmm as omm
import simtk.unit as unit

from openmm_systems.test_systems import LennardJonesPair

from wepy.resampling.resamplers.resampler import NoResampler
from wepy.runners.openmm import OpenMMRunner, gen_walker_state
from wepy.walker import Walker
from wepy.sim_manager import Manager

# use a ready made system for OpenMM MD simulation
test_sys = LennardJonesPair()

integrator = omm.LangevinIntegrator(300.0*unit.kelvin,
                                    1/unit.picosecond,
                                    0.002*unit.picoseconds)

init_state = gen_walker_state(test_sys.positions, test_sys.system, integrator)

runner = OpenMMRunner(test_sys.system, test_sys.topology, integrator,
                      platform='Reference')

# a trivial resampler which does nothing
resampler = NoResampler()

# Run the simulation

# number of cycles of WE to perform
n_cycles = 5

# the number of MD dynamics steps for each cycle
n_steps = 1000
steps = [n_steps for i in range(n_cycles)]

# number of parallel simulations
n_walkers = 10



# create the initial walkers with equal weights
init_weight = 1.0 / n_walkers
init_walkers = [Walker(copy(init_state), init_weight) for i in range(n_walkers)]

sim_manager = Manager(init_walkers,
                      runner=runner,
                      resampler=resampler)

# run the simulation and get the results
final_walkers, _ = sim_manager.run_simulation(n_cycles, steps)
