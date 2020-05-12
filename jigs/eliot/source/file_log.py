"""Very basic example of a simulation without a resampler or boundary
conditions"""

from copy import copy
import sys

## Logging

import sys

from eliot import start_action, to_file
to_file(open("_output/file_log.eliot.json", 'ab'))

### Setup Junk
with start_action(action_type="Setup") as setup_cx:

    ## Application Imports
    with start_action(action_type="imports"):

        import simtk.openmm as omm
        import simtk.unit as unit

        from openmm_systems.test_systems import LennardJonesPair

        from wepy.resampling.resamplers.resampler import NoResampler
        from wepy.runners.openmm import OpenMMRunner, gen_walker_state
        from wepy.walker import Walker
        from wepy.sim_manager import Manager

        from wepy.work_mapper.mapper import Mapper
        # from wepy.work_mapper.thread import ThreadMapper


    # use a ready made system for OpenMM MD simulation
    with start_action(action_type="Instantiate test system"):
        test_sys = LennardJonesPair()

    with start_action(action_type="Gen Runner"):
        integrator = omm.LangevinIntegrator(300.0*unit.kelvin,
                                            1/unit.picosecond,
                                            0.002*unit.picoseconds)

        init_state = gen_walker_state(test_sys.positions, test_sys.system, integrator)

        runner = OpenMMRunner(test_sys.system, test_sys.topology, integrator,
                              platform='Reference')

    # a trivial resampler which does nothing
    with start_action(action_type="Instantiate Resampler"):
        resampler = NoResampler()

    # Run the simulation

    # number of cycles of WE to perform
    n_cycles = 1

    # the number of MD dynamics steps for each cycle
    n_steps = 1000000
    steps = [n_steps for i in range(n_cycles)]

    # number of parallel simulations
    n_walkers = 10

    # the work mapper
    # work_mapper = ThreadMapper()

    work_mapper = Mapper()

    # create the initial walkers with equal weights
    with start_action(action_type="Init Walkers") as ctx:
         init_weight = 1.0 / n_walkers
         init_walkers = [Walker(copy(init_state), init_weight) for i in range(n_walkers)]

    with start_action(action_type="Init Sim Manager") as ctx:
         sim_manager = Manager(
             init_walkers,
             runner=runner,
             resampler=resampler,
             work_mapper=work_mapper)

# run the simulation and get the results
with start_action(action_type="Simulation") as ctx:
    final_walkers, _ = sim_manager.run_simulation(n_cycles, steps)

