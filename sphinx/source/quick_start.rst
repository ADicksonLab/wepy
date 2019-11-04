Quick Start
===========

A trivial example
-----------------

Here is one of the simplest possible examples of running wepy using
OpenMM.

This requires that the openmmtools library is installed to get a
pre-built MD system:

.. code:: bash

    conda install -c conda-forge openmmtools

This example uses the ``NoResampler`` which performs no resampling so
this is equivalent to running 10 simulations in parallel.

.. code:: python

    from copy import copy

    import simtk.openmm as omm
    import simtk.unit as unit

    from openmmtools.testsystems import LennardJonesPair

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

In this example we see the core components of a wepy simulation:

-  **Runner**: for running dynamics ('sampling' in wepy parlance)
-  **Resampler**: for performing resampling (i.e. cloning and merging of
   walkers)
-  **Manager**: the main simulation loop

Being the trivial example it is, not only does it do no resampling it
produces no output other than the final walkers and then only in memory
as ``Walker`` objects.

Further in the docs we will show how to add reporters to the simulation
so that your results can be saved and how to use and customize
resamplers that do useful work.

Writing scripts like this is the primary way in which wepy is intended
to be used.

You can run this wepy simulation by running this on the command line
after you have copy and pasted it to a file:

.. code:: bash

    python run_wepy.py

The ``wepy`` command line application introduces some useful tools for
working with and managing many interconnected simulations with
checkpointing capabilities. This is the ``orchestration`` sub-module and
should be a considered an advanced feature. Just know that if you are
running a lot of simulations, long simulations which tend to fail due to
hardware issues, or if you need to repeatedly stop and restart
simulations the orchestration sub-module is available for that.

So ignore the wepy commands like ``wepy run`` for now.

A somewhat more realistic example
---------------------------------
