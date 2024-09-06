Setting Up the Simulation
=========================

If you just want to poke Wepy with a stick to see if the dang thing
works we have the script for you!

While this script uses a very simple system, it is a pretty good
introduction to the inputs and kind of results you can expect from Wepy
simulations.

Just make a new directory where you want your results to be placed
(anywhere on your filesystem) and then copy the commands below into a
python script saved in that directory.

.. code:: bash

   mkdir test_drive
   cd test_drive

This script can also be found in the Wepy repository in the
examples/test\ :sub:`drive` folder.

This code runs a simulation of a pair of particles that is created using
the ``test_system_builder`` module from Wepy. This will build a system
with one Na\ :sup:`+` and one Cl\ :sup:`-` atom. We will use 10 walkers,
and run 5 cycles of dynamics and resampling, where each round of
dynamics lasts for 2 picoseconds (τ = 2 ps in the WE terminology). This
example uses the ``NoResampler`` which performs no resampling so this is
equivalent to running 10 simulations in parallel. More details on
resampling will be given in later sections.

We first start by intializing our simulation system and importing the
required packages.

.. code:: python

   from copy import copy

   import openmm as omm
   import openmm.unit as unit

   from wepy.util.test_system_builder import NaClPair
   from wepy.util.mdtraj import mdtraj_to_json_topology

   from wepy.resampling.resamplers.resampler import NoResampler
   from wepy.runners.openmm import OpenMMRunner, gen_walker_state
   from wepy.walker import Walker
   from wepy.sim_manager import Manager

   from wepy.reporter.dashboard import DashboardReporter
   from wepy.reporter.openmm import OpenMMRunnerDashboardSection


   # number of cycles of sampling to perform
   n_cycles = 5
   # number of steps in each dynamics run
   n_steps = 1000
   steps = [n_steps for i in range(n_cycles)]
   # number of parallel simulations to run
   n_walkers = 50

   # use a ready made system for OpenMM MD simulation
   test_sys = NaClPair()

-  Now that we have a system, we will continue by creating the core
   components of a wepy simulation:

   -  **Runner**: for running dynamics.
   -  **Resampler**: for performing resampling (i.e. cloning and merging
      of walkers using the WE algorithm).
   -  **Manager**: for executing the main simulation loop.

.. code:: python

   test_mdj_topology = test_sys.mdtraj_topology
   json_top = mdtraj_to_json_topology(test_mdj_topology)

   integrator = omm.LangevinIntegrator(300.0*unit.kelvin,
                                       1/unit.picosecond,
                                       0.002*unit.picoseconds)

   init_state = gen_walker_state(test_sys.positions, test_sys.system, integrator)
   # create the initial walkers with equal weights
   init_weight = 1.0 / n_walkers
   init_walkers = [Walker(copy(init_state), init_weight) for i in range(n_walkers)]

   runner = OpenMMRunner(test_sys.system, test_sys.topology, integrator,
                       platform='Reference')

   # a trivial resampler which does nothing
   resampler = NoResampler()

We are almost there, all we need is to create the reporter where we will
store the results of the simulation. Then we are ready to run!

.. code:: python

   # Set up the dashboard reporter
   dashboard_path = 'wepy.dash.org'
   openmm_dashboard_sec = OpenMMRunnerDashboardSection(runner)
   dashboard_reporter = DashboardReporter(file_path = dashboard_path,
                                       runner_dash = openmm_dashboard_sec)

Now we can start our simulation.

.. code:: python

   sim_manager = Manager(init_walkers,
                       runner=runner,
                       resampler=resampler,
                       reporters=[dashboard_reporter]
                       )

   # run the simulation and get the results
   final_walkers, _ = sim_manager.run_simulation(n_cycles, steps)

You can run this wepy simulation by running this on the command line
after you have copied it to a file (named, say, ``wepy_script.py``):

.. code:: bash

   python wepy_script.py

After running the simulation you should see one file appear in your
directory: wepy.dash.org. This is a text file that can be opened with
emacs, or any other text editor. It contains a multitude of information
about the progress of the simulation, the timings of the trajectory
segments and can be extended to include other information as well.

Wepy is extremely customizable and just about any component can be
changed to match your needs. For now we just show you one of the
simplest possible examples of running wepy using OpenMM to give you a
flavor of how this looks like. In the following sections, we will go
into more detail about using a resampler and reporters which will be
essential for restarting/continuing a simulation.
