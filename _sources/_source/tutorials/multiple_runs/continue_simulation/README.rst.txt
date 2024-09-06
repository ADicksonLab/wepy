Continue the Simulation
=======================

Easiest way to continue a wepy simulation is to use the pickle file that
was saved in the previous simulation. You can use the walker states from
the previous simulation to continue where you left off. It is a good
idea to seperate the folders of the two simulations to avoid any
confusion. Later on, you can merge the hdf5 files and continue your
analysis. Let's first define the input and output directories for the
new simulation, input being the previous simulation folder.

While initializing the previous simulation, we set up a
WalkerPklReporter to save the simulation state. Depending on the
parameters you set, you can find the last pickkle files in your
simulation folder. For example, if you have ``num_backups=1`` and
``save_dir='pkls'``, you can find the state of the last cycle in
``pkls/walkers_cycle_<last_cycle>.pkl``.

Let's start by importing the necessary libraries for the new simulation.

.. code:: python

   import os
   import os.path as osp
   import pickle as pkl

   import simtk.openmm.app as omma
   import simtk.openmm as omm
   import simtk.unit as unit

   import mdtraj as mdj

   from wepy.sim_manager import Manager
   from wepy.resampling.resamplers.revo import REVOResampler
   from wepy.resampling.distances.receptor import UnbindingDistance
   from wepy.runners.openmm import OpenMMGPUWalkerTaskProcess, OpenMMRunner, OpenMMWalker, OpenMMState, gen_sim_state
   from wepy.boundary_conditions.receptor import UnbindingBC
   from wepy.reporter.hdf5 import WepyHDF5Reporter
   from wepy.work_mapper.task_mapper import TaskMapper
   from wepy.util.mdtraj import mdtraj_to_json_topology

   from walker_pkl_reporter import WalkersPickleReporter

   from wepy.reporter.dashboard import DashboardReporter
   from wepy.reporter.openmm import OpenMMRunnerDashboardSection

Let's assume that your previous run had 150 cycles and you want to
continue the simulation with 150 more cycles. Since the cycle indices
start from 0, 150th cycle will have the index of 149. You can confirm
how many cycles that your previous run had by (1) checking the last
pickle file or (2) in wepy.dashboard.org denoted as 'Number of Cycles'.
It is important to note that simulation folders for previous and
continuing simulation should be different. If not, the new simulation
will not proceed since the .h5 file already exists. Later on, you can
merge the .h5 files to continue your analysis.

You can set the parameters for the new simulation as follows:

.. code:: python

   # These parameters can be different for your system, make sure to change them accordingly
   num_walkers = 8
   n_run = 1   # Run number
   n_steps = 100000 # Number of steps for the new simulation
   n_cycles = 150 # Number of cycles for the new simulation
   n_last_cycles = 150 # How many cycles last simulation had
   last_cycle = 149 # ID for the pickle file of the last cycle

   inp_path = './'
   pdb_path = f'{inp_path}/step3_input.pdb'
   rst_path = f'{inp_path}/step5_10.rst'
   system_path = f'{inp_path}/system.pkl'
   topology_path = f'{inp_path}/topology.pkl'
   input_dir = f'simdata_run{n_run}_steps{n_steps}_cycs{n_last_cycles}_1' # Previous simulation directory
   output_dir = f'simdata_run{n_run}_steps{n_steps}_cycs{n_last_cycles}_2' # New simulation directory
   os.makedirs(outputs_dir, exist_ok=True)

Next, we can setup the simulation as we did in the Unbinding Simulation
tutorial. You can use the same setup for the simulation with the
addition of loading the last cycle of the previous simulation:

.. code:: python

   with open(system_path,'rb') as f:
   system = pkl.load(f)

   with open(topology_path,'rb') as f:
       omm_top = pkl.load(f)

   # Get positions and box vectors from an rst file
   # This is typically a restart file post nvt/npt from openmm
   # However, you can build it from scratch using omma.PDBFile() too
   with open(rst_path, 'r') as f:
       simtk_state = omm.XmlSerializer.deserialize(f.read())
       bv = simtk_state.getPeriodicBoxVectors()
       pos = simtk_state.getPositions()

   system.setDefaultPeriodicBoxVectors(bv[0],bv[1],bv[2])

   with open (f'{input_dir}/pkls/walkers_cycle_{last_cycle}.pkl', 'rb') as f:
       last_cycle = pkl.load(f)

Now that we have our previous pickle file, we can extract the walkers.
This is the important step that allows us to continue the simulation. We
will use these walkers to set walker states and walker weights for our
simulation objects.

.. code:: python

   init_walkers = []

   for walker in last_cycle:
       init_walkers.append(walker)

Then, we need an integrator and a runner for our simulation.

.. code:: python

   integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                   1/unit.picosecond,
                                   0.002*unit.picoseconds)

   runner = OpenMMRunner(system, omm_top, integrator, platform='CUDA')

Now that we have the walkers and the runner, we need to introduce the
distance metric for measuring differences between states as we did in
Unbinding Simulation tutorial.

.. code:: python

   pdb = mdj.load_pdb(pdb_path)
   json_top = mdtraj_to_json_topology(pdb.top)

   # Save some relevant indices
   lig_idxs = pdb.top.select('resname UNK') # TODO: Update
   protein_idxs = pdb.top.select('protein and not resname UNK') # TODO: update
   binding_selection_idxs =  mdj.compute_neighbors(pdb, 0.5, lig_idxs, haystack_indices=protein_idxs, periodic=True)[0]

   # Distance metric to be used in REVO
   unb_distance = UnbindingDistance(lig_idxs,
                                   binding_selection_idxs,
                                   [walker.state for walker in init_walkers])

Next, we will setup the boundary conditions for the simulation.

.. code:: python

   ubc = UnbindingBC(cutoff_distance=1.0,  # nm
                   initial_state=[walker.state for walker in init_walkers],
                   initial_weights=[walker.weight for walker in init_walkers],
                   topology=json_top,
                   ligand_idxs=lig_idxs,
                   receptor_idxs=protein_idxs)

Next, we will use the REVOResampler class from the wepy library to setup
the resampler.

.. code:: python

   # Set up the REVO Resampler with the parameters
   resampler = REVOResampler(distance=unb_distance,
                           init_state=[walker.state for walker in init_walkers],
                           weights=True,
                           pmax=0.1,
                           dist_exponent=4,
                           merge_dist=0.25,
                           char_dist=0.1)

And finally, we need to setup the reporters for recording the simulation
data, define the task mapper and the simulation manager.

.. code:: python

   # Set up the HDF5 reporter
   hdf5_reporter = WepyHDF5Reporter(save_fields=('positions','box_vectors'),
                               file_path=osp.join(outputs_dir,f'wepy.results.h5') ,
                               resampler=resampler,
                               boundary_conditions=ubc,
                               topology=json_top)

   # Set up the pickle reporter (Essential for restarts)
   out_folder_pkl = osp.join(outputs_dir,f'pkls')
   pkl_reporter = WalkersPickleReporter(save_dir = out_folder_pkl,
                                   freq = 1,
                                   num_backups = 2)

   # Set up the dashboard reporter
   dashboard_path = osp.join(outputs_dir,f'wepy.dash.org')
   openmm_dashboard_sec = OpenMMRunnerDashboardSection(runner)
   dashboard_reporter = DashboardReporter(file_path = dashboard_path,
                                       runner_dash = openmm_dashboard_sec)


   # Create a work mapper for NVIDIA GPUs for a GPU cluster
   mapper = TaskMapper(walker_task_type=OpenMMGPUWalkerTaskProcess,
                       num_workers=2,
                       platform='CUDA',
                       device_ids=[0,1])

   # Build the simulation manager
   sim_manager = Manager(init_walkers,
                       runner=runner,
                       resampler=resampler,
                       boundary_conditions=ubc,
                       work_mapper=mapper,
                       reporters=[hdf5_reporter, pkl_reporter, dashboard_reporter])

   print('Running the simulation...')
   # run a simulation with the manager for 'n_cycles' with 'n_steps' of integrator steps in each
   steps_list = [n_steps for i in range(n_cycles)]

   # and..... go!
   sim_manager.run_simulation(n_cycles,
                           steps_list)

And that is it! Now you can start the simulation by running
``python wepy_run.py``. The simulation will continue for the specified
number of cycles and steps. The simulation data will be saved in the
``outputs_dir`` directory. Later on, we will cover how to merge the hdf5
files from the two simulations to continue the analysis.
