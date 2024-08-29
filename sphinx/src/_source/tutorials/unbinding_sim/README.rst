Unbinding Simulation
==============================

In the previous step we mentioned about how to prepare your data for running a Wepy simulation. In this part, we will cover the topic of how to setup the simulation with this data.

Which files you need depends on whether you are doing a rebinding or unbinding simulation. Since we are focusing on unbinding simulation for this tutorial we need the following files:

- system.pkl : Generated in the `Building System and Topology Pickle Files` section.
- topology.pkl : Generated in the `Building System and Topology Pickle Files` section.
- \*.rst : A restart file from the simulation you want to continue from. This is need for getting the positions and boundary values of the system. E.g. `step5_10.rst` in this case.
- \*.pdb : A PDB file of the system. E.g. `step3_input.pdb` in this case.

If you do not have these files, please visit the :any:`How to Prepare Your Data <prepare_unbinding>` section of this tutorial.

After confirming that you have the necessary files, you can proceed to the next step. Let's start by importing necessary modules.

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


Great! Now that we have imported the necessary modules, we can proceed to setup simulation parameters file directories. In this tutorial, we will utilize 8 walkers and run 2 cycles of 5 steps. These parameters can be adjusted to suit your needs since these parameters are for a short demonstration. 

.. TODO: Provide a brief explanation on number of steps and number of cycles.

.. code:: python

    # Typical sim details
    num_walkers = 8
    run = 1
    n_steps = 5
    n_cycles = 2

    #### Paths: Set it to your preference
    inp_path = './'
    pdb_path = f'{inp_path}/step3_input.pdb'
    rst_path = f'{inp_path}/step5_10.rst'
    system_path = f'{inp_path}/system.pkl'
    topology_path = f'{inp_path}/topology.pkl'
    outputs_dir = f'simdata_run{run}_steps{n_steps}_cycs{n_cycles}'
    os.makedirs(outputs_dir, exist_ok=True)

The next step is to use the files we have prepared to setup the simulation. We will start by loading the system and topology files.

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

Now we need an integrator and a runner for our simulation. an integrator is a computational method that updates the positions and velocities of particles over time. It calculates the motion of particles based on the forces acting on them, following Newton's laws of motion. In this tutorial, we will use the Langevin integrator. This simulation will be running on a GPU, so we will use the `platform='CUDA'` option while setting up the `OpenMMRunner`.

.. code:: python

    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                    1/unit.picosecond,
                                    0.002*unit.picoseconds)

    runner = OpenMMRunner(system, omm_top, integrator, platform='CUDA')

Next, we will build the walkers for the simulation. For this, we need to generate the state from the system and topology files. All of the initial walkers will have the same state and weight at the start of the simulation.

.. code:: python

    # Generate a new simtk "state"
    new_simtk_state = gen_sim_state(pos,
                                    system,
                                    integrator)

    # Set up parameters for running the simulation
    init_weight = 1.0 / num_walkers

    # Generate the walker state in wepy format
    walker_state = OpenMMState(new_simtk_state)
        
    # Make a list of the initial walkers
    init_walkers = [OpenMMWalker(walker_state, init_weight) for i in range(num_walkers)]

Now that we have the walkers, we need a distance metric for measuring differences between walker states. In this case, we will use the UnbindingDistance class from the wepy library, however you can define your own distance metrics if needed. For this tutorial, we will use the the indices of the ligand, binding site and the walker state.

.. code:: python

    pdb = mdj.load_pdb(pdb_path)
    json_top = mdtraj_to_json_topology(pdb.top)

    # Save some relevant indices
    lig_idxs = pdb.top.select('<selection of your ligand>')
    protein_idxs = pdb.top.select('<selection of your protein>')
    binding_selection_idxs =  mdj.compute_neighbors(pdb, 0.5, lig_idxs, haystack_indices=protein_idxs, periodic=True)[0]

    # Distance metric to be used in REVO
    unb_distance = UnbindingDistance(lig_idxs,
                                     binding_selection_idxs,
                                     walker_state)

Next, we will setup the boundary conditions for the simulation. In this case, we will use the UnbindingBC class from the wepy library. This class will be used to check if the walker has crossed the boundary and should be resampled.

.. code:: python

    ubc = UnbindingBC(cutoff_distance=1.0,  # nm
                      initial_state=walker_state,
                      topology=json_top,
                      ligand_idxs=lig_idxs,
                      receptor_idxs=protein_idxs)


.. TODO: Add explanation for resampler.

Next, we will use the REVOResampler class from the wepy library to setup the resampler. This class will be used to resample the walkers based on the distance metric and boundary conditions we have defined.

.. code:: python

    # Set up the REVO Resampler with the parameters
    resampler = REVOResampler(distance=unb_distance,
                              init_state=walker_state,
                              weights=True,
                              pmax=0.1,
                              dist_exponent=4,
                              merge_dist=0.25,
                              char_dist=0.1)

And finally, we need to setup the reporters for recording the simulation data, define the task mapper and the simulation manager. An important point in this step is to define the number of workers and the device ids for the GPU. In this case, we will use 2 workers and the GPU device ids 0 and 1. If you are using more GPUs, you can increase the number of workers and device ids accordingly.

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

And that is it! Now you can start the simulation by running ``python wepy_run.py``. The simulation will run for the specified number of cycles and steps. The simulation data will be saved in the ``outputs_dir`` directory. You can use the saved data to analyze the simulation results.