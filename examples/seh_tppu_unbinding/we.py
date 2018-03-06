import os.path as osp
import pickle

import numpy as np
import mdtraj as mdj

# OpenMM libraries for setting up simulation objects and loading
# the forcefields
import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

## Wepy classes

# the simulation manager and work mapper for actually running the simulation
from wepy.sim_manager import Manager
from wepy.work_mapper.mapper import WorkerMapper

# the runner for running dynamics and making and it's particular
# state class
from wepy.runners.openmm import OpenMMRunner, OpenMMState, OpenMMGPUWorker, UNIT_NAMES
from wepy.walker import Walker

# classes for making the resampler
from wepy.resampling.distances.receptor import UnbindingDistance
from wepy.resampling.wexplore1 import WExplore1Resampler

# A standard Boundary condition object for unbinding
from wepy.boundary_conditions.unbinding import UnbindingBC

# standard reporters
from wepy.reporter.hdf5 import WepyHDF5Reporter

## PARAMETERS

# OpenMM simulation parameters
CUBE_LENGTH = 8.2435*unit.nanometer
CUBE_ANGLE = 90*unit.degree
NONBONDED_CUTOFF = 1.0 * unit.nanometer

# Monte Carlo Barostat
PRESSURE = 1.0*unit.atmosphere
TEMPERATURE = 300.0*unit.kelvin
VOLUME_MOVE_FREQ = 50

PLATFORM = 'CUDA'

# Langevin Integrator
FRICTION_COEFFICIENT = 1/unit.picosecond
STEP_SIZE = 0.002*unit.picoseconds

# Distance metric parameters
BINDING_SITE_CUTOFF = 0.8 # in nanometers
LIG_RESID = "2RV"

# Resampler parameters
PMAX = 0.5
#PMIN =

# boundary condition parameters
CUTOFF_DISTANCE = 1.0 # nm

# reporting parameters
PICKLE_FREQUENCY = 1
NUM_PICKLES = 2
SAVE_FIELDS = ('positions', 'box_vectors', 'velocities')
UNITS = UNIT_NAMES
ALL_ATOMS_SAVE_FREQ = 10
SPARSE_FIELDS = (('velocities', 10),
                )

## INPUTS/OUTPUTS

# the inputs directory
inputs_dir = osp.realpath('./inputs')
# the outputs path
outputs_dir = osp.realpath('./outputs')

# inputs filenames
json_top_filename = "sEH_TPPU_system.top.json"
omm_state_filename = "initial_openmm_state.pkl"
charmm_psf_filename = 'sEH_TPPU_system.psf'
charmm_param_files = ['all36_cgenff.rtf',
                      'all36_cgenff.prm',
                      'all36_prot.rtf',
                      'all36_prot.prm',
                      'tppu.str',
                      'toppar_water_ions.str']

starting_coords_pdb = 'sEH_TPPU_system.pdb'

# outputs
hdf5_filename = 'wepy_results.h5'
pickles_dir = 'pickle_backups'

# normalize the input paths
json_top_path = osp.join(inputs_dir, json_top_filename)
omm_state_path = osp.join(inputs_dir, omm_state_filename)
charmm_psf_path = osp.join(inputs_dir, charmm_psf_filename)
charmm_param_paths = [osp.join(inputs_dir, filename) for filename
                      in charmm_param_files]

pdb_path = osp.join(inputs_dir, starting_coords_pdb)

# normalize the output paths
hdf5_path = osp.join(outputs_dir, hdf5_filename)
pickles_dir = osp.join(outputs_dir, pickles_dir)

def ligand_idxs(mdtraj_topology, ligand_resid):
    return mdtraj_topology.select('resname "{}"'.format(ligand_resid))

def protein_idxs(mdtraj_topology):
    return np.array([atom.index for atom in mdtraj_topology.atoms if atom.residue.is_protein])


def binding_site_atoms(mdtraj_topology, ligand_resid, coords):

    # selecting ligand and protein binding site atom indices for
    # resampler and boundary conditions
    lig_idxs = ligand_idxs(mdtraj_topology, ligand_resid)
    prot_idxs = protein_idxs(mdtraj_topology)

    # select water atom indices
    water_atom_idxs = mdtraj_topology.select("water")
    #select protein and ligand atom indices
    protein_lig_idxs = [atom.index for atom in mdtraj_topology.atoms
                        if atom.index not in water_atom_idxs]

    # make a trajectory to compute the neighbors from
    traj = mdj.Trajectory([coords], mdtraj_topology)

    # selects protein atoms which have less than 8 A from ligand
    # atoms in the crystal structure
    neighbors_idxs = mdj.compute_neighbors(traj, BINDING_SITE_CUTOFF, lig_idxs)

    # selects protein atoms from neighbors list
    binding_selection_idxs = np.intersect1d(neighbors_idxs, prot_idxs)

    return binding_selection_idxs

def main(n_runs, n_cycles, steps, n_walkers, n_workers=1, debug_prints=False, seed=None):
    ## Load objects needed for various purposes

    # load a json string of the topology
    with open(json_top_path, mode='r') as rf:
        sEH_TPPU_system_top_json = rf.read()

    # an openmm.State object for setting the initial walkers up
    with open(omm_state_path, mode='rb') as rf:
        omm_state = pickle.load(rf)

    ## set up the OpenMM Runner

    # load the psf which is needed for making a system in OpenMM with
    # CHARMM force fields
    psf = omma.CharmmPsfFile(charmm_psf_path)

    # set the box size lengths and angles
    lengths = [CUBE_LENGTH for i in range(3)]
    angles = [CUBE_ANGLE for i in range(3)]
    psf.setBox(*lengths, *angles)

    # charmm forcefields parameters
    params = omma.CharmmParameterSet(*charmm_param_paths)

    # create a system using the topology method giving it a topology and
    # the method for calculation
    system = psf.createSystem(params,
                              nonbondedMethod=omma.CutoffPeriodic,
                              nonbondedCutoff=NONBONDED_CUTOFF,
                              constraints=omma.HBonds)

    # make this a constant temperature and pressure simulation at 1.0
    # atm, 300 K, with volume move attempts every 50 steps
    barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)

    # add it as a "Force" to the system
    system.addForce(barostat)

    # make an integrator object that is constant temperature
    integrator = omm.LangevinIntegrator(TEMPERATURE,
                                        FRICTION_COEFFICIENT,
                                        STEP_SIZE)

    # set up the OpenMMRunner with the system
    runner = OpenMMRunner(system, psf.topology, integrator, platform=PLATFORM)

    ## Make the distance Metric

    # load the crystal structure coordinates
    crystal_traj = mdj.load_pdb(pdb_path)

    # get the atoms in the binding site according to the crystal structure
    bs_idxs = binding_site_atoms(crystal_traj.top, LIG_RESID, crystal_traj.xyz[0])
    lig_idxs = ligand_idxs(crystal_traj.top, LIG_RESID)
    prot_idxs = protein_idxs(crystal_traj.top)

    unb_distance = UnbindingDistance(lig_idxs, bs_idxs)

    ## Make the resampler

    # make a WExplore1 resampler with default parameters and our
    # distance metric
    resampler = WExplore1Resampler(distance=unb_distance,
                                   pmax=PMAX)

    ## Make the Boundary Conditions

    # makes ref_traj and selects lingand_atom and protein atom  indices
    # instantiate a wexplore2 unbindingboudaryconditiobs
    ubc = UnbindingBC(cutoff_distance=CUTOFF_DISTANCE,
                      initial_state=omm_state,
                      topology=crystal_traj.topology,
                      ligand_idxs=lig_idxs,
                      binding_site_idxs=bs_idxs)


    ## make the reporters

    # WepyHDF5

    # make a dictionary of units for adding to the HDF5
    # open it in truncate mode first, then switch after first run
    hdf5_reporter = WepyHDF5Reporter(hdf5_path, mode='w',
                                     # the fields of the State that will be saved in the HDF5 file
                                     save_fields=SAVE_FIELDS,
                                     # the topology in a JSON format
                                     topology=sEH_TPPU_system_top_json,
                                     # the units to save the fields in
                                     units=dict(UNITS),
                                     # the types of decisions for saving the resampling records
                                     decisions=resampler.DECISION.ENUM,
                                     # data types and shapes for other types fo records
                                     instruction_dtypes=resampler.DECISION.instruction_dtypes(),
                                     warp_dtype=ubc.WARP_INSTRUCT_DTYPE,
                                     warp_aux_dtypes=ubc.WARP_AUX_DTYPES,
                                     warp_aux_shapes=ubc.WARP_AUX_SHAPES,
                                     # sparse (in time) fields
                                     sparse_fields=dict(SPARSE_FIELDS),
                                     # sparse atoms fields
                                     main_rep_idxs=np.concatenate((prot_idxs, lig_idxs)),
                                     all_atoms_rep_freq=ALL_ATOMS_SAVE_FREQ
    )

    ## The work mapper

    # we use a mapper that uses GPUs
    work_mapper = WorkerMapper(worker_type=OpenMMGPUWorker,
                               num_workers=n_workers)

    ## Combine all these parts and setup the simulatino manager

    # set up parameters for running the simulation
    # initial weights
    init_weight = 1.0 / n_walkers

    # a list of the initial walkers
    init_walkers = [Walker(OpenMMState(omm_state), init_weight) for i in range(n_walkers)]

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=ubc,
                          work_mapper=work_mapper,
                          reporters=[hdf5_reporter])


    ### RUN the simulation
    for run_idx in range(n_runs):
        print("Starting run: {}".format(run_idx))
        sim_manager.run_simulation(n_cycles, steps, debug_prints=True)
        print("Finished run: {}".format(run_idx))


if __name__ == "__main__":
    import time
    import multiprocessing as mp
    import sys

    # needs to call spawn for starting processes due to CUDA not
    # tolerating fork
    mp.set_start_method('spawn')

    n_runs = int(sys.argv[1])
    n_cycles = int(sys.argv[2])
    n_steps = int(sys.argv[3])
    n_walkers = int(sys.argv[4])
    n_workers = int(sys.argv[5])

    print("Number of steps: {}".format(n_steps))
    print("Number of cycles: {}".format(n_cycles))

    steps = [n_steps for i in range(n_cycles)]

    start = time.time()
    main(n_runs, n_cycles, steps, n_walkers, n_workers, debug_prints=True)
    end = time.time()

    print("time {}".format(end-start))
