import os
import os.path as osp
import pickle
import sys
from copy import copy, deepcopy

import numpy as np
import h5py

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

from wepy.util.mdtraj import mdtraj_to_json_topology

# the simulation manager and work mapper for actually running the simulation
from wepy.sim_manager import Manager
from wepy.work_mapper.mapper import WorkerMapper

# the runner for running dynamics and making and it's particular
# state class
from wepy.runners.openmm import OpenMMRunner, OpenMMState, OpenMMGPUWorker, UNIT_NAMES
from wepy.walker import Walker

# distance metric
from wepy.resampling.distances.receptor import UnbindingDistance

# resampler
from wepy.resampling.resamplers.wexplore import WExploreResampler

# boundary condition object for ligand unbinding
from wepy.boundary_conditions.unbinding import UnbindingBC

# reporters
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.reporter.wexplore.dashboard import WExploreDashboardReporter

# Orchestration
from wepy.orchestration.orchestrator import WepySimApparatus, Orchestrator, dump_orchestrator
from wepy.orchestration.configuration import Configuration


## INPUTS/OUTPUTS

# the inputs directory
inputs_dir = osp.realpath('./inputs')
# the outputs path
outputs_dir = osp.realpath('./outputs')
# make the outputs dir if it doesn't exist
os.makedirs(outputs_dir, exist_ok=True)

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

# normalize the input paths
json_top_path = osp.join(inputs_dir, json_top_filename)
omm_state_path = osp.join(inputs_dir, omm_state_filename)
charmm_psf_path = osp.join(inputs_dir, charmm_psf_filename)
charmm_param_paths = [osp.join(inputs_dir, filename) for filename
                      in charmm_param_files]

pdb_path = osp.join(inputs_dir, starting_coords_pdb)


## Helper functions

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



## PARAMETERS

# OpenMM simulation parameters

# cubic simulation side length
CUBE_LENGTH = 8.2435*unit.nanometer
# angles of the cubic simulation box
CUBE_ANGLE = 90*unit.degree

# vector of all lengths
LENGTHS = [CUBE_LENGTH for i in range(3)]
ANGLES = [CUBE_ANGLE for i in range(3)]


# method for nonbonded interactions
NONBONDED_METHOD = omma.CutoffPeriodic
# distance cutoff for non-bonded interactions
NONBONDED_CUTOFF = 1.0 * unit.nanometer

# constraints on MD calculations
MD_CONSTRAINTS = omma.HBonds

# force field parameters
FORCE_FIELD = omma.CharmmParameterSet(*charmm_param_paths)

# Monte Carlo Barostat
# pressure to be maintained
PRESSURE = 1.0*unit.atmosphere
# temperature to be maintained
TEMPERATURE = 300.0*unit.kelvin
# frequency at which volume moves are attempted
VOLUME_MOVE_FREQ = 50

# Platform used for OpenMM which uses different hardware computation
# kernels. Options are: Reference, CPU, OpenCL, CUDA.

# CUDA is the best for NVIDIA GPUs
PLATFORM = 'OpenCL'

# Langevin Integrator
FRICTION_COEFFICIENT = 1/unit.picosecond
# step size of time integrations
STEP_SIZE = 0.002*unit.picoseconds

# Resampler parameters

# the maximum weight allowed for a walker
PMAX = 0.1
# the minimum weight allowed for a walker
PMIN = 1e-12

# the maximum number of regions allowed under each parent region
MAX_N_REGIONS = (10, 10, 10, 10)

# the maximum size of regions, new regions will be created if a walker
# is beyond this distance from each voronoi image unless there is an
# already maximal number of regions
MAX_REGION_SIZES = (1, 0.5, .35, 0.25) # nanometers

# boundary condition parameters

# maximum distance between between any atom of the ligand and any
# other atom of the protein, if the shortest such atom-atom distance
# is larger than this the ligand will be considered unbound and
# restarted in the initial state
CUTOFF_DISTANCE = 1.0 # nm


# minimization parameters
NUMBER_OF_EQUILIB_STEPS = 10


# Distance metric parameters, these are not used in OpenMM and so
# don't need the units associated with them explicitly, so be careful!

# distance from the ligand in the crystal structure used to determine
# the binding site, used to align ligands in the Unbinding distance
# metric
BINDING_SITE_CUTOFF = 0.8 # in nanometers

# the residue id for the ligand so that it's indices can be determined
LIG_RESID = "2RV"

# Initial experimental coordinates
PDB = mdj.load_pdb(pdb_path)
EXPERIMENTAL_POSITIONS = PDB.xyz[0]
EXPERIMENTAL_OMM_POSITIONS = PDB.openmm_positions(frame=0)

# topologies
TOP_MDTRAJ = PDB.topology
TOP_JSON = mdtraj_to_json_topology(TOP_MDTRAJ)

# the protein, binding site, and ligand atoms
BS_IDXS = binding_site_atoms(TOP_MDTRAJ, LIG_RESID, EXPERIMENTAL_POSITIONS)
LIG_IDXS = ligand_idxs(TOP_MDTRAJ, LIG_RESID)
PROT_IDXS = protein_idxs(TOP_MDTRAJ)


# reporting parameters

# these are the properties of the states (i.e. from OpenMM) which will
# be saved into the HDF5
SAVE_FIELDS = ('positions', 'box_vectors', 'velocities')
# these are the names of the units which will be stored with each
# field in the HDF5
UNITS = UNIT_NAMES
# this is the frequency to save the full system as an alternate
# representation, the main "positions" field will only have the atoms
# for the protein and ligand which will be determined at run time
ALL_ATOMS_SAVE_FREQ = 10
# we can specify some fields to be only saved at a given frequency in
# the simulation, so here each tuple has the field to be saved
# infrequently (sparsely) and the cycle frequency at which it will be
# saved
SPARSE_FIELDS = (('velocities', 10),
                )

# the idxs of the main representation
MAIN_REP_IDXS = np.concatenate((LIG_IDXS, PROT_IDXS))

## OpenMM System setup

# load the charmm file for the topology
psf = omma.CharmmPsfFile(charmm_psf_path)

omm_topology = psf.topology

# set the box size lengths and angles
psf.setBox(*LENGTHS, *ANGLES)

# create a system using the topology method giving it a topology and
# the method for calculation
system = psf.createSystem(FORCE_FIELD,
                          nonbondedMethod=NONBONDED_METHOD,
                          nonbondedCutoff=NONBONDED_CUTOFF,
                          constraints=MD_CONSTRAINTS)

# barostat to keep pressure constant
barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)

# set up for a short simulation to minimize and prepare
# instantiate an integrator
integrator = omm.LangevinIntegrator(TEMPERATURE,
                                    FRICTION_COEFFICIENT,
                                    STEP_SIZE)

# set the platform that you are going to run it on
platform = omm.Platform.getPlatformByName(PLATFORM)


## Minimization

# instantiate a simulation object
simulation = omma.Simulation(omm_topology, system, integrator, platform)

# initialize the positions
simulation.context.setPositions(EXPERIMENTAL_OMM_POSITIONS)

print("\nminimizing\n")
# minimize the energy
simulation.minimizeEnergy()

# run the simulation for a number of initial time steps
simulation.step(NUMBER_OF_EQUILIB_STEPS)
print("done minimizing\n")

# get the initial state from the context
MINIMIZED_INIT_OMM_STATE = simulation.context.getState(getPositions=True,
                                              getVelocities=True,
                                              getParameters=True,
                                              getForces=True,
                                              getEnergy=True,
                                              getParameterDerivatives=True)



#### Wepy Orchestrator

### Apparatus

## Runner
RUNNER = OpenMMRunner(system, psf.topology, integrator, platform=PLATFORM)

# the initial state, which is used as reference for many things
INIT_STATE = OpenMMState(MINIMIZED_INIT_OMM_STATE)

## Resampler

# Distance Metric

# make the distance metric with the ligand and binding site
# indices for selecting atoms for the image and for doing the
# alignments to only the binding site. All images will be aligned
# to the reference initial state
DISTANCE_METRIC = UnbindingDistance(LIG_IDXS, BS_IDXS, INIT_STATE)

# WExplore Resampler

# make a Wexplore resampler with default parameters and our
# distance metric
RESAMPLER = WExploreResampler(distance=DISTANCE_METRIC,
                               init_state=INIT_STATE,
                               max_n_regions=MAX_N_REGIONS,
                               max_region_sizes=MAX_REGION_SIZES,
                               pmin=PMIN, pmax=PMAX)

## Boundary Conditions

# makes ref_traj and selects lingand_atom and protein atom  indices
# instantiate a revo unbindingboudaryconditiobs
BC = UnbindingBC(cutoff_distance=CUTOFF_DISTANCE,
                  initial_state=INIT_STATE,
                  topology=TOP_MDTRAJ,
                  ligand_idxs=LIG_IDXS,
                  receptor_idxs=PROT_IDXS)


APPARATUS = WepySimApparatus(RUNNER, resampler=RESAMPLER,
                             boundary_conditions=BC)

print("created apparatus")

## CONFIGURATION

# REPORTERS
# list of reporter classes and partial kwargs for using in the
# orchestrator

# WepyHDF5

REPORTER_CLASSES = [WepyHDF5Reporter, WExploreDashboardReporter]

hdf5_reporter_kwargs = {'save_fields' : SAVE_FIELDS,
                        'topology' : TOP_JSON,
                        'resampler' : RESAMPLER,
                        'boundary_conditions' : BC,
                        'units' : dict(UNITS),
                        'sparse_fields' : dict(SPARSE_FIELDS),
                        'main_rep_idxs' : MAIN_REP_IDXS,
                        'all_atoms_rep_freq' : ALL_ATOMS_SAVE_FREQ}

dashboard_reporter_kwargs = {'step_time' : STEP_SIZE.value_in_unit(unit.second),
                             'max_n_regions' : RESAMPLER.max_n_regions,
                             'max_region_sizes' : RESAMPLER.max_region_sizes,
                             'bc_cutoff_distance' : BC.cutoff_distance}

REPORTER_KWARGS = [hdf5_reporter_kwargs, dashboard_reporter_kwargs]

N_WORKERS = 8

CONFIGURATION = Configuration(n_workers=N_WORKERS,
                              reporter_classes=REPORTER_CLASSES,
                              reporter_partial_kwargs=REPORTER_KWARGS)

print("created configuration")

### Initial Walkers
N_WALKERS = 48
INIT_WEIGHT = 1.0 / N_WALKERS
INIT_WALKERS = [Walker(deepcopy(INIT_STATE), INIT_WEIGHT) for i in range(N_WALKERS)]

print("created init walkers")

### Orchestrator
ORCHESTRATOR = Orchestrator(APPARATUS,
                            default_init_walkers=INIT_WALKERS,
                            default_configuration=CONFIGURATION)

print("created orchestrator, creating object file now")

ORCH_NAME = "sEH-TPPU"
dump_orchestrator(ORCHESTRATOR, "{}.orch".format(ORCH_NAME), mode='wb')
