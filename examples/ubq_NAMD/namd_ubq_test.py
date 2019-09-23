import pickle
import logging
import os
import os.path as osp

import math
import numpy as np
import pandas as pd

from wepy.work_mapper.gpu_namd import GPUMapper

import mdtraj as mdj

from wepy.util.mdtraj import mdtraj_to_json_topology, json_to_mdtraj_topology
from wepy.util.util import box_vectors_to_lengths_angles

# the simulation manager and work mapper for actually running the simulation
from wepy.sim_manager import Manager
# the runner for running dynamics and making and it's particular
# state class
from wepy.runners.namd import NAMDRunner, NAMDWalker, generate_state

from wepy.walker import Walker, WalkerState

# distance metric
from wepy.resampling.distances.rmsd import RMSDDistance

# resampler
from wepy.resampling.resamplers.wexplore import WExploreResampler

# reporters
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.reporter.wexplore.dashboard import WExploreDashboardReporter
from wepy.reporter.wexplore.image import WExploreAtomImageReporter
from wepy.reporter.restree import ResTreeReporter

# reporting parameters

# these are the properties of the states (i.e. from OpenMM) which will
# be saved into the HDF5
SAVE_FIELDS = ('positions', 'box_vectors')
# this is the frequency to save the full system as an alternate
# representation, the main "positions" field will only have the atoms
# for the protein and ligand which will be determined at run time
ALL_ATOMS_SAVE_FREQ = 10

## INPUTS/OUTPUTS

# the inputs directory
inputs_dir = osp.realpath('./inputs')
# the outputs path
outputs_dir = osp.realpath('./outputs')
# make the outputs dir if it doesn't exist
os.makedirs(outputs_dir, exist_ok=True)

# inputs filenames
starting_coords_pdb = 'ubq_wb.pdb'

# outputs
hdf5_filename = 'ubq_test.wepy.h5'
dashboard_filename = 'ubq_test.dash.txt'

# normalize the input paths
pdb_path = osp.join(inputs_dir, starting_coords_pdb)

# normalize the output paths
hdf5_path = osp.join(outputs_dir, hdf5_filename)
dashboard_path = osp.join(outputs_dir, dashboard_filename)

# Number of dimensions to reduce the distance history to
# Used in resampling.
n_dimensions = 3

def protein_idxs(mdtraj_topology):
    return np.array([atom.index for atom in mdtraj_topology.atoms if atom.residue.is_protein])

if __name__ == "__main__":
    
    # set logging threshold
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

    crystal_traj = mdj.load_pdb(pdb_path)
    json_top = mdtraj_to_json_topology(crystal_traj.top)
    
    protein_idxs = protein_idxs(crystal_traj.top)
    
    #### END SETUP -----------------------------------------------------------------

    # set up the NAMDRunner with your system
    runcmd = "/Users/alexrd/programs/NAMD_2.13_MacOSX-x86_64-multicore/namd2 +p4"
    common_path = "/Users/alexrd/research/NAMD_runner/common"
    conf_template = osp.join(inputs_dir,"ubq_wb_eq.conf")
    work_dir = "/Users/alexrd/research/NAMD_runner/work"
    
    runner = NAMDRunner(runcmd, common_path, conf_template, work_dir)

    # set up parameters for running the simulation
    num_walkers = 24
    # initial weights
    init_weight = 1.0 / num_walkers

    # a list of the initial walkers
    state = generate_state(work_dir,"ubq_start")
    
    init_walkers = [NAMDWalker(state, init_weight, cycle=-1, next_input_pref="ubq_start") for i in range(num_walkers)]
    
    # set up RMSD distance function
    distance_metric = RMSDDistance(protein_idxs)
    
    # set up the WExplore Resampler with the parameters
    resampler = WExploreResampler(distance=distance_metric,
                                   init_state=state,
                                   max_n_regions=(10,10,10,10),
                                   max_region_sizes=(1, 0.5, .35, 0.25),
                                   pmin=1e-12, pmax=1.0)

    ## make the reporters

    # WepyHDF5
    #print(tspo_pk_system_top_json)
    # make a dictionary of units for adding to the HDF5
    # open it in truncate mode first, then switch after first run
    hdf5_reporter = WepyHDF5Reporter(file_path=hdf5_path, mode='w',
                                     # the fields of the State that will be saved in the HDF5 file
                                     save_fields=SAVE_FIELDS,
                                     # the topology in a JSON format
                                     topology=json_top,
                                     resampler=resampler,
                                     swmr_mode = True
                                    )

    reporters = [hdf5_reporter]
    
    # create a work mapper for NVIDIA GPUs for a GPU cluster
    gpumapper  = GPUMapper(num_walkers, gpu_indices=[0,1])

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=gpumapper,
                          reporters=reporters)
    n_steps = 10000
    n_cycles = 10

    # run a simulation with the manager for n_steps cycles of length 1000 each
    steps = [ n_steps for i in range(n_cycles)]
    logging.info("Running simulation")
    sim_manager.run_simulation(n_cycles,
                               steps)

    # your data should be in the 'wepy_results.h5'
