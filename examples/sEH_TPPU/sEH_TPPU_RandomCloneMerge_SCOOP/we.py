import sys
from collections import namedtuple
import pickle

import numpy as np
import pandas as pd
import h5py

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import scoop.futures

import mdtraj as mdj

from wepy.sim_manager import Manager
from wepy.resampling.clone_merge import RandomCloneMergeResampler
from wepy.openmm import OpenMMRunner, OpenMMWalker
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.boundary_conditions.unbinding import UnbindingBC

if __name__ == "__main__":

    #### SETUP -----------------------------------------
    # load a json string of the topology
    with open("../sEH_TPPU_system.top.json", mode='r') as rf:
        sEH_TPPU_system_top_json = rf.read()

    # load the pdb: this topology (for now) is needed in the WEXplore2
    # resampler which uses OpenMM to compute RMSDs and distances
    # through periodic boundary conditions
    pdb = mdj.load_pdb('../sEH_TPPU_system.pdb')

    # load the openmm state that is used to set the state of the
    # OpenMMWalker
    with open("../initial_openmm_state.pkl", mode='rb') as rf:
        omm_state = pickle.load(rf)

    # selecting ligand and protein binding site atom indices for
    # resampler and boundary conditions
    pdb = pdb.remove_solvent()
    lig_idxs = pdb.topology.select('resname "2RV"')
    atom_idxs = [atom.index for atom in pdb.topology.atoms]
    protein_idxs = np.delete(atom_idxs, lig_idxs)

    # selects protien atoms which have less than 2.5 A from ligand
    # atoms in the crystal structure
    binding_selection_idxs = mdj.compute_neighbors(pdb, 0.8, lig_idxs)
    binding_selection_idxs = np.delete(binding_selection_idxs, lig_idxs)

    # create a system for use in OpenMM

    # load the psf which is needed for making a system in OpenMM with
    # CHARMM force fields
    psf = omma.CharmmPsfFile('../sEH_TPPU_system.psf')

    # set the box size lengths and angles
    psf.setBox(82.435, 82.435, 82.435, 90, 90, 90)

    # to use charmm forcefields get your parameters
    params = omma.CharmmParameterSet('../all36_cgenff.rtf',
                                     '../all36_cgenff.prm',
                                     '../all36_prot.rtf',
                                     '../all36_prot.prm',
                                     '../tppu.str',
                                     '../toppar_water_ions.str')

    # create a system using the topology method giving it a topology and
    # the method for calculation
    system = psf.createSystem(params,
                              nonbondedMethod=omma.CutoffPeriodic,
                              nonbondedCutoff=1.0 * unit.nanometer,
                              constraints=omma.HBonds)

    # set the string identifier for the platform to be used by openmm
    platform = 'CUDA'

    #### END SETUP -----------------------------------------------------------------

    # set up parameters for running the simulation
    num_walkers = 8
    # initial weights, split equally between the walkers
    init_weight = 1.0 / num_walkers
    # make the initial walkers
    init_walkers = [OpenMMWalker(omm_state, init_weight) for i in range(num_walkers)]

    # set up the OpenMMRunner with your system
    runner = OpenMMRunner(system, psf.topology)

    # makes ref_traj and selects lingand_atom and protein atom  indices
    # instantiate a wexplore2 unbindingboudaryconditiobs
    ubc = UnbindingBC(cutoff_distance=1.0,
                      initial_state=init_walkers[0],
                      topology=pdb.topology,
                      ligand_idxs=lig_idxs,
                      binding_site_idxs=protein_idxs)

    # set up the RandomResampler with the same random seed
    resampler = RandomCloneMergeResampler()

    # instantiate a reporter for HDF5
    report_path = 'wepy_results.h5'
    reporter = WepyHDF5Reporter(report_path, mode='w',
                                decisions=resampler.DECISION,
                                instruction_dtypes=resampler.INSTRUCTION_DTYPES,
                                bc_dtype=ubc.WARP_INSTRUCT_DTYPE,
                                topology=sEH_TPPU_system_top_json)

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          boundary_conditions=ubc,
                          resampler=resampler,
                          # the mapper may be swapped out for simply `map`
                          work_mapper=scoop.futures.map,
                          reporter=reporter)


    # run a simulation with the manager for 3 cycles of length 1000 each
    print("Running simulation")
    sim_manager.run_simulation(3, [1000, 1000, 1000])

    # your data should be in the 'wepy_results.h5'

