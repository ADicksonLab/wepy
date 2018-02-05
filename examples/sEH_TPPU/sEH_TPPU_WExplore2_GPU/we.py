import pickle

import numpy as np
import pandas as pd

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

from wepy.sim_manager import Manager
from wepy.resampling.distances.distance import Distance
from wepy.resampling.scoring.scorer import AllToAllScorer
from wepy.resampling.wexplore2 import WExplore2Resampler
from wepy.resampling.distances.openmm import OpenMMUnbindingDistance
from wepy.runners.openmm import OpenMMRunner, OpenMMWalker
from wepy.runners.openmm import UNIT_NAMES
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.reporter.reporter import WalkersPickleReporter
from wepy.work_mapper.gpu import GPUMapper

if __name__ == "__main__":

    # SETUP ------------------------------------------------------

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
    lig_idxs = pdb.topology.select('resname "2RV"')
    protein_idxs = np.array([atom.index for atom in pdb.topology.atoms if atom.residue.is_protein])

    # select water atom indices
    water_atom_idxs = pdb.top.select("water")
    #select protein and ligand atom indices
    protein_lig_idxs = [atom.index for atom in pdb.topology.atoms
                    if atom.index not in water_atom_idxs]

    # selects protien atoms which have less than 8 A from ligand
    # atoms in the crystal structure

    neighbors_idxs = mdj.compute_neighbors(pdb, 0.8, lig_idxs)
    # selects protein atoms from neighbors list
    binding_selection_idxs = np.intersect1d(neighbors_idxs, protein_idxs)

    # create a system for use in OpenMM

    # load the psf which is needed for making a system in OpenMM with
    # CHARMM force fields
    psf = omma.CharmmPsfFile('../sEH_TPPU_system.psf')

    # set the box size lengths and angles
    psf.setBox(8.2435*unit.nanometer, 8.2435*unit.nanometer, 8.2435*unit.nanometer, 90.0*unit.degree, 90.0*unit.degree, 90.0*unit.degree)

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

    # make this a constant temperature and pressure simulation at 1.0
    # atm, 300 K, with volume move attempts every 50 steps
    barostat = omm.MonteCarloBarostat(1.0*unit.atmosphere, 300.0*unit.kelvin, 50)

    # add it as a "Force" to the system
    system.addForce(barostat)

    # set the string identifier for the platform to be used by openmm
    platform = 'CUDA'

    # make an integrator object that is constant temperature
    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                            1/unit.picosecond,
                                            0.002*unit.picoseconds)

    #### END SETUP -----------------------------------------------------------------

    # set up the OpenMMRunner with your system
    runner = OpenMMRunner(system, psf.topology, integrator, platform=platform)

    # set up parameters for running the simulation
    num_walkers = 8
    # initial weights
    init_weight = 1.0 / num_walkers

    # a list of the initial walkers
    init_walkers = [OpenMMWalker(omm_state, init_weight) for i in range(num_walkers)]

    # set up unbinding distance function
    unb_distance = OpenMMUnbindingDistance(topology=pdb.topology,
                                           ligand_idxs=lig_idxs,
                                           binding_site_idxs=binding_selection_idxs)

    # make a WExplore2 resampler with default parameters and our
    # distance metric
    resampler = WExplore2Resampler(scorer=unb_distance,
                                   pmax=0.5)


    # makes ref_traj and selects lingand_atom and protein atom  indices
    # instantiate a wexplore2 unbindingboudaryconditiobs
    ubc = UnbindingBC(cutoff_distance=1.0,
                      initial_state=init_walkers[0].state,
                      topology=pdb.topology,
                      ligand_idxs=lig_idxs,
                      binding_site_idxs=protein_idxs)


    # make a dictionary of units for adding to the HDF5
    units = dict(UNIT_NAMES)


    # instantiate a reporter for HDF5
    report_path = 'wepy_results.h5'
    # open it in truncate mode first, then switch after first run
    hdf5_reporter = WepyHDF5Reporter(report_path, mode='w',
                                    save_fields=['positions', 'box_vectors', 'velocities'],
                                    decisions=resampler.DECISION.ENUM,
                                    instruction_dtypes=resampler.DECISION.instruction_dtypes(),
                                    warp_dtype=ubc.WARP_INSTRUCT_DTYPE,
                                    warp_aux_dtypes=ubc.WARP_AUX_DTYPES,
                                    warp_aux_shapes=ubc.WARP_AUX_SHAPES,
                                    topology=sEH_TPPU_system_top_json,
                                    units=units,
                                     sparse_fields={'velocities' : 10},
                                    # sparse atoms fields
                                    main_rep_idxs=protein_lig_idxs,
                                    all_atoms_rep_freq=10
    )


    pkl_reporter = WalkersPickleReporter(save_dir='./pickle_backups', freq=1, num_backups=2)

    # create a work mapper for NVIDIA GPUs for a GPU cluster
    num_workers = 4
    gpumapper  = GPUMapper(num_walkers, n_workers=num_workers)

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=ubc,
                          work_mapper=gpumapper.map,
                          reporters=[hdf5_reporter, pkl_reporter])
    n_steps = 10000
    n_cycles = 10

    # run a simulation with the manager for n_steps cycles of length 1000 each
    steps = [ n_steps for i in range(n_cycles)]
    print("Running simulation")
    sim_manager.run_simulation(n_cycles,
                               steps,
                               debug_prints=True)

    # your data should be in the 'wepy_results.h5'
