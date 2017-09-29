import pickle

import h5py
import numpy as np
import pandas as pd

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

from wepy.sim_manager import Manager
from wepy.resampling.wexplore2 import WExplore2Resampler
from wepy.openmm import OpenMMRunner, OpenMMWalker
from wepy.openmm import UNITS
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.hdf5 import TrajHDF5
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
    pdb = pdb.remove_solvent()
    lig_idxs = pdb.topology.select('resname "2RV"')
    print("Ligand: {}".format(','.join([str(idx) for idx in lig_idxs])))
    atom_idxs = [atom.index for atom in pdb.topology.atoms]
    protein_idxs = np.delete(atom_idxs, lig_idxs)


    # selects protien atoms which have less than 2.5 A from ligand
    # atoms in the crystal structure
    binding_selection_idxs = mdj.compute_neighbors(pdb, 0.8, lig_idxs)
    binding_selection_idxs = np.delete(binding_selection_idxs, lig_idxs)
    print("Protein: {}".format(','.join([str(idx) for idx in binding_selection_idxs])))

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
    num_walkers = 3
    # initial weights
    init_weight = 1.0 / num_walkers

    # a list of the initial walkers
    init_walkers = [OpenMMWalker(omm_state, init_weight) for i in range(num_walkers)]

    # set up the WExplore2 Resampler with the parameters
    resampler = WExplore2Resampler(topology=pdb.top,
                                   ligand_idxs=lig_idxs,
                                   binding_site_idxs=binding_selection_idxs,
                                   # algorithm parameters
                                   pmax=0.1)

    # makes ref_traj and selects lingand_atom and protein atom  indices
    # instantiate a wexplore2 unbindingboudaryconditiobs
    ubc = UnbindingBC(cutoff_distance=1.0,
                      initial_state=init_walkers[0],
                      topology=pdb.topology,
                      ligand_idxs=lig_idxs,
                      binding_site_idxs=protein_idxs)


    # make a dictionary of units for adding to the HDF5
    units = {}
    for key, value in dict(UNITS).items():
        try:
            unit_name = value.get_name()
        except AttributeError:
            print("not a unit")
            unit_name = False

        if unit_name:
            units[key] = unit_name

    # instantiate a reporter for HDF5
    report_path = 'wepy_results.h5'
    reporter = WepyHDF5Reporter(report_path, mode='w',
                                decisions=resampler.DECISION,
                                instruction_dtypes=resampler.INSTRUCTION_DTYPES,
                                resampling_aux_dtypes=None,
                                resampling_aux_shapes=None,
                                warp_dtype=ubc.WARP_INSTRUCT_DTYPE,
                                warp_aux_dtypes=ubc.WARP_AUX_DTYPES,
                                warp_aux_shapes=ubc.WARP_AUX_SHAPES,
                                topology=sEH_TPPU_system_top_json,
                                units=units)


    # create a work mapper for NVIDIA GPUs for a GPU cluster
    num_workers = 2
    gpumapper  = GPUMapper(num_walkers, n_workers=num_workers)

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=ubc,
                          work_mapper=gpumapper.map,
                          reporter=reporter)
    n_steps = 100
    n_cycles = 2

    # run a simulation with the manager for n_steps cycles of length 1000 each
    steps = [ n_steps for i in range(n_cycles)]
    print("Running simulation")
    sim_manager.run_simulation(n_cycles,
                               steps,
                               debug_prints=True)

    # your data should be in the 'wepy_results.h5'
