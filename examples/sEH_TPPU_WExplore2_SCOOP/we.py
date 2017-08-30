import h5py
import numpy as np
import pandas as pd

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import scoop.futures

import mdtraj as mdj

from wepy.sim_manager import Manager
from wepy.resampling.wexplore2 import WExplore2Resampler
from wepy.openmm import OpenMMRunner, OpenMMWalker
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.reporter.hdf5 import WepyHDF5Reporter

if __name__ == "__main__":

    psf = omma.CharmmPsfFile('sEH_TPPU_system.psf')

    # load the coordinates
    pdb = mdj.load_pdb('sEH_TPPU_system.pdb')

    # write out an hdf5 storage of the system
    pdb.save_hdf5('sEH_TPPU_system.h5')
    # we need a JSON string for now in the topology section of the
    # HDF5 so we just load the topology from the hdf5 file
    top_h5 = h5py.File("sEH_TPPU_system.h5")
    # it is in bytes so we need to decode to a string, which is in JSON format
    top_str = top_h5['topology'][0].decode()
    top_h5.close()

    # to use charmm forcefields get your parameters
    params = omma.CharmmParameterSet('all36_cgenff.rtf',
                                         'all36_cgenff.prm',
                                         'all36_prot.rtf',
                                         'all36_prot.prm',
                                         'tppu.str',
                                         'toppar_water_ions.str')

    # set the box size lengths and angles
    psf.setBox(82.435, 82.435, 82.435, 90, 90, 90)

    # create a system using the topology method giving it a topology and
    # the method for calculation
    system = psf.createSystem(params,
                                  nonbondedMethod=omma.CutoffPeriodic,
                                   nonbondedCutoff=1.0 * unit.nanometer,
                                  constraints=omma.HBonds)
    topology = psf.topology

    print("\nminimizing\n")
    # set up for a short simulation to minimize and prepare
    # instantiate an integrator
    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                            1/unit.picosecond,
                                            0.002*unit.picoseconds)
    platform = omm.Platform.getPlatformByName('CPU')

    # instantiate a simulation object
    simulation = omma.Simulation(psf.topology, system, integrator, platform)
    # initialize the positions
    simulation.context.setPositions(pdb.openmm_positions(frame=0))
    # minimize the energy
    simulation.minimizeEnergy()
    # run the simulation for a number of initial time steps
    simulation.step(1000)
    print("done minimizing\n")

        # get the initial state from the context
    minimized_state = simulation.context.getState(getPositions=True,
                                                  getVelocities=True,

                                                  getParameters=True)
    print ('finished initialization')

    # set up parameters for running the simulation
    num_walkers = 4
    # initial weights
    init_weight = 1.0 / num_walkers

    # a list of the initial walkers
    init_walkers = [OpenMMWalker(minimized_state, init_weight) for i in range(num_walkers)]

    # set up the OpenMMRunner with your system
    runner = OpenMMRunner(system, topology)

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

    # set up the WExplore2 Resampler with the parameters
    resampler = WExplore2Resampler(topology=pdb.topology,
                                   ligand_idxs=lig_idxs,
                                   binding_site_idxs=binding_selection_idxs,
                                   # algorithm parameters
                                   pmax=0.1)

    # makes ref_traj and selects lingand_atom and protein atom  indices
    # instantiate a wexplore2unbindingboudaryconditiobs
    ubc = UnbindingBC(cutoff_distance=1.0,
                      initial_state=init_walkers[0],
                      topology=pdb.topology,
                      ligand_idxs=lig_idxs,
                      binding_site_idxs=protein_idxs)

    # instantiate a reporter for HDF5
    report_path = 'wepy_results.h5'
    reporter = WepyHDF5Reporter(report_path,
                                resampler.DECISION,
                                resampler.INSTRUCTION_DTYPES,
                                top_str, mode='w')

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          num_workers,
                          runner=runner,
                          resampler=resampler,
                          boundary_condition = ubc,
                          work_mapper=scoop.futures.map)
    n_steps = 100
    n_cycles = 2

    # run a simulation with the manager for n_steps cycles of length 1000 each
    steps = [ n_steps for i in range(n_cycles)]
    sim_manager.run_simulation(n_cycles,
                               steps,
                               debug_prints=True)
