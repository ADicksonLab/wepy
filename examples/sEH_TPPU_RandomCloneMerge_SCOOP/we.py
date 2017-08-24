import sys
from collections import namedtuple

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

from wepy.resampling.clone_merge import clone_parent_table, clone_parent_panel


from resampling_tree.tree import monte_carlo_minimization, make_graph


if __name__ == "__main__":

    #### SETUP: skip this for understanding -----------------------------------------

    # write out an hdf5 storage of the system
    mdj_pdb = mdj.load_pdb('sEH_TPPU_system.pdb')
    mdj_pdb.save_hdf5('sEH_TPPU_system.h5')

    # we need a JSON string for now in the topology section of the
    # HDF5 so we just load the topology from the hdf5 file
    top_h5 = h5py.File("sEH_TPPU_system.h5")
    # it is in bytes so we need to decode to a string, which is in JSON format
    top_str = top_h5['topology'][0].decode()
    top_h5.close()

    # set up the system which will be passed to the arguments of the mapped functions
    # the topology from the PSF
    psf = omma.CharmmPsfFile('sEH_TPPU_system.psf')

    # load the coordinates
    omm_pdb = omma.PDBFile('sEH_TPPU_system.pdb')

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

    print("\nminimizing\n")
    # set up for a short simulation to minimize and prepare
    # instantiate an integrator
    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                        1/unit.picosecond,
                                        0.002*unit.picoseconds)
    # instantiate a simulation object
    simulation = omma.Simulation(psf.topology, system, integrator)
    # initialize the positions
    simulation.context.setPositions(omm_pdb.positions)
    # minimize the energy
    simulation.minimizeEnergy()
    # run the simulation for a number of initial time steps
    simulation.step(1000)
    print("done minimizing\n")

    # get the initial state from the context
    minimized_state = simulation.context.getState(getPositions=True,
                                                  getVelocities=True,
                                                  getParameters=True)


    #### END SETUP -----------------------------------------------------------------

    # set up parameters for running the simulation
    num_walkers = 8
    # initial weights, split equally between the walkers
    init_weight = 1.0 / num_walkers
    # make the initial walkers
    init_walkers = [OpenMMWalker(minimized_state, init_weight) for i in range(num_walkers)]

    # set up the OpenMMRunner with your system
    runner = OpenMMRunner(system, psf.topology)

    # set up the RandomResampler with the same random seed
    resampler = RandomCloneMergeResampler()

    # make a reporter for recording an HDF5 file for the simulation
    report_path = 'wepy_results.h5'
    reporter = WepyHDF5Reporter(report_path,
                                resampler.DECISION,
                                resampler.INSTRUCTION_DTYPES,
                                top_str, mode='w')

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          # the mapper may be swapped out for simply `map`
                          work_mapper=scoop.futures.map,
                          reporter=reporter)


    # run a simulation with the manager for 3 cycles of length 1000 each
    print("Running simulation")
    sim_manager.run_simulation(3, [1000, 1000, 1000])

    # your data should be in the 'wepy_results.h5'

