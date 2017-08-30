import sys
from collections import namedtuple

import numpy as np
import pandas as pd
import h5py

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

from wepy.sim_manager import Manager
from wepy.resampling.clone_merge import CloneMergeDecision, CLONE_MERGE_INSTRUCTION_DTYPES
from wepy.resampling.clone_merge import RandomCloneMergeResampler
from wepy.openmm import OpenMMRunner, OpenMMWalker
from wepy.reporter.hdf5 import WepyHDF5Reporter

from wepy.resampling.clone_merge import clone_parent_table, clone_parent_panel


if __name__ == "__main__":

    # write out an hdf5 storage of the system
    mdj_pdb = mdj.load_pdb('sEH_TPPU_system.pdb')
    mdj_pdb.save_hdf5('sEH_TPPU_system.h5')

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

    # set up parameters for running the simulation
    num_walkers = 3
    # initial weights
    init_weight = 1.0 / num_walkers
    # make a generator for the initial walkers
    init_walkers = [OpenMMWalker(minimized_state, init_weight) for i in range(num_walkers)]

    # make a template string for pretty printing results as we go
    result_template_str = "|".join(["{:^10}" for i in range(num_walkers + 1)])

    # print the initial walkers
    print("The initial walkers:")
    # slots
    slot_str = result_template_str.format("slot", *[i for i in range(num_walkers)])
    print(slot_str)
    # weights
    walker_weight_str = result_template_str.format("weight",
        *[str(init_weight) for i in range(num_walkers)])
    print(walker_weight_str)

    # set up the OpenMMRunner with your system
    runner = OpenMMRunner(system, psf.topology)


    # set up the RandomResampler with the same random seed
    seed = 3247862378
    resampler = RandomCloneMergeResampler(3247862378)

    # we need a JSON string for now in the topology section of the
    # HDF5 so we just load the topology from the hdf5 file
    top_h5 = h5py.File("sEH_TPPU_system.h5")
    # it is in bytes so we need to decode to a string, which is in JSON format
    top_str = top_h5['topology'][0].decode()
    top_h5.close()

    # we also need to give the reporter the decision types and the
    # instruction datatypes for the resampler we are using.
    decision = resampler.DECISION
    instruction_dtypes = resampler.INSTRUCTION_DTYPES

    # make a reporter for recording an HDF5 file for the simulation
    report_path = 'wepy_results.h5'
    reporter = WepyHDF5Reporter(report_path, decision, instruction_dtypes, top_str, mode='w')


    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=map,
                          reporter=reporter)


    # run a simulation with the manager for 3 cycles of length 1000 each
    sim_manager.run_simulation(2, [1000, 2000])
