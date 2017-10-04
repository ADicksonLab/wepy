"""This script was used to produce the inputs that would be used in
the examples and shows an example of how you might go from a crystal
structure of a protein ligand complex to a set of inputs for
wepy. This is by no means the only way to do this. Furthermore, a few
of the things written out will be deprecated in the future as
necessary, or are only used for convenience here"""

import os
import pickle

import numpy as np
import h5py

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

from wepy.openmm import OpenMMWalker
from wepy.hdf5 import TrajHDF5

if __name__ == "__main__":
    psf = omma.CharmmPsfFile('sEH_TPPU_system.psf')

    # load the coordinates
    pdb = mdj.load_pdb('sEH_TPPU_system.pdb')

    # write out an hdf5 storage of the system
    pdb.save_hdf5('tmp_mdtraj_system.h5')

    # we need a JSON string for now in the topology section of the
    # HDF5 so we just load the topology from the hdf5 file
    top_h5 = h5py.File("tmp_mdtraj_system.h5")
    # it is in bytes so we need to decode to a string, which is in JSON format
    top_str = top_h5['topology'][0].decode()
    top_h5.close()

    os.remove("tmp_mdtraj_system.h5")

    # write the JSON topology out
    with open("sEH_TPPU_system.top.json", mode='w') as json_wf:
        json_wf.write(top_str)

    # to use charmm forcefields get your parameters
    params = omma.CharmmParameterSet('all36_cgenff.rtf',
                                     'all36_cgenff.prm',
                                     'all36_prot.rtf',
                                     'all36_prot.prm',
                                     'tppu.str',
                                     'toppar_water_ions.str')

    # set the box size lengths and angles
    psf.setBox(8.2435*unit.nanometer, 8.2435*unit.nanometer, 8.2435*unit.nanometer, 90*unit.degree, 90*unit.degree, 90*unit.degree)

    # create a system using the topology method giving it a topology and
    # the method for calculation
    system = psf.createSystem(params,
                                  nonbondedMethod=omma.CutoffPeriodic,
                                   nonbondedCutoff=1.0 * unit.nanometer,
                                  constraints=omma.HBonds)
    # we want to have constant pressure
    barostat = omm.MonteCarloBarostat(1.0*unit.atmosphere, 300.0*unit.kelvin, 50)

    topology = psf.topology

    print("\nminimizing\n")
    # set up for a short simulation to minimize and prepare
    # instantiate an integrator
    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                            1/unit.picosecond,
                                            0.002*unit.picoseconds)
    platform = omm.Platform.getPlatformByName('CUDA')
#    platform = omm.Platform.getPlatformByName('Reference')

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
                                                  getParameters=True,
                                                  getForces=True,
                                                  getEnergy=True,
                                                  getParameterDerivatives=True)

    # pickle it for use in seeding simulations
    with open("initial_openmm_state.pkl", mode='wb') as wf:
        pickle.dump(minimized_state, wf)

    print ('finished initialization')
