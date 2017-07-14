import sys

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

import scoop.futures

from wepy.sim_manager import Manager
from wepy.resampling.wexplore2  import WExplore2Resampler
from wepy.openmm import OpenMMRunner, OpenMMWalker

if __name__ == "__main__":

    # set up the system which will be passed to the arguments of the mapped functions
    # the topology from the PSF
    psf = omma.CharmmPsfFile('sEH_TPPU_system.psf')

    # load the coordinates
    pdb = omma.PDBFile('sEH_TPPU_system.pdb')

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
    simulation.context.setPositions(pdb.positions)
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
    num_workers = 8
    num_walkers = 8
    # initial weights
    init_weight = 1.0 / num_walkers
    # make a generator for the initial walkers
    init_walkers = [OpenMMWalker(minimized_state, init_weight) for i in range(num_walkers)]

    # set up the OpenMMRunner with your system
    runner = OpenMMRunner(system, psf.topology)

    # set up the Resampler (with parameters if needed)
    ref_traj = mdj.load('sEH_TPPU_system.pdb')
    resampler = WExplore2Resampler(refrence_trajectory=ref_traj)

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          num_workers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=map)

    # run a simulation with the manager for 3 cycles of length 1000 each
    walker_records, resampling_records = sim_manager.run_simulation(3,
                                                                    [1000, 1000, 1000],
                                                                    debug_prints=True)
