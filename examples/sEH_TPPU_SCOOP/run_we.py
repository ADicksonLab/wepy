from sys import stdout

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from scoop import futures

from wepy.sim_manager import SimManager

### The function for a single segment of simulation
def run_simulation(topology, system, positions, worker_id):
    print("starting a simulation")
    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                        1/unit.picosecond,
                                        0.002*unit.picoseconds)
    # instantiate a simulation object
    simulation = omma.Simulation(topology, system, integrator)
    # initialize the positions
    simulation.context.setPositions(positions)
    # Reporter
    simulation.reporters.append(omma.StateDataReporter("worker_{}.log".format(worker_id),
                                                       100,
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True))

    # run the simulation fo the number of time steps
    simulation.step(1000)
    print("finished with simulation")

    return simulation.context.getState(getPositions=True)




if __name__ == "__main__":



    ### The setup for all workers

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

    # instantiate an integrator with the desired properties
    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                        1/unit.picosecond,
                                        0.002*unit.picoseconds)

    # instantiate a simulation object
    simulation = omma.Simulation(psf.topology, system, integrator)

    # initialize the positions
    simulation.context.setPositions(pdb.positions)

    print("\nminimizing\n")
    # minimize the energy
    simulation.minimizeEnergy()

    # Reporter
    simulation.reporters.append(omma.StateDataReporter(stdout, 100,
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True))
    # file for the minimized structure
    simulation.reporters.append(omma.PDBReporter('minimized.pdb', 1000))

    # run the simulation for a number of initial time steps
    simulation.step(1000)

    minimized_state = simulation.context.getState(getPositions=True)
    minimized_positions = minimized_state.getPositions()

    print("done minimizing\n")

    # start a sim_manager
    num_workers = 8
    num_walkers = 8
    sim_manager = SimManager(psf.topology, system,
                             minimized_positions, num_workers, num_walkers)
    # compare the results

