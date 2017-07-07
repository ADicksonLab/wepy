import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
from sys import stdout
from scoop import futures


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
    # load the input files
    # the topology from PDB
    pdb = omma.PDBFile('input.pdb')
    print(pdb.topology)
    topology = pdb.topology

    # the forcefields from the OpenMM XML format
    forcefield = omma.ForceField('amber99sb.xml', 'tip3p.xml')

    # create a system using the topology method giving it a topology and
    # the method for calculation
    system = forcefield.createSystem(topology,
                                     nonbondedMethod=omma.PME,
                                     nonbondedCutoff=1*unit.nanometer,
                                     constraints=omma.HBonds)

    # instantiate an integrator with the desired properties
    integrator = omm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)

    # instantiate a simulation object
    simulation = omma.Simulation(topology, system, integrator)

    # initialize the positions
    simulation.context.setPositions(pdb.positions)

    print("minimizing")
    # minimize the energy
    simulation.minimizeEnergy()

    # Reporter
    simulation.reporters.append(omma.StateDataReporter(stdout, 100,
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True))
    # file for the minimized structure
    simulation.reporters.append(omma.PDBReporter('minimized.pdb', 1000))

    # run the simulation fo the number of time steps
    simulation.step(1000)

    minimized_state = simulation.context.getState(getPositions=True)
    minimized_positions = minimized_state.getPositions()

    print("done minimizing")

    ### Run separate simulations from the minimized structure

    num_workers = 8
    topologies = [topology for i in range(num_workers)]
    systems = [system for i in range(num_workers)]
    positions = [minimized_positions for i in range(num_workers)]
    worker_ids = [i for i in range(num_workers)]

    print("running simulations")
    results = list(futures.map(run_simulation, topologies, systems, positions, worker_ids))
