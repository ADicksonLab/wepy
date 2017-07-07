import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from scoop import futures

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


class SimManager(object):

    def __init__(self, topology, system, positions, num_walkers, num_workers):
        self.topology = topology
        self.system = system
        self.positions = positions
        self.num_walkers = num_walkers
        self.num_workers = num_workers

    def run_cycle(self):
        """Run a time segment for all walkers using the available workers. """
        topologies_gen = (self.topology for i in range(self.num_workers))
        systems_gen = (self.system for i in range(self.num_workers))
        positions_gen = (self.positions for i in range(self.num_workers))
        worker_ids = [i for i in range(self.num_workers)]

        results = list(futures.map(run_simulation,
                                   topologies_gen,
                                   systems_gen,
                                   positions_gen,
                                   worker_ids))

        return results

