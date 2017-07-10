import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from scoop import futures

from wepy.decision import NoCloneMerge, DecisionModel

def run_segment(topology, system, positions, worker_id, segment_length):
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
    simulation.step(segment_length)
    print("finished with simulation")

    return simulation.context.getState(getPositions=True)


class Manager(object):

    def __init__(self, topology, system, positions,
                 num_walkers, num_workers,
                 decision_model=NoCloneMerge,
                 work_mapper=futures.map):

        self.topology = topology
        self.system = system
        self.positions = positions
        self.num_walkers = num_walkers
        self.num_workers = num_workers
        assert issubclass(decision_model, DecisionModel), \
            "decision_model is not a subclass of DecisionModel"
        self.decision_model = decision_model
        self.map = work_mapper

    def run_segment(self, segment_length):
        """Run a time segment for all walkers using the available workers. """

        # create generators to map inputs to the function
        # topology and system are the same for all
        topologies_gen = (self.topology for i in range(self.num_workers))
        systems_gen = (self.system for i in range(self.num_workers))

        # the positions are different for each walker and tracked in the attribute
        positions_gen = (self.positions for i in range(self.num_workers))

        # the worker ids
        worker_ids_gen = (i for i in range(self.num_workers))

        # the segment length is also the same for a single segment but
        # not between segments
        segment_length_gen = (segment_length for i in range(self.num_workers))

        results = list(self.map(run_segment,
                                topologies_gen,
                                systems_gen,
                                positions_gen,
                                worker_ids_gen,
                                segment_length_gen))

        return results

    def compute_decisions(self, walkers):
        return self.decision_model.decide(walkers)

    def resample(self, decisions):
        raise NotImplementedError

    def write_results(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError

    def run_simulation(self, n_cycles, segment_lengths, output="memory"):
        """Run a simulation for a given number of cycles with specified
        lengths of MD segments in between.

        Can either return results in memory or write to a file.
        """

        for cycle_idx in range(n_cycles):
            results = self.run_segment(segment_lengths[cycle_idx])
            decisions = self.compute_decisions(results)
            self.resample(decisions)
            if output == "memory":
                return self.report()
            else output == "hdf5":
                self.write_results()
