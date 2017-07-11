import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from wepy.walker import Walker
from wepy.runner import Runner

class OpenMMRunner(Runner):

    def __init__(self, system):
        self.system = system
        self.topology = self.system.topology

    def run_segment(self, init_walkers, segment_length):

        # TODO can we do this outside of this?
        # instantiate an integrator
        integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                            1/unit.picosecond,
                                            0.002*unit.picoseconds)

        # instantiate a simulation object
        simulation = omma.Simulation(self.topology, self.system, integrator)

        # initialize the positions
        simulation.context.setPositions(init_walkers.coordinates)

        # run the simulation segment for the number of time steps
        simulation.step(segment_length)
        print("finished with simulation")

        return simulation.context.getState(getPositions=True)


class TrajWalker(Walker):

    def __init__(self, coordinates, weight, parent):
        super().__init__(coordinates, weight, parent)
