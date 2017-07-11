import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from wepy.walker import Walker
from wepy.runner import Runner

class OpenMMRunner(Runner):

    def __init__(self, system, topology):
        self.system = system
        self.topology = topology

    def run_segment(self, walker, segment_length):

        # TODO can we do this outside of this?
        # instantiate an integrator
        integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                            1/unit.picosecond,
                                            0.002*unit.picoseconds)

        # instantiate a simulation object
        simulation = omma.Simulation(self.topology, self.system, integrator)

        # initialize the positions
        simulation.context.setPositions(walker.positions)

        # run the simulation segment for the number of time steps
        simulation.step(segment_length)

        # save the state of the system with all possible values
        new_state = simulation.context.getState(getPositions=True,
                                            getVelocities=True,
                                            getParameters=True,
                                            getParameterDerivatives=True,
                                            getForces=True,
                                            getEnergy=True,
                                            enforcePeriodicBox=True
                                            )

        # create a new walker for this
        new_walker = OpenMMWalker(new_state, walker.weight)

        return new_walker


class OpenMMWalker(Walker):

    def __init__(self, state, weight):
        super().__init__(state, weight)

    @property
    def positions(self):
        return self.state.getPositions()

    @property
    def velocities(self):
        return self.state.getVelocities()

    @property
    def forces(self):
        return self.state.getForces()

    @property
    def kinetic_energy(self):
        return self.state.getKineticEnergy()

    @property
    def potential_energy(self):
        return self.state.getPotentialEnergy()

    @property
    def time(self):
        return self.state.getTime()

    @property
    def box_vectors(self):
        return self.state.getPeriodicBoxVectors()

    @property
    def box_volume(self):
        return self.state.getPeriodicBoxVolume()

    @property
    def parameters(self):
        return self.state.getParameters()

    @property
    def parameter_derivatives(self):
        return self.state.getEnergyParameterDerivatives()
