import numpy as np

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
    def positions_unit(self):
        return self.positions.unit

    def positions_values(self):
        return np.array(self.positions.value_in_unit(self.positions_unit))

    @property
    def velocities(self):
        return self.state.getVelocities()

    @property
    def velocities_unit(self):
        return self.velocities.unit

    def velocities_values(self):
        return np.array(self.velocities.value_in_unit(self.velocities_unit))

    @property
    def forces(self):
        return self.state.getForces()

    @property
    def forces_unit(self):
        return self.forces.unit

    def forces_values(self):
        return np.array(self.forces.value_in_unit(self.forces_unit))

    @property
    def kinetic_energy(self):
        return self.state.getKineticEnergy()

    @property
    def kinetic_energy_unit(self):
        return self.kinetic_energy.unit

    def kinetic_energy_value(self):
        return self.kinetic_energy.value_in_unit(self.kinetic_energy_unit)

    @property
    def potential_energy(self):
        return self.state.getPotentialEnergy()

    @property
    def potential_energy_unit(self):
        return self.potential_energy.unit

    def potential_energy_value(self):
        return self.potential_energy.value_in_unit(self.potential_energy_unit)

    @property
    def time(self):
        return self.state.getTime()

    @property
    def time_unit(self):
        return self.time.unit

    def time_value(self):
        return self.time.value_in_unit(self.time_unit)

    @property
    def box_vectors(self):
        return self.state.getPeriodicBoxVectors()

    @property
    def box_vectors_unit(self):
        return self.box_vectors.unit

    def box_vectors_values(self):
        return np.array(self.box_vectors.value_in_unit(self.box_vectors_unit))

    @property
    def box_volume(self):
        return self.state.getPeriodicBoxVolume()

    @property
    def box_volume_unit(self):
        return self.box_volume.unit

    def box_volume_value(self):
        return self.box_volume.value_in_unit(self.box_volume_unit)

    @property
    def parameters(self):
        return self.state.getParameters()

    # TODO test this, this is jsut a guess because I don't use parameters
    @property
    def parameters_unit(self):
        param_units = {key : val.unit for key, val in self.parameters.items()}
        return param_units

    # TODO test this, this is jsut a guess because I don't use parameters
    def parameters_values(self):
        param_arrs = {key : np.array(val.value_in_unit(val.unit)) for key, val
                          in self.parameters.items()}
        return param_arrs

    @property
    def parameter_derivatives(self):
        return self.state.getEnergyParameterDerivatives()

    # TODO test this, this is jsut a guess because I don't use parameters
    @property
    def parameter_derivatives_unit(self):
        param_units = {key : val.unit for key, val in self.parameter_derivatives.items()}
        return param_units

    # TODO test this, this is jsut a guess because I don't use parameter_derivatives
    def parameter_derivatives_values(self):
        param_arrs = {key : np.array(val.value_in_unit(val.unit)) for key, val
                          in self.parameter_derivatives.items()}
        return param_arrs


    def to_mdtraj(self):
        """ Returns an mdtraj.Trajectory object from this walker's state."""
        raise NotImplementedError
        import mdtraj as mdj
        return mdj.Trajectory(self.positions_values,
                              time=self.time_value, unitcell_vectors=self.box_vectors)
