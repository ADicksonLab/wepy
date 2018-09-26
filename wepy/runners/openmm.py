from copy import copy
import random as rand
from warnings import warn
import logging

import numpy as np

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from wepy.walker import Walker, WalkerState
from wepy.runners.runner import Runner
from wepy.work_mapper.worker import Worker
from wepy.reporter.reporter import Reporter


## Constants

KEYS = ('positions', 'velocities', 'forces', 'kinetic_energy',
        'potential_energy', 'time', 'box_vectors', 'box_volume',
        'parameters', 'parameter_derivatives')

# when we use the get_state function from the simulation context we
# can pass options for what kind of data to get, this is the default
# to get all the data. TODO not really sure what the 'groups' keyword
# is for though
GET_STATE_KWARG_DEFAULTS = (('getPositions', True),
                            ('getVelocities', True),
                            ('getForces', True),
                            ('getEnergy', True),
                            ('getParameters', True),
                            ('getParameterDerivatives', True),
                            ('enforcePeriodicBox', True),)
                            # TODO unsure of how to use this kwarg
                            #('groups') )


# the Units objects that OpenMM uses internally and are returned from
# simulation data
UNITS = (('positions_unit', unit.nanometer),
         ('time_unit', unit.picosecond),
         ('box_vectors_unit', unit.nanometer),
         ('velocities_unit', unit.nanometer/unit.picosecond),
         ('forces_unit', unit.kilojoule / (unit.nanometer * unit.mole)),
         ('box_volume_unit', unit.nanometer),
         ('kinetic_energy_unit', unit.kilojoule / unit.mole),
         ('potential_energy_unit', unit.kilojoule / unit.mole),
        )

# the names of the units from the units objects above. This is used
# for saving them to files
UNIT_NAMES = (('positions_unit', unit.nanometer.get_name()),
         ('time_unit', unit.picosecond.get_name()),
         ('box_vectors_unit', unit.nanometer.get_name()),
         ('velocities_unit', (unit.nanometer/unit.picosecond).get_name()),
         ('forces_unit', (unit.kilojoule / (unit.nanometer * unit.mole)).get_name()),
         ('box_volume_unit', unit.nanometer.get_name()),
         ('kinetic_energy_unit', (unit.kilojoule / unit.mole).get_name()),
         ('potential_energy_unit', (unit.kilojoule / unit.mole).get_name()),
        )

# a random seed will be chosen from 1 to RAND_SEED_RANGE_MAX when the
# Langevin integrator is created. 0 is the default and special value
# which will then choose a random value when the integrator is created
RAND_SEED_RANGE_MAX = 1000000

# the runner for the simulation which runs the actual dynamics
class OpenMMRunner(Runner):

    def __init__(self, system, topology, integrator, platform=None):

        # we save the different components. However, if we are to make
        # this runner picklable we have to convert the SWIG objects to
        # a picklable form
        self.system = system
        self.integrator = integrator

        # these are not SWIG objects
        self.topology = topology
        self.platform_name = platform

    def _openmm_swig_objects(self):
        """Just returns all of the foreign OpenMM module objects this class
        uses that are actually SWIG wrappers."""

        return (self.system, self.integrator)

    def run_segment(self, walker, segment_length, getState_kwargs=None, **kwargs):

        # set the kwargs that will be passed to getState
        tmp_getState_kwargs = getState_kwargs
        getState_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
        if tmp_getState_kwargs is not None:
            getState_kwargs.update(tmp_getState_kwargs)


        # make a copy of the integrator for this particular segment
        new_integrator = copy(self.integrator)
        # force setting of random seed to 0, which is a special
        # value that forces the integrator to choose another
        # random number
        new_integrator.setRandomNumberSeed(0)

        # if a platform was given we use it to make a Simulation object
        if self.platform_name is not None:
            # get the platform by its name to use
            platform = omm.Platform.getPlatformByName(self.platform_name)
            # set properties from the kwargs if they apply to the platform
            for key, value in kwargs.items():
                if key in platform.getPropertyNames():
                    platform.setPropertyDefaultValue(key, value)

            # make a new simulation object
            simulation = omma.Simulation(self.topology, self.system,
                                         new_integrator, platform)

        # otherwise just use the default or environmentally defined one
        else:
            simulation = omma.Simulation(self.topology, self.system,
                                         new_integrator)

        # set the state to the context from the walker
        simulation.context.setState(walker.state.sim_state)

        # Run the simulation segment for the number of time steps
        simulation.step(segment_length)

        # save the state of the system with all possible values
        new_sim_state = simulation.context.getState(**getState_kwargs)

        # make an OpenMMState wrapper with this
        new_state = OpenMMState(new_sim_state)

        # create a new walker for this
        new_walker = OpenMMWalker(new_state, walker.weight)

        return new_walker


class OpenMMState(WalkerState):

    KEYS = KEYS

    OTHER_KEY_TEMPLATE = "{}_OTHER"

    def __init__(self, sim_state, **kwargs):

        # save the simulation state
        self._sim_state = sim_state

        # save additional data if given
        self._data = {}
        for key, value in kwargs.items():

            # if the key is already in the sim_state keys we need to
            # modify it and raise a warning
            if key in self.KEYS:

                warn("Key {} in kwargs is already taken by this class, renaming to {}".format(
                    self.OTHER_KEY_TEMPLATE).format(key))

                # make a new key
                new_key = self.OTHER_KEY_TEMPLATE.format(key)

                # set it in the data
                self._data[new_key] = value

            # otherwise just set it
            else:
                self._data[key] = value

    @property
    def sim_state(self):
        return self._sim_state

    def __getitem__(self, key):

        # if this was a key for data not mapped from the OpenMM.State
        # object we use the _data attribute
        if key not in self.KEYS:
            return self._data[key]

        # otherwise we have to specifically get the correct data and
        # process it into an array from the OpenMM.State
        else:

            if key == 'positions':
                return self.positions_values()
            elif key == 'velocities':
                return self.velocities_values()
            elif key == 'forces':
                return self.forces_values()
            elif key == 'kinetic_energy':
                return self.kinetic_energy_value()
            elif key == 'potential_energy':
                return self.potential_energy_value()
            elif key == 'time':
                return self.time_value()
            elif key == 'box_vectors':
                return self.box_vectors_values()
            elif key == 'box_volume':
                return self.box_volume_value()
            elif key == 'parameters':
                return self.parameters_values()
            elif key == 'parameter_derivatives':
                return self.parameter_derivatives_values()

    ## Array properties

    # Positions
    @property
    def positions(self):
        try:
            return self.sim_state.getPositions(asNumpy=True)
        except:
            warn("Unknown exception handled from `self.sim_state.getPositions()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def positions_unit(self):
        return self.positions.unit

    def positions_values(self):
        return self.positions.value_in_unit(self.positions_unit)

    # Velocities
    @property
    def velocities(self):
        try:
            return self.sim_state.getVelocities(asNumpy=True)
        except:
            warn("Unknown exception handled from `self.sim_state.getVelocities()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def velocities_unit(self):
        return self.velocities.unit

    def velocities_values(self):
        velocities = self.velocities
        if velocities is None:
            return None
        else:
            return self.velocities.value_in_unit(self.velocities_unit)

    # Forces
    @property
    def forces(self):
        try:
            return self.sim_state.getForces(asNumpy=True)
        except:
            warn("Unknown exception handled from `self.sim_state.getForces()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def forces_unit(self):
        return self.forces.unit

    def forces_values(self):
        forces = self.forces
        if forces is None:
            return None
        else:
            return self.forces.value_in_unit(self.forces_unit)

    # Box Vectors
    @property
    def box_vectors(self):
        try:
            return self.sim_state.getPeriodicBoxVectors(asNumpy=True)
        except:
            warn("Unknown exception handled from `self.sim_state.getPeriodicBoxVectors()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def box_vectors_unit(self):
        return self.box_vectors.unit

    def box_vectors_values(self):
        box_vectors = self.box_vectors
        if box_vectors is None:
            return None
        else:
            return self.box_vectors.value_in_unit(self.box_vectors_unit)


    ## non-array properties

    # Kinetic Energy
    @property
    def kinetic_energy(self):
        try:
            return self.sim_state.getKineticEnergy()
        except:
            warn("Unknown exception handled from `self.sim_state.getKineticEnergy()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def kinetic_energy_unit(self):
        return self.kinetic_energy.unit

    def kinetic_energy_value(self):
        kinetic_energy = self.kinetic_energy
        if kinetic_energy is None:
            return None
        else:
            return np.array([self.kinetic_energy.value_in_unit(self.kinetic_energy_unit)])

    # Potential Energy
    @property
    def potential_energy(self):
        try:
            return self.sim_state.getPotentialEnergy()
        except:
            warn("Unknown exception handled from `self.sim_state.getPotentialEnergy()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def potential_energy_unit(self):
        return self.potential_energy.unit

    def potential_energy_value(self):
        potential_energy = self.potential_energy
        if potential_energy is None:
            return None
        else:
            return np.array([self.potential_energy.value_in_unit(self.potential_energy_unit)])

    # Time
    @property
    def time(self):
        try:
            return self.sim_state.getTime()
        except:
            warn("Unknown exception handled from `self.sim_state.getTime()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def time_unit(self):
        return self.time.unit

    def time_value(self):
        time = self.time
        if time is None:
            return None
        else:
            return np.array([self.time.value_in_unit(self.time_unit)])

    # Box Volume
    @property
    def box_volume(self):
        try:
            return self.sim_state.getPeriodicBoxVolume()
        except:
            warn("Unknown exception handled from `self.sim_state.getPeriodicBoxVolume()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def box_volume_unit(self):
        return self.box_volume.unit

    def box_volume_value(self):
        box_volume = self.box_volume
        if box_volume is None:
            return None
        else:
            return np.array([self.box_volume.value_in_unit(self.box_volume_unit)])

    ## Dictionary properties
    ## Unitless

    # Parameters
    @property
    def parameters(self):
        try:
            return self.sim_state.getParameters()
        except:
            warn("Unknown exception handled from `self.sim_state.getParameters()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def parameters_unit(self):
        param_units = {key : None for key, val in self.parameters.items()}
        return param_units

    def parameters_values(self):
        if self.parameters is None:
            return None

        param_arrs = {key : np.array(val) for key, val
                          in self.parameters.items()}

        # return None if there is nothing in this
        if len(param_arrs) == 0:
            return None
        else:
            return param_arrs

    # Parameter Derivatives
    @property
    def parameter_derivatives(self):
        try:
            return self.sim_state.getEnergyParameterDerivatives()
        except:
            warn("Unknown exception handled from `self.sim_state.getEnergyParameterDerivatives()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def parameter_derivatives_unit(self):
        param_units = {key : None for key, val in self.parameter_derivatives.items()}
        return param_units

    def parameter_derivatives_values(self):

        if self.parameter_derivatives is None:
            return None

        param_arrs = {key : np.array(val) for key, val
                          in self.parameter_derivatives.items()}

        # return None if there is nothing in this
        if len(param_arrs) == 0:
            return None
        else:
            return param_arrs

    def omm_state_dict(self):
        """return a dict of the values for the keys that are hardcoded in this class."""
        return {'positions' : self.positions_values(),
                'velocities' : self.velocities_values(),
                'forces' : self.forces_values(),
                'kinetic_energy' : self.kinetic_energy_value(),
                'potential_energy' : self.potential_energy_value(),
                'time' : self.time_value(),
                'box_vectors' : self.box_vectors_values(),
                'box_volume' : self.box_volume_value(),
                'parameters' : self.parameters_values(),
                'parameter_derivatives' : self.parameter_derivatives_values()
                    }

    def dict(self):
        """Return a dict of the values for all attributes of this state."""

        d = {}
        for key, value in self._data.items():
            d[key] = value
        for key, value in self.omm_state_dict().items():
            d[key] = value
        return d

    def to_mdtraj(self):
        """ Returns an mdtraj.Trajectory object from this walker's state."""
        raise NotImplementedError
        import mdtraj as mdj
        # resize the time to a 1D vector
        return mdj.Trajectory(self.positions_values,
                              time=self.time_value[:,0], unitcell_vectors=self.box_vectors_values)

class OpenMMWalker(Walker):

    def __init__(self, state, weight):

        assert isinstance(state, OpenMMState), \
            "state must be an instance of class OpenMMState not {}".format(type(state))

        super().__init__(state, weight)

class OpenMMGPUWorker(Worker):

    def run_task(self, task):
        # run the task and pass in the DeviceIndex for OpenMM to
        # assign work to the correct GPU
        return task(DeviceIndex=str(self.worker_idx))
