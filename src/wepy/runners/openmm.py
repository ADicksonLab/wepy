"""OpenMM molecular dynamics runner with accessory classes.

OpenMM is a library with support for running molecular dynamics
simulations with specific support for fast GPU calculations. The
component based architecture of OpenMM makes it a perfect fit with
wepy.

In addition to the principle OpenMMRunner class there are a few
classes here that make using OpenMM runner more efficient.

First is a WalkerState class (OpenMMState) that wraps the openmm state
object directly, itself is a wrapper around the C++
datastructures. This gives better performance by not performing copies
to a WalkerState dictionary.

Second, is the OpenMMWalker which is identical to the Walker class
except that it enforces the state is an actual instantiation of
OpenMMState. Use of this is optional.

Finally, is the OpenMMGPUWorker class. This is to be used as the
worker type for the WorkerMapper work mapper. This is necessary to
allow passing of the device index to OpenMM for which GPU device to
use.

"""
from copy import copy
import random as rand
from warnings import warn
import logging
import time

import numpy as np

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from wepy.walker import Walker, WalkerState
from wepy.runners.runner import Runner
from wepy.work_mapper.worker import Worker
from wepy.reporter.reporter import Reporter
from wepy.util.util import box_vectors_to_lengths_angles

## Constants

KEYS = ('positions', 'velocities', 'forces', 'kinetic_energy',
        'potential_energy', 'time', 'box_vectors', 'box_volume',
        'parameters', 'parameter_derivatives')
"""Names of the fields of the OpenMMState."""

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
"""Mapping of key word arguments to the simulation.context.getState
method for retrieving data for a simulation state. By default we set
each as True to retrieve all information. The presence or absence of
them is handled by the OpenMMState.

"""

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
"""Mapping of units identifiers to the corresponding simtk.units Unit objects."""

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
"""Mapping of unit identifier strings to the serialized string spec of the unit."""

# a random seed will be chosen from 1 to RAND_SEED_RANGE_MAX when the
# Langevin integrator is created. 0 is the default and special value
# which will then choose a random value when the integrator is created

#TODO: test this isn't needed
#RAND_SEED_RANGE_MAX = 1000000

# the runner for the simulation which runs the actual dynamics
class OpenMMRunner(Runner):
    """Runner for OpenMM simulations."""

    def __init__(self, system, topology, integrator, platform=None):
        """Constructor for OpenMMRunner.

        Parameters
        ----------
        system : simtk.openmm.System object
            The system (forcefields) for the simulation.

        topology : simtk.openmm.app.Topology object
            The topology for you system.

        integrator : subclass simtk.openmm.Integrator object
            Integrator for propagating dynamics.

        platform : str
            The specification for the computational platform to
            use. If None uses OpenMM default platform, see OpenMM
            documentation for all value but typical ones are:
            Reference, CUDA, OpenCL. If value is None the automatic
            platform determining mechanism in OpenMM will be used.

        """

        # we save the different components. However, if we are to make
        # this runner picklable we have to convert the SWIG objects to
        # a picklable form
        self.system = system
        self.integrator = integrator

        # these are not SWIG objects
        self.topology = topology
        self.platform_name = platform

    #TODO: deprecate?
    def _openmm_swig_objects(self):
        """Just returns all of the foreign OpenMM module objects this class
        uses that are actually SWIG wrappers.

        Returns
        -------

        """

        return (self.system, self.integrator)

    def run_segment(self, walker, segment_length, getState_kwargs=None, **kwargs):
        """Run dynamics for the walker.

        Parameters
        ----------
        walker : object implementing the Walker interface
            The walker for which dynamics will be propagated.

        segment_length : int or float
            The numerical value that specifies how much dynamics are to be run.

        getState_kwargs : dict of str : bool, optional
            Specify the key-word arguments to pass to
            simulation.context.getState when getting simulation
            states. If None defaults to the values in the
            GET_STATE_KWARG_DEFAULTS module constant.


        Returns
        -------
        new_walker : object implementing the Walker interface
            Walker after dynamics was run, only the state should be modified.

        """

        run_segment_start = time.time()

        # set the kwargs that will be passed to getState
        tmp_getState_kwargs = getState_kwargs
        getState_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
        if tmp_getState_kwargs is not None:
            getState_kwargs.update(tmp_getState_kwargs)


        gen_sim_start = time.time()

        # make a copy of the integrator for this particular segment
        new_integrator = copy(self.integrator)
        # force setting of random seed to 0, which is a special
        # value that forces the integrator to choose another
        # random number
        new_integrator.setRandomNumberSeed(0)


        # create simulation object

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

        gen_sim_end = time.time()
        gen_sim_time = gen_sim_end - gen_sim_start
        logging.info("Time to generate the system: {}".format(gen_sim_time))


        # actually run the simulation

        steps_start = time.time()

        # Run the simulation segment for the number of time steps
        simulation.step(segment_length)

        steps_end = time.time()
        steps_time = steps_end - steps_start
        logging.info("Time to run {} sim steps: {}".format(segment_length, steps_time))


        get_state_start = time.time()


        get_state_end = time.time()
        get_state_time = get_state_end - get_state_start
        logging.info("Getting context state time: {}".format(get_state_time))


        # generate the new state/walker
        new_state = self.generate_state(simulation, segment_length,
                                        walker, getState_kwargs)

        # create a new walker for this
        new_walker = OpenMMWalker(new_state, walker.weight)

        run_segment_end = time.time()
        run_segment_time = run_segment_end - run_segment_start
        logging.info("Total internal run_segment time: {}".format(run_segment_time))

        return new_walker

    def generate_state(self, simulation, segment_length, starting_walker, getState_kwargs):
        """Method for generating a wepy compliant state from an OpenMM
        simulation object and data about the last segment of dynamics run.

        Parameters
        ----------

        simulation : simtk.openmm.app.Simulation object
            A complete simulation object from which the state will be extracted.

        segment_length : int
            The number of integration steps run in a segment of simulation.

        starting_walker : wepy.walker.Walker subclass object
            The walker that was the beginning of this segment of simyulation.

        getState_kwargs : dict of str : bool, optional
            Specify the key-word arguments to pass to
            simulation.context.getState when getting simulation
            states. If None defaults to the values in the
            GET_STATE_KWARG_DEFAULTS module constant.

        Returns
        -------

        new_state : wepy.runners.openmm.OpenMMState object
            A new state from the simulation state.

        This method is meant to be called from within the
        `run_segment` method during a simulation. It can be customized
        in subclasses to allow for the addition of custom attributes
        for a state, in addition to the base ones implemented in the
        interface to the openmm simulation state in OpenMMState.

        The extra arguments to this function are data that would allow
        for the calculation of integral values over the duration of
        the segment, such as time elapsed and differences from the
        starting state.

        """

        # save the state of the system with all possible values
        new_sim_state = simulation.context.getState(**getState_kwargs)

        # make an OpenMMState wrapper with this
        new_state = OpenMMState(new_sim_state)

        return new_state



class OpenMMState(WalkerState):
    """Walker state that wraps an simtk.openmm.State object.

    The keys for which values in the state are available are given by
    the KEYS module constant (accessible through the class constant of
    the same name as well).

    Additional fields can be added to these states through passing
    extra kwargs to the constructor. These will be automatically given
    a suffix of "_OTHER" to avoid name clashes.

    """

    KEYS = KEYS
    """The provided attribute keys for the state."""

    OTHER_KEY_TEMPLATE = "{}_OTHER"
    """String formatting template for attributes not set in KEYS."""

    def __init__(self, sim_state, **kwargs):
        """Constructor for OpenMMState.

        Parameters
        ----------
        state : simtk.openmm.State object
            The simulation state retrieved from the simulation constant.

        kwargs : optional

            Additional attributes to set for the state. Will add the
        "_OTHER" suffix to the keys

        """

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
        """The underlying simtk.openmm.State object this is wrapping."""
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

            # handle the parameters differently since they are dictionaries of values
            elif key.startswith('parameters'):
                parameters_dict = self.parameters_values()
                if parameters_dict is None:
                    return None
                else:
                    return self._get_nested_attr_from_compound_key(key, parameters_dict)
            elif key.startswith('parameter_derivatives'):
                pd_dict = self.parameter_derivatives_values()
                if pd_dict is None:
                    return None
                else:
                    return self._get_nested_attr_from_compound_key(key, pd_dict)

    ## Array properties

    # Positions
    @property
    def positions(self):
        """The positions of the state as a numpy array simtk.units.Quantity object."""
        try:
            return self.sim_state.getPositions(asNumpy=True)
        except:
            warn("Unknown exception handled from `self.sim_state.getPositions()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def positions_unit(self):
        """The units (as a simtk.units.Unit object) the positions are in."""
        return self.positions.unit

    def positions_values(self):
        """The positions of the state as a numpy array in the positions_unit
        simtk.units.Unit. This is what is returned by the __getitem__
        accessor.

        """
        return self.positions.value_in_unit(self.positions_unit)

    # Velocities
    @property
    def velocities(self):
        """The velocities of the state as a numpy array simtk.units.Quantity object."""
        try:
            return self.sim_state.getVelocities(asNumpy=True)
        except:
            warn("Unknown exception handled from `self.sim_state.getVelocities()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def velocities_unit(self):
        """The units (as a simtk.units.Unit object) the velocities are in."""
        return self.velocities.unit

    def velocities_values(self):
        """The velocities of the state as a numpy array in the velocities_unit
        simtk.units.Unit. This is what is returned by the __getitem__
        accessor.

        """

        velocities = self.velocities
        if velocities is None:
            return None
        else:
            return self.velocities.value_in_unit(self.velocities_unit)

    # Forces
    @property
    def forces(self):
        """The forces of the state as a numpy array simtk.units.Quantity object."""
        try:
            return self.sim_state.getForces(asNumpy=True)
        except:
            warn("Unknown exception handled from `self.sim_state.getForces()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def forces_unit(self):
        """The units (as a simtk.units.Unit object) the forces are in."""
        return self.forces.unit

    def forces_values(self):
        """The forces of the state as a numpy array in the forces_unit
        simtk.units.Unit. This is what is returned by the __getitem__
        accessor.

        """

        forces = self.forces
        if forces is None:
            return None
        else:
            return self.forces.value_in_unit(self.forces_unit)

    # Box Vectors
    @property
    def box_vectors(self):
        """The box vectors of the state as a numpy array simtk.units.Quantity object."""
        try:
            return self.sim_state.getPeriodicBoxVectors(asNumpy=True)
        except:
            warn("Unknown exception handled from `self.sim_state.getPeriodicBoxVectors()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def box_vectors_unit(self):
        """The units (as a simtk.units.Unit object) the box vectors are in."""
        return self.box_vectors.unit

    def box_vectors_values(self):
        """The box vectors of the state as a numpy array in the
        box_vectors_unit simtk.units.Unit. This is what is returned by
        the __getitem__ accessor.

        """

        box_vectors = self.box_vectors
        if box_vectors is None:
            return None
        else:
            return self.box_vectors.value_in_unit(self.box_vectors_unit)


    ## non-array properties

    # Kinetic Energy
    @property
    def kinetic_energy(self):
        """The kinetic energy of the state as a numpy array simtk.units.Quantity object."""
        try:
            return self.sim_state.getKineticEnergy()
        except:
            warn("Unknown exception handled from `self.sim_state.getKineticEnergy()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def kinetic_energy_unit(self):
        """The units (as a simtk.units.Unit object) the kinetic energy is in."""
        return self.kinetic_energy.unit

    def kinetic_energy_value(self):
        """The kinetic energy of the state as a numpy array in the kinetic_energy_unit
        simtk.units.Unit. This is what is returned by the __getitem__
        accessor.

        """

        kinetic_energy = self.kinetic_energy
        if kinetic_energy is None:
            return None
        else:
            return np.array([self.kinetic_energy.value_in_unit(self.kinetic_energy_unit)])

    # Potential Energy
    @property
    def potential_energy(self):
        """The potential energy of the state as a numpy array simtk.units.Quantity object."""
        try:
            return self.sim_state.getPotentialEnergy()
        except:
            warn("Unknown exception handled from `self.sim_state.getPotentialEnergy()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def potential_energy_unit(self):
        """The units (as a simtk.units.Unit object) the potential energy is in."""
        return self.potential_energy.unit

    def potential_energy_value(self):
        """The potential energy of the state as a numpy array in the potential_energy_unit
        simtk.units.Unit. This is what is returned by the __getitem__
        accessor.

        """

        potential_energy = self.potential_energy
        if potential_energy is None:
            return None
        else:
            return np.array([self.potential_energy.value_in_unit(self.potential_energy_unit)])

    # Time
    @property
    def time(self):
        """The time of the state as a numpy array simtk.units.Quantity object."""
        try:
            return self.sim_state.getTime()
        except:
            warn("Unknown exception handled from `self.sim_state.getTime()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def time_unit(self):
        """The units (as a simtk.units.Unit object) the time is in."""
        return self.time.unit

    def time_value(self):
        """The time of the state as a numpy array in the time_unit
        simtk.units.Unit. This is what is returned by the __getitem__
        accessor.

        """

        time = self.time
        if time is None:
            return None
        else:
            return np.array([self.time.value_in_unit(self.time_unit)])

    # Box Volume
    @property
    def box_volume(self):
        """The box volume of the state as a numpy array simtk.units.Quantity object."""
        try:
            return self.sim_state.getPeriodicBoxVolume()
        except:
            warn("Unknown exception handled from `self.sim_state.getPeriodicBoxVolume()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def box_volume_unit(self):
        """The units (as a simtk.units.Unit object) the box volume is in."""
        return self.box_volume.unit

    def box_volume_value(self):
        """The box volume of the state as a numpy array in the box_volume_unit
        simtk.units.Unit. This is what is returned by the __getitem__
        accessor.

        """

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
        """The parameters of the state as a dictionary mapping the names of
        the parameters to their values which are numpy array
        simtk.units.Quantity objects.

        """

        try:
            return self.sim_state.getParameters()
        except:
            warn("Unknown exception handled from `self.sim_state.getParameters()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def parameters_unit(self):
        """The units for each parameter as a dictionary mapping parameter
        names to their corresponding unit as a simtk.units.Unit
        object.

        """
        param_units = {key : None for key, val in self.parameters.items()}
        return param_units

    def parameters_values(self):
        """The parameters of the state as a dictionary mapping the name of the
        parameter to a numpy array in the unit for the parameter of the
        same name in the parameters_unit corresponding
        simtk.units.Unit object. This is what is returned by the
        __getitem__ accessor using the compound key syntax with the
        prefix 'parameters', e.g. state['parameter/paramA'] for the
        parameter 'paramA'.

        """

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
        """The parameter derivatives of the state as a dictionary mapping the
        names of the parameters to their values which are numpy array
        simtk.units.Quantity objects.

        """

        try:
            return self.sim_state.getEnergyParameterDerivatives()
        except:
            warn("Unknown exception handled from `self.sim_state.getEnergyParameterDerivatives()`, "
                     "this is probably because this attribute is not in the State.")
            return None

    @property
    def parameter_derivatives_unit(self):
        """The units for each parameter derivative as a dictionary mapping
        parameter names to their corresponding unit as a
        simtk.units.Unit object.

        """

        param_units = {key : None for key, val in self.parameter_derivatives.items()}
        return param_units

    def parameter_derivatives_values(self):
        """The parameter derivatives of the state as a dictionary mapping the
        name of the parameter to a numpy array in the unit for the
        parameter of the same name in the parameters_unit
        corresponding simtk.units.Unit object. This is what is
        returned by the __getitem__ accessor using the compound key
        syntax with the prefix 'parameter_derivatives',
        e.g. state['parameter_derivatives/paramA'] for the parameter
        'paramA'.

        """

        if self.parameter_derivatives is None:
            return None

        param_arrs = {key : np.array(val) for key, val
                          in self.parameter_derivatives.items()}

        # return None if there is nothing in this
        if len(param_arrs) == 0:
            return None
        else:
            return param_arrs

    # for the dict attributes we need to transform the keys for making
    # a proper state where all __getitem__ things are arrays
    def _dict_attr_to_compound_key_dict(self, root_key, attr_dict):
        """Transform a dictionary of values within the compound key 'root_key'
        to a dictionary mapping compound keys to values.

        For example give the root_key 'parameters' and the parameters
        dictionary {'paramA' : 1.234} returns {'parameters/paramA' : 1.234}.

        Parameters
        ----------
        root_key : str
            The compound key prefix
        attr_dict : dict of str : value
            The dictionary with simple keys within the root key namespace.

        Returns
        -------
        compound_key_dict : dict of str : value
            The dictionary with the compound keys.

        """

        key_template = "{}/{}"
        cmpd_key_d = {}
        for key, value in attr_dict.items():

            new_key = key_template.format(root_key, key)
            # if this is a proper feature
            if type(value) == np.ndarray:
                cmpd_key_d[new_key] = value
            elif hasattr(value, '__getitem__'):
                cmpd_key_d.update(self._dict_attr_to_compound_key_dict(new_key, value))
            else:
                raise TypeError("Unsupported attribute type")

        return cmpd_key_d

    def _get_nested_attr_from_compound_key(self, compound_key, compound_feat_dict):
        """Get arbitrarily deeply nested compound keys from the full
        dictionary tree.

        Parameters
        ----------
        compound_key : str
            Compound key separated by '/' characters

        compound_feat_dict : dict
            Dictionary of arbitrary depth

        Returns
        -------
        value
            Value requested by the key.

        """

        key_components = compound_key.split('/')

        # if there is only one component of the key then it is not
        # really compound, we won't complain just return the
        # "dictionary" if it is not actually a dict like
        if not hasattr(compound_feat_dict, '__getitem__'):
            raise TypeError("Must provide a dict-like with the compound key")

        value = compound_feat_dict[key_components[0]]

        # if the value itself is compound recursively fetch the value
        if hasattr(value, '__getitem__') and len(key_components[1:]) > 0:

            subgroup_key = '/'.join(key_components[1:])

            return self._get_nested_attr_from_compound_key(subgroup_key, value)

        elif hasattr(value, '__getitem__') and len(key_components[1:]) < 1:
            raise ValueError("Key does not reference a leaf node of attribute")

        # otherwise we have the right key so return the object
        else:
            return value

    def parameters_features(self):
        """Returns a dictionary of the parameters with their appropriate
        compound keys. This can be used for placing them in the same namespace
        as the rest of the attributes."""

        parameters = self.parameters_values()
        if parameters is None:
            return None
        else:
            return self._dict_attr_to_compound_key_dict('parameters', parameters)

    def parameter_derivatives_features(self):
        """Returns a dictionary of the parameter derivatives with their appropriate
        compound keys. This can be used for placing them in the same namespace
        as the rest of the attributes."""

        parameter_derivatives = self.parameter_derivatives_values()
        if parameter_derivatives is None:
            return None
        else:
            return self._dict_attr_to_compound_key_dict('parameter_derivatives',
                                                        parameter_derivatives)

    def omm_state_dict(self):
        """Return a dictionary with all of the default keys from the wrapped
        simtk.openmm.State object"""

        feature_d = {'positions' : self.positions_values(),
                'velocities' : self.velocities_values(),
                'forces' : self.forces_values(),
                'kinetic_energy' : self.kinetic_energy_value(),
                'potential_energy' : self.potential_energy_value(),
                'time' : self.time_value(),
                'box_vectors' : self.box_vectors_values(),
                'box_volume' : self.box_volume_value(),
                    }

        params = self.parameters_features()
        param_derivs = self.parameter_derivatives_features()
        if params is not None:
            feature_d.update(params)
        if param_derivs is not None:
            feature_d.update(param_derivs)

        return feature_d

    def dict(self):
        # documented in superclass

        d = {}
        for key, value in self._data.items():
            d[key] = value
        for key, value in self.omm_state_dict().items():
            d[key] = value
        return d

    def to_mdtraj(self, topology):
        """Returns an mdtraj.Trajectory object from this walker's state.

        Parameters
        ----------
        topology : mdtraj.Topology object
            Topology for the state.

        Returns
        -------
        state_traj : mdtraj.Trajectory object

        """

        import mdtraj as mdj
        # resize the time to a 1D vector
        unitcell_lengths, unitcell_angles = box_vectors_to_lengths_angles(self.box_vectors)
        return mdj.Trajectory([self.positions],
                              unitcell_lengths=[unitcell_lengths],
                              unitcell_angles=[unitcell_angles],
                              topology=topology)

class OpenMMWalker(Walker):
    """Walker for OpenMMRunner simulations.

    This simply enforces the use of an OpenMMState object for the
    walker state attribute.

    """

    def __init__(self, state, weight):
        # documented in superclass

        assert isinstance(state, OpenMMState), \
            "state must be an instance of class OpenMMState not {}".format(type(state))

        super().__init__(state, weight)


class OpenMMCPUWorker(Worker):
    """Worker for OpenMM GPU simulations (CUDA or OpenCL platforms).

    This is intended to be used with the wepy.work_mapper.WorkerMapper
    work mapper class.

    This class must be used in order to ensure OpenMM runs jobs on the
    appropriate GPU device.

    """

    NAME_TEMPLATE = "OpenMMCPUWorker-{}"
    """The name template the worker processes are named to substituting in
    the process number."""

    def __init__(self, worker_idx, task_queue, result_queue, num_threads=1, **kwargs):

        super().__init__(worker_idx, task_queue, result_queue,
                         num_threads=num_threads, **kwargs)

    def run_task(self, task):
        # documented in superclass

        # run the task and pass in the DeviceIndex for OpenMM to
        # assign work to the correct GPU
        return task(CpuThreads=self.attributes['num_threads'])


class OpenMMGPUWorker(Worker):
    """Worker for OpenMM GPU simulations (CUDA or OpenCL platforms).

    This is intended to be used with the wepy.work_mapper.WorkerMapper
    work mapper class.

    This class must be used in order to ensure OpenMM runs jobs on the
    appropriate GPU device.

    """

    NAME_TEMPLATE = "OpenMMGPUWorker-{}"
    """The name template the worker processes are named to substituting in
    the process number."""

    def run_task(self, task):
        # documented in superclass

        # run the task and pass in the DeviceIndex for OpenMM to
        # assign work to the correct GPU
        return task(DeviceIndex=str(self.worker_idx))
