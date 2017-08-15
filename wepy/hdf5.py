import json

import numpy as np

import h5py

def load_dataset(path):
    return None

class TrajHDF5(object):

    def __init__(self, filename, mode='x', overwrite=True,
                 topology=None,
                 positions=None,
                 time=None,
                 box_vectors=None,
                 velocities=None,
                 positions_unit=None,
                 time_unit=None,
                 box_vectors_unit=None,
                 velocities_unit=None,
                 forces=None,
                 parameters=None,
                 observables=None,
                 forces_units=None,
                 parameters_units=None,
                 observables_units=None):

        """Initializes a TrajHDF5 object which is a format for storing
        trajectory data in and HDF5 format file which can be used on
        it's own or encapsulated in a WepyHDF5 object.

        mode:
        r        Readonly, file must exist
        w        Create file, truncate if exists
        x        Create file, fail if exists
        a        Append mode, file must exist

        If `overwrite` is True then the previous data will be
        re-initialized upon this constructor being called.

        """
        assert mode in ['r', 'w', 'x', 'a'], "mode must be either r, w, x or a"

        self._filename = filename

        # open the file
        h5 = h5py.File(filename, mode)
        self._h5 = h5
        self.closed = False

        # all the keys for the datasets and groups
        self._keys = ['topology', 'positions',
                      'time', 'box_vectors',
                      'velocities',
                      'forces', 'parameters',
                      'observables']

        # collect the non-topology attributes into a dict
        data = {'positions' : positions,
                'time' : time,
                'box_vectors' : box_vectors,
                'velocities' : velocities,
                'forces' : forces,
                'parameters' : parameters,
                'observables' : observables
               }

        units = {'positions' : positions_unit,
                 'time' : time_unit,
                 'box_vectors' : box_vectors_unit,
                 'velocities' : velocities_unit,
                 'forces' : forces_units,
                 'parameters' : parameters_units,
                 'observables' : observables_units
                }

        # some of these data fields are mandatory and others are
        # optional
        self._mandatory_keys = ['positions']

        # some data fields are compound and have more than one dataset
        # associated with them
        self._compound_keys = ['forces', 'parameters', 'observables']

        ## DataSet satisfaction
        # a file which has different levels of keys can be used for
        # different things, so we define these collections of keys,
        # and flags to keep track of which ones this dataset
        # satisfies, ds here stands for "dataset"
        self._compliance_types = ['COORDS', 'TRAJ', 'RESTART', 'FORCED']
        for key in self._compliance_types:
            key = "_{}_DS".format(key)
            self.__dict__[key] = False

        # the minimal requirement (and need for this class) is to associate
        # a collection of coordinates to some molecular structure (topology)

        # only a collection of molecular coordinates
        self._COORDS_keys = ['topology', 'positions']
        # frames from an actual dynamics trajectory
        self._TRAJ_keys = ['topology', 'positions', 'time', 'box_vectors']
        # restart trajectories
        self._RESTART_keys = ['topology', 'positions',
                              'time', 'box_vectors',
                              'velocities']
        # simulation with external forces defined, forced trajectory
        self._FORCED_keys = ['topology', 'positions',
                             'time', 'box_vectors',
                             'velocities',
                             'forces']

        # the overwrite option indicates that we want to completely
        # overwrite the old file
        if mode in ['w', 'x'] and overwrite:
            # use the hidden init function for writing a new hdf5 file
            self._overwrite_init(topology, data, units)
        # in this mode we do not completely overwrite the old file and
        # start again but rather write over top of values if requested
        elif mode in ['w', 'x']:
            self._write_init(topology=topology, data=data, units=units)
        elif mode == 'a':
            # use the hidden init function for appending data
            self._append_init(data, units)
        elif mode == 'r':
            self._read_init()

        self._update_compliance()

    def _write_init(self, topology=None, data=None, units=None):
        raise NotImplementedError("feature not finished")

    def _overwrite_init(self, topology, data, units):

        # initialize the topology flag
        self._topology = False
        # set the topology, will raise error internally
        self.topology = topology

        # go through each data field and add them, using the associated units
        for key, value in data.items():

            # initialize the attribute
            attr_key = "_{}".format(key)
            self.__dict__[attr_key] = False

            # if the value is None it was not set and we should just
            # continue without checking silently, unless it is mandatory
            if value is None:
                if key in self._mandatory_keys:
                    raise ValueError("{} is mandatory and must be given a value".format(key))
                else:
                    continue

            # try to add the data using the setter
            try:
                self.__setattr__(key, value)
            except AssertionError:
                raise ValueError("{} value not valid".format(key))

            ## Units

            # make the key for the unit
            if key in self._compound_keys:
                # if it is compound name it plurally for heterogeneous data
                unit_key = "{}_units".format(key)
            else:
                # or just keep it singular for homogeneous data
                unit_key = "{}_unit".format(key)

            # try to add the units
            try:
                self.__setattr__(unit_key, units[key])
            except AssertionError:
                raise ValueError("{} unit not valid".format(key))

    def _read_init(self):

        # we just need to set the flags for which data is present and
        # which is not
        for key in self._keys:
            flag_key = "_{}".format(key)
            if key in list(self._h5.keys()):
                self.__dict__[flag_key] = True
            else:
                self.__dict__[flag_key] = False

    def _append_init(self, data, units):

        # _read_init figures out which data is present and sets the
        # flags so we will initialize with it
        self._read_init()

        # we go through and add given data if it is not already set,
        # we rely on h5py to enforce write rules for append
        for key, value in data.items():

            # if the value is None it was not set and we should just
            # continue without checking silently
            if value is None:
                continue

            # try to add the data using the setter
            try:
                self.__setattr__(key, value)
            except AssertionError:
                raise ValueError("{} value not valid".format(key))

            ## Units

            # make the key for the unit
            if key in self._compound_keys:
                # if it is compound name it plurally for heterogeneous data
                unit_key = "{}_units".format(key)
            else:
                # or just keep it singular for homogeneous data
                unit_key = "{}_unit".format(key)

            # try to add the units
            try:
                self.__setattr__(unit_key, units[key])
            except AssertionError:
                raise ValueError("{} unit not valid".format(key))

        # TODO units

    @property
    def filename(self):
        return self._filename

    def close(self):
        if not self.closed:
            self._h5.close()
            self.closed = True

    def __del__(self):
        self.close()

    def _update_compliance(self):
        """Checks whether the flags for different datasets and updates the
        flags for the compliance groups."""
        for compliance_type in self._compliance_types:
            # we will get the function from the compliance type token
            test_func_str = "check_compliance_{}".format(compliance_type)
            test_func = self.__getattribute__(test_func_str)
            # then test this object
            result = test_func()
            # and save the result
            compliance_key = "_{}_DS".format(compliance_type)
            self.__dict__[compliance_key] = result

    def _check_compliance_keys(self, compliance_type):
        """Checks whether the flags for the datasets have been set to True."""
        compliance_keys_str = "_{}_keys".format(compliance_type)
        results = []
        for key in self.__dict__[compliance_keys_str]:
            attr_key = "_{}".format(key)
            results.append(self.__getattribute__(attr_key))
        return all(results)

    def check_compliance_COORDS(self):
        return self._check_compliance_keys('COORDS')

    def check_compliance_TRAJ(self):
        return self._check_compliance_keys('TRAJ')

    def check_compliance_RESTART(self):
        return self._check_compliance_keys('RESTART')

    def check_compliance_FORCED(self):
        return self._check_compliance_keys('FORCED')

    @property
    def h5(self):
        return self._h5

    @property
    def topology(self):
        return self._h5['topology']

    @topology.setter
    def topology(self, topology):
        try:
            json_d = json.loads(topology)
            del json_d
        except json.JSONDecodeError:
            raise ValueError("topology must be a valid JSON string")

        self._h5.create_dataset('topology', data=topology)
        self._topology = True

    @property
    def positions(self):
        return self._h5['positions']

    @positions.setter
    def positions(self, positions):
        assert isinstance(positions, np.ndarray), "positions must be a numpy array"
        self._h5.create_dataset('positions', data=positions)
        self._positions = True

    @property
    def time(self):
        if self._time:
            return self._h5['time']
        else:
            return None

    @time.setter
    def time(self, time):
        assert isinstance(time, np.ndarray), "time must be a numpy array"
        self._h5.create_dataset('time', data=time)
        self._time = True

    @property
    def box_vectors(self):
        if self._box_vectors:
            return self._h5['box_vectors']
        else:
            return None

    @box_vectors.setter
    def box_vectors(self, box_vectors):
        assert isinstance(box_vectors, np.ndarray), "box_vectors must be a numpy array"
        self._h5.create_dataset('box_vectors', data=box_vectors)
        self._box_vectors = True

    @property
    def velocities(self):
        if self._velocities:
            return self._h5['velocities']
        else:
            return None

    @velocities.setter
    def velocities(self, velocities):
        assert isinstance(velocities, np.ndarray), "velocities must be a numpy array"
        self._h5.create_dataset('velocities', data=velocities)
        self._velocities = True


    ### These properties are not a simple dataset and should actually
    ### each be groups of datasets, even though there will be a net
    ### force we want to be able to have all forces which then the net
    ### force will be calculated from
    @property
    def forces(self):
        if self._forces:
            return self._h5['forces']
        else:
            return None

    @forces.setter
    def forces(self, forces):
        self._h5.create_dataset('forces', data=forces)
        self._forces = True

    @property
    def parameters(self):
        if self._parameters:
            return self._h5['parameters']
        else:
            return None

    @parameters.setter
    def parameters(self, parameters):
        self._h5.create_dataset('parameters', data=parameters)
        self._parameters = True

    @property
    def observables(self):
        if self._observables:
            return self._h5['observables']
        else:
            return None

    @observables.setter
    def observables(self, observables):
        self._h5.create_dataset('observables', data=observables)
        self._observables = True



class WepyHDF5(object):

    def __init__(self, filename, mode='x', topology=None, overwrite=True):
        """Initialize a new Wepy HDF5 file. This is a file that organizes
        wepy.TrajHDF5 dataset subsets by simulations by runs and
        includes resampling records for recovering walker histories.

        mode:
        r        Readonly, file must exist
        w        Create file, truncate if exists
        x        Create file, fail if exists
        a        Append mode, file must exist

        If `overwrite` is True then the previous data will be
        re-initialized upon this constructor being called.

        """
        assert mode in ['r', 'w', 'x', 'a'], "mode must be either r, w, x or a"

        self._filename = filename

        # open the file
        h5 = h5py.File(filename, mode)
        self._h5 = h5
        self.closed = False

        if mode in ['w', 'x'] and overwrite:
            self._runs = self._h5.create_group('runs')
            # this keeps track of the number of runs. The current
            # value will be the name of the next run that is added,
            # and this should be incremented when that happens
            self._run_idx_counter = 0
            if topology:
                self.topology = topology


    @property
    def filename(self):
        return self._filename

    def close(self):
        if not self.closed:
            self._h5.close()
            self.closed = True

    def __del__(self):
        self.close()

    @property
    def h5(self):
        return self._h5

    @property
    def runs(self):
        return self._h5['runs']

    @property
    def topology(self):
        return self._h5['topology']

    @topology.setter
    def topology(self, topology):
        self._h5.create_dataset('topology', data=topology)

    def new_run(self, **kwargs):
        # create a new group named the next integer in the counter
        run_grp = self._h5.create_group('runs/{}'.format(str(self._run_idx_counter)))
        # increment the counter
        self._run_idx_counter += 1

        # add metadata if given
        for key, val in kwargs.items():
            run_grp.attrs[key] = val

        return run_grp
