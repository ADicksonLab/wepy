import json

import numpy as np

import h5py

def load_dataset(path):
    return None

class TrajHDF5(object):

    def __init__(self, filename, mode='x',
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
        r+       Read/write, file must exist
        w        Create file, truncate if exists
        x or w-  Create file, fail if exists
        a        Read/write if exists, create otherwise
        c        Append (concatenate) file if exists, create otherwise,
                   read access only to existing data,
                   can append to existing datasets
        c-       Append file if exists, create otherwise,
                   read access only to existing data,
                   cannot append to existing datasets

        The c and c- modes use the h5py 'a' mode underneath and limit
        access to data that is read in.

        If `overwrite` is True then the previous data will be
        re-initialized upon this constructor being called.

        """
        assert mode in ['r', 'r+', 'w', 'w-', 'x', 'a', 'c', 'c-'], \
          "mode must be either 'r', 'r+', 'w', 'x', 'w-', 'a', 'c', or 'c-'"

        self._filename = filename
        # the top level mode enforced by wepy.hdf5
        self._wepy_mode = mode

        # get h5py compatible I/O mode
        if self._wepy_mode in ['c', 'c-']:
            h5py_mode = 'a'
        else:
            h5py_mode = self._wepy_mode
        # the lower level h5py mode
        self._h5py_mode = h5py_mode

        # open the file
        h5 = h5py.File(filename, mode=self._h5py_mode)
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

        # initialize the exist flags, which say whether a dataset
        # exists or not
        self._exist_flags = {key : False for key in self._keys}

        # initialize the append flags dictionary, this keeps track of
        # whether a data field can be appended to or not
        self._append_flags = {key : True for key in self._keys}

        # some of these data fields are mandatory and others are
        # optional
        self._mandatory_keys = ['positions']

        # some data fields are compound and have more than one dataset
        # associated with them
        self._compound_keys = ['forces', 'parameters', 'observables']

        ## Dataset Compliances
        # a file which has different levels of keys can be used for
        # different things, so we define these collections of keys,
        # and flags to keep track of which ones this dataset
        # satisfies, ds here stands for "dataset"
        self._compliance_types = ['COORDS', 'TRAJ', 'RESTART', 'FORCED']
        self._compliance_flags = {key : False for key in self._compliance_types}
        # the minimal requirement (and need for this class) is to associate
        # a collection of coordinates to some molecular structure (topology)
        self._compliance_requirements = {'COORDS' :  ['topology', 'positions'],
                                         'TRAJ' :    ['topology', 'positions',
                                                      'time', 'box_vectors'],
                                         'RESTART' : ['topology', 'positions',
                                                      'time', 'box_vectors',
                                                      'velocities'],
                                         'FORCED' :  ['topology', 'positions',
                                                      'time', 'box_vectors',
                                                      'velocities',
                                                      'forces']
                                            }

        # create file mode: 'w' will create a new file or overwrite,
        # 'w-' and 'x' will not overwrite but will create a new file
        if self._wepy_mode in ['w', 'w-', 'x']:
            self._create_init(topology, data, units)

        # read/write mode: in this mode we do not completely overwrite
        # the old file and start again but rather write over top of
        # values if requested
        elif self._wepy_mode in ['r+']:
            self._read_write_init(topology, data, units)

        # add mode: read/write create if doesn't exist
        elif self._wepy_mode in ['a']:
            self._add_init(topology, data, units)

        # append mode
        elif self._wepy_mode in ['c', 'c-']:
            # use the hidden init function for appending data
            self._append_init(topology, data, units)

        # read only mode
        elif self._wepy_mode == 'r':
            # if any data was given, raise an error
            assert (not any([True if (value is not None) else False for key, value in data.items()]))\
              or (not any([True if (value is not None) else False for key, value in units.items()])),\
                "Data was provided for a read-only operation"

            # then run the initialization process
            self._read_init()

        # update the compliance type flags of the dataset
        self._update_compliance_flags()

    def _update_exist_flags(self):
        """Inspect the hdf5 object and set flags if data exists for the fields."""
        for key in self._keys:
            if key in list(self._h5.keys()):
                self._exist_flags[key] = True
            else:
                self._exist_flags[key] = False

    def _update_append_flags(self):
        """Sets flags to False (they are initialized to True) if the dataset
        currently exists."""
        for dataset_key, exist_flag in self._exist_flags.items():
            if exist_flag:
                self._append_flags[dataset_key] = False

    def _write_datasets(self, data, units):

        # go through each data field and add them, using the associated units
        for key, value in data.items():

            # if the value is None it was not set and we should just
            # continue without checking silently, unless it is mandatory
            if value is None:
                continue

            # try to add the data using the setter
            try:
                self.__setattr__(key, value)
            except AssertionError:
                raise ValueError("{} value not valid".format(key))

            ## Units
            # # make the key for the unit
            # if key in self._compound_keys:
            #     # if it is compound name it plurally for heterogeneous data
            #     unit_key = "{}_units".format(key)
            # else:
            #     # or just keep it singular for homogeneous data
            #     unit_key = "{}_unit".format(key)

            # # try to add the units
            # try:
            #     self.__setattr__(unit_key, units[key])
            # except AssertionError:
            #     raise ValueError("{} unit not valid".format(key))

    ### The init functions for different I/O modes
    def _create_init(self, topology, data, units):
        """Completely overwrite the data in the file. Reinitialize the values
        and set with the new ones if given."""
        # make sure the mandatory data is here
        assert topology is not None, "Topology must be given"
        assert data['positions'] is not None, "positions must be given"

        # assign the topology
        self.topology = topology

        # write all the datasets
        self._write_datasets(data, units)

    def _read_write_init(self, topology, data, units):
        """Write over values if given but do not reinitialize any old ones. """

        # set the flags for existing data
        self._update_exist_flags()

        # add new topology if it is given
        if topology is not None:
            self.topology = topology

        # add new data if given
        self._write_datasets(data, units)

    def _add_init(self, topology, data, units):
        """Create the dataset if it doesn't exist and put it in r+ mode,
        otherwise, just open in r+ mode."""

        # set the flags for existing data
        self._update_exist_flags()

        if not any(self._exist_flags):
            self._create_init(topology, data, units)
        else:
            self._read_write_init(topology, data, units)

    def _append_init(self, topology, data, units):
        """Append mode initialization. Checks for given data and sets flags,
        and adds new data if given."""

        # if the file has any data to start we write the data to it
        if not any(self._exist_flags):
            self.topology = topology
            self._write_datasets(data, units)
            if self.mode == 'c-':
                self._update_append_flags()
        # otherwise we first set the flags for what data was read and
        # then add to it only
        else:
            # initialize the flags from the read data
            self._update_exist_flags()

            # restrict append permissions for those that have their flags
            # set from the read init
            if self.mode == 'c-':
                self._update_append_flags()

            # write new datasets
            self._write_datasets(data, units)

    def _read_init(self):
        """Read only mode initialization. Simply checks for the presence of
        and sets attribute flags."""
        # we just need to set the flags for which data is present and
        # which is not
        self._update_exist_flags()

    @property
    def filename(self):
        return self._filename

    @property
    def mode(self):
        return self._wepy_mode

    @property
    def h5_mode(self):
        return self._h5.mode

    def close(self):
        if not self.closed:
            self._h5.close()
            self.closed = True

    def __del__(self):
        self.close()

    def _update_compliance_flags(self):
        """Checks whether the flags for different datasets and updates the
        flags for the compliance groups."""
        for compliance_type in self._compliance_types:
            # check if compliance for this type is met
            result = self._check_compliance_keys(compliance_type)
            # set the flag
            self._compliance_flags[compliance_type] = result

    def _check_compliance_keys(self, compliance_type):
        """Checks whether the flags for the datasets have been set to True."""
        results = []
        for dataset_key in self._compliance_requirements[compliance_type]:
            results.append(self._exist_flags[dataset_key])
        return all(results)

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
        self._exist_flags['topology'] = True
        # if we are in strict append mode we cannot append after we create something
        if self.mode == 'c-':
            self._append_flags['topology'] = False

    def append_dataset(self, dataset_key, data):
        if self.mode == 'c-':
            assert self._append_flags[dataset_key], "dataset is not available for appending to"
        raise NotImplementedError('feature not finished')

    @property
    def positions(self):
        return self._h5['positions']

    @positions.setter
    def positions(self, positions):
        assert isinstance(positions, np.ndarray), "positions must be a numpy array"
        self._h5.create_dataset('positions', data=positions)
        self._exist_flags['positions'] = True
        # if we are in strict append mode we cannot append after we create something
        if self.mode == 'c-':
            self._append_flags['positions'] = False

    @property
    def time(self):
        if self._exist_flags['time']:
            return self._h5['time']
        else:
            return None

    @time.setter
    def time(self, time):
        assert isinstance(time, np.ndarray), "time must be a numpy array"
        self._h5.create_dataset('time', data=time)
        self._exist_flags['time'] = True
        # if we are in strict append mode we cannot append after we create something
        if self.mode == 'c-':
            self._append_flags['time'] = False

    @property
    def box_vectors(self):
        if self._exist_flags['box_vectors']:
            return self._h5['box_vectors']
        else:
            return None

    @box_vectors.setter
    def box_vectors(self, box_vectors):
        assert isinstance(box_vectors, np.ndarray), "box_vectors must be a numpy array"
        self._h5.create_dataset('box_vectors', data=box_vectors)
        self._exist_flags['box_vectors'] = True
        # if we are in strict append mode we cannot append after we create something
        if self.mode == 'c-':
            self._append_flags['box_vectors'] = False

    @property
    def velocities(self):
        if self._exist_flags['velocities']:
            return self._h5['velocities']
        else:
            return None

    @velocities.setter
    def velocities(self, velocities):
        assert isinstance(velocities, np.ndarray), "velocities must be a numpy array"
        self._h5.create_dataset('velocities', data=velocities)
        self._exist_flags['velocities'] = True
        # if we are in strict append mode we cannot append after we create something
        if self.mode == 'c-':
            self._append_flags['velocities'] = False


    ### These properties are not a simple dataset and should actually
    ### each be groups of datasets, even though there will be a net
    ### force we want to be able to have all forces which then the net
    ### force will be calculated from
    @property
    def forces(self):
        if self._exist_flags['forces']:
            return self._h5['forces']
        else:
            return None

    @forces.setter
    def forces(self, forces):
        self._h5.create_dataset('forces', data=forces)
        self._exist_flags['forces'] = True
        # if we are in strict append mode we cannot append after we create something
        if self.mode == 'c-':
            self._append_flags['forces'] = False

    @property
    def parameters(self):
        if self._exist_flags['parameters']:
            return self._h5['parameters']
        else:
            return None

    @parameters.setter
    def parameters(self, parameters):
        self._h5.create_dataset('parameters', data=parameters)
        self._exist_flags['parameters'] = True
        # if we are in strict append mode we cannot append after we create something
        if self.mode == 'c-':
            self._append_flags['parameters'] = False

    @property
    def observables(self):
        if self._exist_flags['observables']:
            return self._h5['observables']
        else:
            return None

    @observables.setter
    def observables(self, observables):
        self._h5.create_dataset('observables', data=observables)
        self._exist_flags['observables'] = True
        # if we are in strict append mode we cannot append after we create something
        if self.mode == 'c-':
            self._append_flags['observables'] = False


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
