import json
from warnings import warn

import numpy as np

import h5py

# Constants
N_DIMS = 3
TRAJ_DATA_FIELDS = ['positions', 'time', 'box_vectors', 'velocities',
                    'forces', 'parameters', 'forces']

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
            # if any data was given, warn the user
            if (not any([True if (value is not None) else False for key, value in data.items()]))\
              or (not any([True if (value is not None) else False for key, value in units.items()])):
                warn("Data was provided for a read-only operation", RuntimeWarning)

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
            if self._wepy_mode == 'c-':
                self._update_append_flags()
        # otherwise we first set the flags for what data was read and
        # then add to it only
        else:
            # initialize the flags from the read data
            self._update_exist_flags()

            # restrict append permissions for those that have their flags
            # set from the read init
            if self._wepy_mode == 'c-':
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

    def close(self):
        if not self.closed:
            self._h5.close()
            self.closed = True

    # TODO is this right? shouldn't we actually delete the data then close
    def __del__(self):
        self.close()

    @property
    def mode(self):
        return self._wepy_mode

    @property
    def h5_mode(self):
        return self._h5.mode


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
        if self._wepy_mode == 'c-':
            self._append_flags['topology'] = False

    @property
    def n_atoms(self):
        return self.positions.shape[1]

    def append_dataset(self, dataset_key, new_data):
        if self._wepy_mode == 'c-':
            assert self._append_flags[dataset_key], "dataset is not available for appending to"

        # get the dataset
        dset = self._h5[dataset_key]
        # append to the dataset on the first dimension, keeping the
        # others the same
        dset.resize( (dset.shape[0] + new_data.shape[0], *dset.shape[1:]) )
        # add the new data
        dset[-new_data.shape[0]:,:] = new_data

    def append_traj(self, **kwargs):
        """ append to the trajectory as a whole """
        assert 'positions' in kwargs.keys()
        pass

    @property
    def positions(self):

        return self._h5['positions']

    @positions.setter
    def positions(self, positions):
        assert isinstance(positions, np.ndarray), "positions must be a numpy array"

        # get the number of atoms
        n_atoms = positions.shape[1]

        self._h5.create_dataset('positions', data=positions, maxshape=(None, n_atoms, N_DIMS))
        self._exist_flags['positions'] = True
        # if we are in strict append mode we cannot append after we create something
        if self._wepy_mode == 'c-':
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
        self._h5.create_dataset('time', data=time, maxshape=(None))
        self._exist_flags['time'] = True
        # if we are in strict append mode we cannot append after we create something
        if self._wepy_mode == 'c-':
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
        self._h5.create_dataset('box_vectors', data=box_vectors,
                                maxshape=(None, N_DIMS, N_DIMS))
        self._exist_flags['box_vectors'] = True
        # if we are in strict append mode we cannot append after we create something
        if self._wepy_mode == 'c-':
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

        self._h5.create_dataset('velocities', data=velocities,
                                maxshape=(None, self.n_atoms, N_DIMS))
        self._exist_flags['velocities'] = True
        # if we are in strict append mode we cannot append after we create something
        if self._wepy_mode == 'c-':
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
        raise NotImplementedError

        # self._h5.create_dataset('forces', data=forces)
        # self._exist_flags['forces'] = True
        # # if we are in strict append mode we cannot append after we create something
        # if self._wepy_mode == 'c-':
        #     self._append_flags['forces'] = False

    @property
    def parameters(self):
        if self._exist_flags['parameters']:
            return self._h5['parameters']
        else:
            return None

    @parameters.setter
    def parameters(self, parameters):
        raise NotImplementedError

        # self._h5.create_dataset('parameters', data=parameters)
        # self._exist_flags['parameters'] = True
        # # if we are in strict append mode we cannot append after we create something
        # if self._wepy_mode == 'c-':
        #     self._append_flags['parameters'] = False

    @property
    def observables(self):
        if self._exist_flags['observables']:
            return self._h5['observables']
        else:
            return None

    @observables.setter
    def observables(self, observables):
        raise NotImplementedError

        # self._h5.create_dataset('observables', data=observables)
        # self._exist_flags['observables'] = True
        # # if we are in strict append mode we cannot append after we create something
        # if self._wepy_mode == 'c-':
        #     self._append_flags['observables'] = False


class WepyHDF5(object):

    def __init__(self, filename, mode='x', topology=None):
        """Initialize a new Wepy HDF5 file. This is a file that organizes
        wepy.TrajHDF5 dataset subsets by simulations by runs and
        includes resampling records for recovering walker histories.

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

        # counters for run and traj indexing
        self._run_idx_counter = 0
        # count the number of trajectories each run has
        self._run_traj_idx_counter = {}


        # all the keys for the top-level items in this class
        self._keys = ['topology', 'runs', 'resampling', 'boundary_conditions']

        # initialize the exist flags, which say whether a dataset
        # exists or not
        self._exist_flags = {key : False for key in self._keys}

        # initialize the append flags dictionary, this keeps track of
        # whether a data field can be appended to or not
        self._append_flags = {key : True for key in self._keys}


        # TODO Dataset Complianaces

        # create file mode: 'w' will create a new file or overwrite,
        # 'w-' and 'x' will not overwrite but will create a new file
        if self._wepy_mode in ['w', 'w-', 'x']:
            self._create_init(topology)

        # read/write mode: in this mode we do not completely overwrite
        # the old file and start again but rather write over top of
        # values if requested
        elif self._wepy_mode in ['r+']:
            self._read_write_init(topology)

        # add mode: read/write create if doesn't exist
        elif self._wepy_mode in ['a']:
            self._add_init(topology)

        # append mode
        elif self._wepy_mode in ['c', 'c-']:
            # use the hidden init function for appending data
            self._append_init(topology)

        # read only mode
        elif self._wepy_mode == 'r':
            # if any data was given, warn the user
            if topology is not None:
               warn("Cannot set topology on read only", RuntimeWarning)

            # then run the initialization process
            self._read_init()

        # TODO update the compliance type flags of the dataset

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

    def _create_init(self, topology):
        """Completely overwrite the data in the file. Reinitialize the values
        and set with the new ones if given."""

        assert topology is not None, "Topology must be given"

        # assign the topology
        self.topology = topology

    def _read_write_init(self, topology):
        """Write over values if given but do not reinitialize any old ones. """

        # set the flags for existing data
        self._update_exist_flags()

        # add new topology if it is given
        if topology is not None:
            self.topology = topology

    def _add_init(self, topology):
        """Create the dataset if it doesn't exist and put it in r+ mode,
        otherwise, just open in r+ mode."""

        # set the flags for existing data
        self._update_exist_flags()

        if not any(self._exist_flags):
            self._create_init(topology)
        else:
            self._read_write_init(topology)

    def _append_init(self, topology, data, units):
        """Append mode initialization. Checks for given data and sets flags,
        and adds new data if given."""

        # if the file has any data to start we write the data to it
        if not any(self._exist_flags):
            self.topology = topology
            if self._wepy_mode == 'c-':
                self._update_append_flags()
        # otherwise we first set the flags for what data was read and
        # then add to it only
        else:
            # initialize the flags from the read data
            self._update_exist_flags()

            # restrict append permissions for those that have their flags
            # set from the read init
            if self._wepy_mode == 'c-':
                self._update_append_flags()

    def _read_init(self):
        """Read only mode initialization. Simply checks for the presence of
        and sets attribute flags."""
        # we just need to set the flags for which data is present and
        # which is not
        self._update_exist_flags()



    @property
    def filename(self):
        return self._filename


    def close(self):
        if not self.closed:
            self._h5.close()
            self.closed = True

    # TODO is this right? shouldn't we actually delete the data then close
    def __del__(self):
        self.close()

    @property
    def mode(self):
        return self._wepy_mode

    @property
    def h5_mode(self):
        return self._h5.mode

    @property
    def h5(self):
        return self._h5

    @property
    def runs(self):
        return self._h5['runs']

    @property
    def n_runs(self):
        return len(self._h5['runs'])

    @property
    def run_idxs(self):
        return range(len(self._h5['runs']))


    def run(self, run_idx):
        return self._h5['runs/{}'.format(int(run_idx))]

    def traj(self, run_idx, traj_idx):
        return self._h5['runs/{}/{}'.format(run_idx, traj_idx)]

    def n_run_trajs(self, run_idx):
        return len(self._h5['runs/{}'.format(run_idx)])

    def run_traj_idxs(self, run_idx):
        return range(len(self._h5['runs/{}'.format(run_idx)]))

    @property
    def n_atoms(self):
        return self.positions.shape[1]

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
        if self._wepy_mode == 'c-':
            self._append_flags['topology'] = False

    def new_run(self, **kwargs):
        # create a new group named the next integer in the counter
        run_grp = self._h5.create_group('runs/{}'.format(str(self._run_idx_counter)))

        # initialize this run's counter for the number of trajectories
        self._run_traj_idx_counter[self._run_idx_counter] = 0

        # add the run idx as metadata in the run group
        self.attrs['run_idx'] = self._run_idx_counter

        # increment the run idx counter
        self._run_idx_counter += 1

        # add metadata if given
        for key, val in kwargs.items():
            if key != 'run_idx':
                run_grp.attrs[key] = val
            else:
                warn('run_idx metadata is set by wepy and cannot be used', RuntimeWarning)

        return run_grp

    def add_traj(self, run_idx, weights=None, **kwargs):

        # get the data from the kwargs related to making a trajectory
        traj_data = _extract_traj_dict(**kwargs)

        # positions are mandatory
        assert 'positions' in traj_data.keys(), "positions must be given to create a trajectory"
        assert isinstance(traj_data['positions'], np.ndarray)

        # if weights are None then we assume they are 1.0
        if weights is None:
            weights = np.ones_like(traj_data['positions'].shape[0], dtype=float)
        else:
            assert isinstance(weights, np.ndarray), "weights must be a numpy.ndarray"
            assert weights.shape[0] == traj_data['positions'].shape[0],\
                "weights and the number of frames must be the same length"

        # the rest is run-level metadata on the trajectory, not
        # details of the MD etc. which should be in the traj_data
        metadata = {}
        for key, value in kwargs.items():
            if not key in traj_data.keys():
                metadata[key] = value


        # make a group for this trajectory, with the current traj_idx
        # for this run
        traj_grp = self._h5.create_group(
                           'runs/{run_idx}/{traj_idx}'.format(run_idx=run_idx,
                                               traj_idx=self._run_traj_idx_counter[run_idx]))

        # add the run_idx as metadata
        traj_grp.attrs['run_idx'] = run_idx
        # add the traj_idx as metadata
        traj_grp.attrs['traj_idx'] = self._run_traj_idx_counter[run_idx]

        # add the rest of the metadata if given
        for key, val in metadata.items():
            if not key in ['run_idx', 'traj_idx']:
                traj_grp.attrs[key] = val
            else:
                warn("run_idx and traj_idx are used by wepy and cannot be set", RuntimeWarning)


        # increment the traj_idx_count for this run
        self._run_traj_idx_counter[run_idx] += 1

        n_atoms = traj_data['positions'].shape[1]
        # add datasets to the traj group

        # weights
        traj_grp.create_dataset('weights', data=weights, maxshape=(None))

        # positions
        traj_grp.create_dataset('positions', data=traj_data['positions'],
                                maxshape=(None, n_atoms, N_DIMS))
        # time
        try:
            time = traj_data['time']
        except KeyError:
            pass
        else:
            traj_grp.create_dataset('time', data=time, maxshape=(None))

        # box vectors
        try:
            box_vectors = traj_data['box_vectors']
        except KeyError:
            pass
        else:
            traj_grp.create_dataset('box_vectors', data=box_vectors,
                                    maxshape=(None, N_DIMS, N_DIMS))
        # velocities
        try:
            velocities = traj_data['velocities']
        except KeyError:
            pass
        else:
            traj_grp.create_dataset('velocities', data=velocities,
                                    maxshape=(None, n_atoms, N_DIMS))

        # TODO set other values, forces, parameters, and observables

        return traj_grp

    def append_traj(self, run_idx, traj_idx, weights=None, **kwargs):


        if self._wepy_mode == 'c-':
            assert self._append_flags[dataset_key], "dataset is not available for appending to"

        # add the weights
        dset = self._h5['runs/{}/{}/{}'.format(run_idx, traj_idx, 'weights')]
        # append to the dataset on the first dimension, keeping the
        # others the same
        dset.resize( (dset.shape[0] + new_data.shape[0], *dset.shape[1:]) )
        # add the new data
        dset[-new_data.shape[0]:,:] = new_data

        # get the dataset
        for key, value in kwargs.items():
            dset = self._h5['runs/{}/{}/{}'.format(run_idx, traj_idx, key)]
            # append to the dataset on the first dimension, keeping the
            # others the same
            dset.resize( (dset.shape[0] + new_data.shape[0], *dset.shape[1:]) )
            # add the new data
            dset[-new_data.shape[0]:,:] = new_data


    def add_resampling_records(self, run_idx):
        pass



def _extract_traj_dict(**kwargs):
    traj_data = {}
    for field in TRAJ_DATA_FIELDS:
        try:
            traj_data[field] = kwargs[field]
        except KeyError:
            pass

    return traj_data
