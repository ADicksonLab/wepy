from collections import Iterable
import json
from warnings import warn

import numpy as np

import h5py

# Constants
N_DIMS = 3
TRAJ_DATA_FIELDS = ['positions', 'time', 'box_vectors', 'velocities',
                    'parameters', 'forces', 'kinetic_energy', 'potential_energy',
                    'box_volume', 'parameters', 'parameter_derivatives', 'observables']
INSTRUCTION_TYPES = ['VARIABLE', 'FIXED']

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

        # open the file in a context and initialize
        with h5py.File(filename, mode=self._h5py_mode) as h5:
            self._h5 = h5


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
                if (any([False if (value is None) else True for key, value in data.items()]))\
                  or (any([False if (value is None) else True for key, value in units.items()])):
                    warn("Data was provided for a read-only operation", RuntimeWarning)

                # then run the initialization process
                self._read_init()

        # the file should be closed after this
        self.closed = True

        # update the compliance type flags of the dataset
        self._update_compliance_flags()

    # context manager methods

    def __enter__(self):
        self._h5 = h5py.File(self._filename)
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._h5.flush()
        self.close()


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
        assert 'positions' in kwargs
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


        ### WepyHDF5 specific variables

        # counters for run and traj indexing
        self._run_idx_counter = 0
        # count the number of trajectories each run has
        self._run_traj_idx_counter = {}

        # the counter for the cycle the records are on, incremented
        # each time the add_cycle_resampling_records is called
        self._current_resampling_rec_cycle = 0

        # for each run there will be a special mapping for the
        # instruction dtypes which we need for a single session,
        # stored here organized by run (int) as the key
        self.instruction_dtypes_tokens = {}


        ### HDF5 file wrapper specific variables

        # all the keys for the top-level items in this class
        self._keys = ['topology', 'runs', 'resampling', 'boundary_conditions']

        # initialize the exist flags, which say whether a dataset
        # exists or not
        self._exist_flags = {key : False for key in self._keys}

        # initialize the append flags dictionary, this keeps track of
        # whether a data field can be appended to or not
        self._append_flags = {key : True for key in self._keys}


        # TODO Dataset Complianaces

        # open the file
        with h5py.File(filename, mode=self._h5py_mode) as h5:
            self._h5 = h5

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

            self._h5.flush()

        # should be closed after initialization
        self.closed = False

        # TODO update the compliance type flags of the dataset

    # context manager methods

    def __enter__(self):
        self._h5 = h5py.File(self._filename)
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._h5.flush()
        self.close()

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
    def metadata(self):
        return dict(self._h5.attrs)

    def add_metadata(self, key, value):
        self._h5.attrs[key] = value

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
        return self._h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]

    def run_trajs(self, run_idx):
        return self._h5['runs/{}/trajectories'.format(run_idx)]

    def n_run_trajs(self, run_idx):
        return len(self._h5['runs/{}/trajectories'.format(run_idx)])

    def run_traj_idxs(self, run_idx):
        return range(len(self._h5['runs/{}/trajectories'.format(run_idx)]))

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
        self._h5['runs/{}'.format(self._run_idx_counter)].attrs['run_idx'] = self._run_idx_counter

        # increment the run idx counter
        self._run_idx_counter += 1

        # initialize the walkers group
        traj_grp = run_grp.create_group('trajectories')

        # add metadata if given
        for key, val in kwargs.items():
            if key != 'run_idx':
                run_grp.attrs[key] = val
            else:
                warn('run_idx metadata is set by wepy and cannot be used', RuntimeWarning)

        return run_grp

    def init_run_resampling(self, run_idx, decision_enum, instruction_dtypes_tokens):

        run_grp = self.run(run_idx)

        # save the instruction_dtypes_tokens in the object
        self.instruction_dtypes_tokens[run_idx] = instruction_dtypes_tokens

        # init a resampling group
        # initialize the resampling group
        res_grp = run_grp.create_group('resampling')
        # initialize the records and data groups
        rec_grp = res_grp.create_group('records')
        data_grp = res_grp.create_group('data')

        # save the decision as a mapping in it's own group
        decision_map = {decision.name : decision.value for decision in decision_enum}
        decision_grp = res_grp.create_group('decision')
        decision_enum_grp = decision_grp.create_group('enum')
        for decision in decision_enum:
            decision_enum_grp.create_dataset(decision.name, data=decision.value)

        # initialize a mapping for whether decision has a variable length instruction type
        is_variable_lengths = {decision.name : False for decision in decision_enum}

        # initialize the decision types as datasets, using the
        # instruction dtypes to set the size of them
        for decision in decision_enum:

            instruct_dtype_tokens = self.instruction_dtypes_tokens[run_idx][decision.name]
            # validate the instruction dtype for whether it is
            # variable or fixed length
            variable_length = _instruction_is_variable_length(instruct_dtype_tokens)
            # if it is fixed length we have to break up the decision
            # into multiple groups which are dynamically created if necessary
            if variable_length:
                # record this in the mapping that will be stored in the file
                is_variable_lengths[decision.name] = True

                # create a group for this decision
                instruct_grp = rec_grp.create_group(decision.name)

                # in the group we start a dataset that keeps track of
                # which width datasets have been initialized
                instruct_grp.create_dataset('_initialized', (0,), dtype=np.int, maxshape=(None,))

            # the datatype is simple and we can initialize the dataset now
            else:
                # we need to create a compound datatype
                # make a proper dtype out of the instruct_dtype tuple
                instruct_dtype = np.dtype(instruct_dtype_tokens)
                # then pass that to the instruction dtype maker
                dt = _make_numpy_instruction_dtype(instruct_dtype)
                # then make the group with that datatype, which can be extended
                rec_grp.create_dataset(decision.name, (0,), dtype=dt, maxshape=(None,))

        # another group in the decision group that keeps track of flags for variable lengths
        varlength_grp = decision_grp.create_group('variable_length')
        for decision_name, flag in is_variable_lengths.items():
            varlength_grp.create_dataset(decision_name, data=flag)

    def add_traj(self, run_idx, weights=None, **kwargs):

        # get the data from the kwargs related to making a trajectory
        traj_data = _extract_traj_dict(**kwargs)

        # positions are mandatory
        assert 'positions' in traj_data, "positions must be given to create a trajectory"
        assert isinstance(traj_data['positions'], np.ndarray)

        n_frames = traj_data['positions'].shape[0]

        # if weights are None then we assume they are 1.0
        if weights is None:
            weights = np.ones(n_frames, dtype=float)
        else:
            assert isinstance(weights, np.ndarray), "weights must be a numpy.ndarray"
            assert weights.shape[0] == n_frames,\
                "weights and the number of frames must be the same length"

        # the rest is run-level metadata on the trajectory, not
        # details of the MD etc. which should be in the traj_data
        metadata = {}
        for key, value in kwargs.items():
            if not key in traj_data:
                metadata[key] = value


        # make a group for this trajectory, with the current traj_idx
        # for this run
        traj_grp = self._h5.create_group(
                           'runs/{}/trajectories/{}'.format(run_idx, self._run_traj_idx_counter[run_idx]))

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
        traj_grp.create_dataset('weights', data=weights, maxshape=(None,))

        # positions
        traj_grp.create_dataset('positions', data=traj_data['positions'],
                                maxshape=(None, n_atoms, N_DIMS))
        # time
        try:
            time = traj_data['time']
        except KeyError:
            pass
        else:
            traj_grp.create_dataset('time', data=time, maxshape=(None,))

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

        # TODO update for multiple forces
        # forces
        try:
            forces = traj_data['forces']
        except KeyError:
            pass
        else:
            traj_grp.create_dataset('forces', data=forces,
                                    maxshape=(None, n_atoms, N_DIMS))

        # TODO set other values, forces, parameters, and observables

        # potential energy
        try:
            potential_energy = traj_data['potential_energy']
        except KeyError:
            pass
        else:
            traj_grp.create_dataset('potential_energy', data=potential_energy,
                                    maxshape=(None,))

        # kinetic energy
        try:
            kinetic_energy = traj_data['kinetic_energy']
        except KeyError:
            pass
        else:
            traj_grp.create_dataset('kinetic_energy', data=kinetic_energy,
                                    maxshape=(None,))

        # box volume
        try:
            box_volume = traj_data['box_volume']
        except KeyError:
            pass
        else:
            traj_grp.create_dataset('box_volume', data=box_volume,
                                    maxshape=(None,))

        # parameters
        try:
            parameters = traj_data['parameters']
        except KeyError:
            pass
        else:
            traj_grp.create_group('parameters')

        # parameter_derivatives
        try:
            parameter_derivatives = traj_data['parameter_derivatives']
        except KeyError:
            pass
        else:
            traj_grp.create_group('parameter_derivatives')

        return traj_grp

    def append_traj(self, run_idx, traj_idx, weights=None, **kwargs):

        if self._wepy_mode == 'c-':
            assert self._append_flags[dataset_key], "dataset is not available for appending to"

        # get trajectory data from the kwargs
        traj_data = _extract_traj_dict(**kwargs)

        # number of frames to add
        n_new_frames = traj_data['positions'].shape[0]

        # if weights are None then we assume they are 1.0
        if weights is None:
            weights = np.ones(n_new_frames, dtype=float)
        else:
            assert isinstance(weights, np.ndarray), "weights must be a numpy.ndarray"
            assert weights.shape[0] == n_new_frames,\
                "weights and the number of frames must be the same length"

        # get the trajectory group
        traj_grp = self._h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]

        # add the weights
        dset = traj_grp['weights']

        # append to the dataset on the first dimension, keeping the
        # others the same, if they exist
        if len(dset.shape) > 1:
            dset.resize( (dset.shape[0] + n_new_frames, *dset.shape[1:]) )
        else:
            dset.resize( (dset.shape[0] + n_new_frames, ) )

        # add the new data
        dset[-n_new_frames:, ...] = weights

        # get the dataset
        for key, value in kwargs.items():

            # if there is no value for a key just ignore it
            if value is None:
                continue

            # get the dataset handle
            dset = self._h5['runs/{}/trajectories/{}/{}'.format(run_idx, traj_idx, key)]

            # append to the dataset on the first dimension, keeping the
            # others the same, if they exist
            if len(dset.shape) > 1:
                dset.resize( (dset.shape[0] + n_new_frames, *dset.shape[1:]) )
            else:
                dset.resize( (dset.shape[0] + n_new_frames, ) )

            # add the new data
            dset[-n_new_frames:, ...] = value

    def add_cycle_resampling_records(self, run_idx, cycle_resampling_records):

        # get the run group
        run_grp = self.run(run_idx)
        # get the resampling group
        resampling_grp = run_grp['resampling']
        # records group
        rec_grp = resampling_grp['records']

        # the cycle you are on for cycle resampling records, this is
        # automatically updated
        cycle_idx = self._current_resampling_rec_cycle

        # the mapping of decision values to names
        decision_value_names = self.decision_value_names(run_idx)

        # go through each step
        for step_idx, step in enumerate(cycle_resampling_records):
            # for each decision and instruction record add the instruction record
            for walker_idx, resampling_rec in enumerate(step):
                # the value for the decision, (an int)
                decision_value = resampling_rec.decision
                # get the string name for this
                decision_key = decision_value_names[decision_value]
                # a tuple for the instruction record
                instruct_record = resampling_rec.instruction
                # add this record to the data
                self.add_instruction_record(run_idx, decision_key,
                                            cycle_idx, step_idx, walker_idx,
                                            instruct_record)

        # update the current cycle idx
        self._current_resampling_rec_cycle += 1

    def add_instruction_record(self, run_idx, decision_key,
                               cycle_idx, step_idx, walker_idx,
                               instruct_record):

        run_grp = self.run(run_idx)
        varlengths = self._instruction_varlength_flags(run_idx)

        # we need to treat variable length decision instructions
        # differently than fixed width

        # test whether the decision has a variable width instruction format
        if varlengths[decision_key][()]:
            self._add_varlength_instruction_record(run_idx, decision_key,
                                                   cycle_idx, step_idx, walker_idx,
                                                   instruct_record)
        else:
            self._add_fixed_width_instruction_record(run_idx, decision_key,
                                                     cycle_idx, step_idx, walker_idx,
                                                     instruct_record)

    def _add_varlength_instruction_record(self, run_idx, decision_key,
                                          cycle_idx, step_idx, walker_idx,
                                          instruct_record):
        # the isntruction record group
        instruct_grp = self._h5['runs/{}/resampling/records/{}'.format(run_idx, decision_key)]
        # the dataset of initialized datasets for different widths
        widths_dset = instruct_grp['_initialized']

        # the width of this record
        instruct_width = len(instruct_record)

        # see if we have a dataset with the width already intialized
        if not instruct_width in widths_dset[:]:
            # we don't have a dataset so create it
            # create the datatype for the dataset

            # we first need to make a proper dtype from the instruction record
            # fetch the instruct dtype tuple
            instruct_dtype_tokens = self.instruction_dtypes_tokens[run_idx][decision_key]
            dt = _make_numpy_varlength_instruction_dtype(instruct_dtype_tokens, instruct_width)

            # make the dataset with the given dtype
            dset = instruct_grp.create_dataset(str(len(instruct_record)), (0,),
                                               dtype=dt, maxshape=(None,))

            # record it in the initialized list
            widths_dset.resize( (widths_dset.shape[0] + 1, ) )
            widths_dset[-1] = instruct_width

        # if it exists get a reference to the dataset according to length
        else:
            dset = instruct_grp[str(len(instruct_record))]

        # make the complete record to add to the dataset
        record = (cycle_idx, step_idx, walker_idx, instruct_record)
        # add the record to the dataset
        self._append_instruct_records(dset, [record])


    def _add_fixed_width_instruction_record(self, run_idx, decision_key,
                                            cycle_idx, step_idx, walker_idx,
                                            instruct_record):
        # the decision dataset
        dset = self._h5['runs/{}/resampling/records/{}'.format(run_idx, decision_key)]
        # make an appropriate record
        record = (cycle_idx, step_idx, walker_idx, instruct_record)
        # add the record to the dataset
        self._append_instruct_records(dset, [record])

    def _append_instruct_records(self, dset, records):
        n_records = len(records)
        dset.resize( (dset.shape[0] + n_records, ) )
        records_arr = np.array(records, dtype=dset.dtype)
        dset[-n_records:] = records


    def resampling(self, run_idx):
        return self._h5['runs/{}/resampling'.format(run_idx)]

    def decision(self, run_idx):
        return self._h5['runs/{}/resampling/decision'.format(run_idx)]

    def decision_enum(self, run_idx):

        enum_grp = self._h5['runs/{}/resampling/decision/enum'.format(run_idx)]
        enum = {}
        for decision_name, dset in enum_grp.items():
            enum[decision_name] = dset

        return enum

    def decision_value_names(self, run_idx):
        enum_grp = self._h5['runs/{}/resampling/decision/enum'.format(run_idx)]
        rev_enum = {}
        for decision_name, dset in enum_grp.items():
            value = dset[()]
            rev_enum[value] = decision_name

        return rev_enum

    def resampling_records_grp(self, run_idx):
        return self._h5['runs/{}/resampling/records'.format(run_idx)]

    def _instruction_varlength_flags(self, run_idx):
        varlength_grp = self._h5['runs/{}/resampling/decision/variable_length'.format(run_idx)]
        varlength = {}
        for decision_name, dset in varlength_grp.items():
            varlength[decision_name] = dset

        return varlength



def _extract_traj_dict(**kwargs):
    traj_data = {}
    for field in TRAJ_DATA_FIELDS:
        try:
            traj_data[field] = kwargs[field]
        except KeyError:
            pass

    return traj_data



def _instruction_is_variable_length(instruction_dtype_tokens):

    # examine it for usage of Nones in field names of struct dtype tokens
    variable_length = False
    for i, token in enumerate(instruction_dtype_tokens):
        if token[0] is None:
            # check whether the None is in the right place, i.e. last position
            if i != len(instruction_dtype_tokens) - 1:
                raise TypeError("None (variable length) field must be the"
                                "last token in the instruction dtype dict")
            else:
                variable_length = True

    return variable_length


def _make_numpy_instruction_dtype(instruct_dtype):

    dtype_map = [('cycle_idx', np.int), ('step_idx', np.int), ('walker_idx', np.int),
                     ('instruction', instruct_dtype)]

    return np.dtype(dtype_map)

def _make_numpy_varlength_instruction_dtype(varlength_instruct_type, varlength_width):

    # replace the (None, type) token with several tokens from the
    # given length
    dtype_tokens = []
    for token in varlength_instruct_type:
        # if this is the None token
        if token[0] is None:
            # we replace it with multiple tokens of the type given
            # in the tokens tuple
            for i in range(varlength_width):
                dtype_tokens.append((str(i), token[1]))
        else:
            dtype_tokens.append(token)

    # make a numpy dtype from it
    instruct_record_dtype = np.dtype(dtype_tokens)

    # make a full instruction dtype from this and return it
    return _make_numpy_instruction_dtype(instruct_record_dtype)
