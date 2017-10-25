from collections import Iterable, Sequence
from types import GeneratorType

import json
from warnings import warn
import operator

import numpy as np

import h5py

# optional dependencies
try:
    import mdtraj as mdj
    import mdtraj.core.element as elem
except ModuleNotFoundError:
    warn("mdtraj is not installed and that functionality will not work", RuntimeWarning)

try:
    import pandas as pd
except ModuleNotFoundError:
    warn("pandas is not installed and that functionality will not work", RuntimeWarning)

# Constants
N_DIMS = 3

# lists of keys etc.
TRAJ_DATA_FIELDS = ('positions', 'time', 'box_vectors', 'velocities',
                    'forces', 'kinetic_energy', 'potential_energy',
                    'box_volume', 'parameters', 'parameter_derivatives', 'observables')

TRAJ_UNIT_FIELDS = ('positions_unit', 'time_unit', 'box_vectors_unit',
                    'velocities_unit',
                    'forces_unit',
                    'box_volume_unit', 'kinetic_energy_unit', 'potential_energy_unit',
                    'parameters_units', 'parameter_derivatives_units', 'observables_units')

DATA_UNIT_MAP = (('positions', 'positions_unit'),
                 ('time', 'time_unit'),
                 ('box_vectors', 'box_vectors_unit'),
                 ('velocities', 'velocities_unit'),
                 ('forces', 'forces_unit'),
                 ('box_volume', 'box_volume_unit'),
                 ('kinetic_energy', 'kinetic_energy_unit'),
                 ('potential_energy', 'potential_energy_unit'),
                 ('parameters', 'parameters_units'),
                 ('parameter_derivatives', 'parameter_derivatives_units'),
                 ('observables', 'observables_units')
                )

# some fields have more than one dataset associated with them
COMPOUND_DATA_FIELDS = ('parameters', 'parameter_derivatives', 'observables')
COMPOUND_UNIT_FIELDS = ('parameters', 'parameter_derivatives', 'observables')

# some fields can have sparse data, non-sparse data must be all the
# same shape and larger than sparse arrays. Sparse arrays have an
# associated index with them aligning them to the non-sparse datasets
SPARSE_DATA_FIELDS = ('velocities', 'forces', 'kinetic_energy', 'potential_energy',
                      'box_volume', 'parameters', 'parameter_derivatives', 'observables')

SPARSE_IDX_KEYS = ('velocities_idxs', 'forces_idxs',
                   'kinetic_energy_idxs', 'potential_energy_idxs',
                   'box_volume_idxs', 'parameters_idxs',
                   'parameter_derivatives_idxs', 'observables_idxs')

SPARSE_DATA_MAP = (('positions', 'positions_idxs'),
                 ('time', 'time_idxs'),
                 ('box_vectors', 'box_vectors_idxs'),
                 ('velocities', 'velocities_idxs'),
                 ('forces', 'forces_idxs'),
                 ('box_volume', 'box_volume_idxs'),
                 ('kinetic_energy', 'kinetic_energy_idxs'),
                 ('potential_energy', 'potential_energy_idxs'),
                 ('parameters', 'parameters_idxs'),
                 ('parameter_derivatives', 'parameter_derivatives_idxs'),
                 ('observables', 'observables_idxs')
                )

# decision instructions can be variable or fixed width
INSTRUCTION_TYPES = ('VARIABLE', 'FIXED')

## Dataset Compliances
# a file which has different levels of keys can be used for
# different things, so we define these collections of keys,
# and flags to keep track of which ones this dataset
# satisfies, ds here stands for "dataset"
COMPLIANCE_TAGS = ['COORDS', 'TRAJ', 'RESTART']

# the minimal requirement (and need for this class) is to associate
# a collection of coordinates to some molecular structure (topology)
COMPLIANCE_REQUIREMENTS = (('COORDS',  ('positions',)),
                           ('TRAJ',    ('positions', 'time', 'box_vectors')),
                           ('RESTART', ('positions', 'time', 'box_vectors',
                                        'velocities')),
                          )


class TrajHDF5(object):

    def __init__(self, filename, topology=None, mode='x', sparse_fields=None, **kwargs):
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


        # set the number of dimensions to use.
        # initialize to the 3D default
        self.n_dims = N_DIMS
        # but if a different one is given go with that
        if 'n_dims' in kwargs:
            assert isinstance(kwargs['n_dims'], int), "n_dims must be an integer"
            assert kwargs['n_dims'] > 0, "n_dims must be a positive integer"
            self.n_dims = kwargs['n_dims']



        # all the keys for the datasets and groups
        self._keys = ['topology', 'positions', 'velocities',
                      'box_vectors',
                      'time', 'box_volume', 'kinetic_energy', 'potential_energy',
                      'forces', 'parameters', 'parameter_derivatives', 'observables']

        # collect the non-topology attributes into a dict
        traj_data = _extract_dict(TRAJ_DATA_FIELDS, **kwargs)

        # units
        units = _extract_dict(TRAJ_UNIT_FIELDS, **kwargs)



        # initialize the sparse field flags to False
        self._sparse_field_flags = {}
        for key in TRAJ_DATA_FIELDS:
            # skip compound fields. They will get path-like keys
            # e.g. 'observables/observable1' when they are added
            if key not in COMPOUND_DATA_FIELDS:
                self._sparse_field_flags[key] = False

        # set the flags for sparse data that will be allowed in this
        # object from the sparse field flags or recognize it from the
        # idxs passed in

        # make a set of the defined sparse fields, if given
        sparse_fields = set([])
        # and a dictionary of them for the compound fields
        sparse_compound_fields = {key : set([]) for key in COMPOUND_DATA_FIELDS}
        if sparse_fields is not None:
            # get the sparse fields from the kwargs

            # figure out if it is compound or not, by looking for a
            # path separator
            for sparse_field in sparse_fields:
                if '/' in sparse_field:
                    # split the group and the field
                    grp_key, field_key = sparse_field.split()
                    # save the field to the compound group sparse fields
                    sparse_compound_fields[grp_key].update(field_key)

            # if there isn't one it is a simple field
            else:
                sparse_fields.update(set(sparse_fields))

        # go through the idxs given to the constructor for different
        # fields, and add them to the sparse fields list (if they
        # aren't already there) and make a dictionary of the indices
        # to set for initializing the arrays with data given to the
        # constructor.
        sparse_kw_idxs = _extract_dict(SPARSE_IDX_KEYS, **kwargs)
        # idxs for fields that are sparse given to the constructor
        sparse_field_idxs = {}
        if len(sparse_kw_idxs) > 0:
            # get the field keys, replace the kwarg keywords, this is the
            # form the functions that add data will understand
            sparse_kw_field_map = {kw : field for field, kw in SPARSE_DATA_MAP}
            for sparse_kw, sparse_obj in sparse_kw_idxs.items():
                # the field name
                sparse_field = sparse_kw_field_map[sparse_kw]
                # add the idxs for this field, depnding on if it is
                # compound or not

                # compound
                if sparse_field in COMPOUND_DATA_FIELDS:
                    grp_field = sparse_field

                    # initialize the group if not already done
                    if grp_field not in sparse_field_idxs:
                        sparse_field_idxs[grp_field] = {}

                    # go through each sub group
                    for sub_key, sparse_idxs in sparse_obj.items():
                        # add the idxs to the nested dictionary
                        sparse_field_idxs[grp_field][sub_key] = sparse_idxs
                        # make the field name pathy which will set the flags
                        sparse_field = '/'.join([grp_field, sub_key])
                        sparse_fields.update([sparse_field])
                # simple
                else:
                    sparse_idxs = sparse_obj
                    sparse_field_idxs[sparse_field] = sparse_idxs
                    # add this field to the set of sparse fields
                    sparse_fields.update([sparse_field])


        # set the flags for all of the sparse fields either specified
        # or given cycle indices for
        for key in sparse_fields:
            self._sparse_field_flags[key] = True


        # warn about unknown kwargs
        test_fields = list(TRAJ_DATA_FIELDS) + list(TRAJ_UNIT_FIELDS) + list(SPARSE_IDX_KEYS)
        for key in kwargs.keys():
            if not (key in test_fields):
                warn("kwarg {} not recognized and was ignored".format(key), RuntimeWarning)


        # append the exist flags
        self._exist_flags = {key : False for key in TRAJ_DATA_FIELDS}
        self._compound_exist_flags = {key : {} for key in COMPOUND_DATA_FIELDS}

        # initialize the append flags
        self._append_flags = {key : True for key in TRAJ_DATA_FIELDS}
        self._compound_append_flags = {key : {} for key in COMPOUND_DATA_FIELDS}

        # initialize the compliance types
        self._compliance_flags = {tag : False for tag, requirements in COMPLIANCE_REQUIREMENTS}

        # open the file in a context and initialize
        with h5py.File(filename, mode=self._h5py_mode) as h5:
            self._h5 = h5


            # create file mode: 'w' will create a new file or overwrite,
            # 'w-' and 'x' will not overwrite but will create a new file
            if self._wepy_mode in ['w', 'w-', 'x']:
                self._create_init(topology, traj_data, units, sparse_idxs=sparse_field_idxs)
                # once we have run the creation we change the mode for
                # opening h5py to read/write (non-creation) so that it
                # doesn't overwrite the WepyHDF5 object when we reopen
                # a WepyHDF5 which was constructed in a create
                # mode. We preserve the original mode given to
                # WepyHDF5 but just change the internals
                self._h5py_mode = 'r+'

            # read/write mode: in this mode we do not completely overwrite
            # the old file and start again but rather write over top of
            # values if requested
            elif self._wepy_mode in ['r+']:
                self._read_write_init()

            # add mode: read/write create if doesn't exist
            elif self._wepy_mode in ['a']:
                self._add_init(topology)

            # append mode
            elif self._wepy_mode in ['c', 'c-']:
                # use the hidden init function for appending data
                self._append_init()

            # read only mode
            elif self._wepy_mode == 'r':
                # then run the initialization process
                self._read_init()

        # the file should be closed after this
        self.closed = True

        # update the compliance type flags of the dataset
        self._update_compliance_flags()

    # context manager methods

    def __enter__(self):
        self._h5 = h5py.File(self._filename, mode=self._h5py_mode)
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


    ### The init functions for different I/O modes
    def _create_init(self, topology, data, units, sparse_idxs=None):
        """Completely overwrite the data in the file. Reinitialize the values
        and set with the new ones if given."""

        # make sure the mandatory data is here
        assert topology is not None, "Topology must be given"
        assert data['positions'] is not None, "positions must be given"
        assert units['positions_unit'] is not None, "positions unit must be given"

        # assign the topology
        self.topology = topology


        # positions
        positions = data.pop('positions')
        self.h5.create_dataset('positions', data=positions,
                                maxshape=(None, *positions.shape[1:]))

        # add data depending on whether it is compound or not
        for key, value in data.items():
            if key in COMPOUND_DATA_FIELDS:
                try:
                    field_sparse_idxs_dict = sparse_idxs[key]
                except KeyError:
                    self._add_compound_traj_data(key, value)
                else:
                    self._add_compound_traj_data(key, value, sparse_idxs=field_sparse_idxs_dict)
            else:
                try:
                    field_sparse_idxs = sparse_idxs[key]
                except KeyError:
                    self._add_traj_data(key, value)
                else:
                    self._add_traj_data(key, value, sparse_idxs=field_sparse_idxs)


        # initialize the units group
        unit_grp = self._h5.create_group('units')

        # initialize the compound unit groups
        for field in COMPOUND_UNIT_FIELDS:
            unit_grp.create_group(field)

        # make a mapping of the unit kwarg keys to the keys used in
        # datastructure and the COMPOUND  key lists
        unit_key_map = {unit_key : field for field, unit_key in DATA_UNIT_MAP}

        # set the units
        for unit_key, unit_value in units.items():

            # get the field name for the unit
            field = unit_key_map[unit_key]

            # ignore the field if not given
            if unit_value is None:
                continue

            # if the units are compound then set compound units
            if field in COMPOUND_UNIT_FIELDS:
                cmp_grp = unit_grp[field]

                # set all the units in the dict for this compound key
                for cmp_key, cmp_value in unit_value.items():
                    cmp_grp.create_dataset(cmp_key, data=cmp_value)

            # its a simple data type
            else:
                unit_grp.create_dataset(field, data=unit_value)

    def _read_write_init(self):
        """Write over values if given but do not reinitialize any old ones. """

        self._read_init()

    def _add_init(self):
        """Create the dataset if it doesn't exist and put it in r+ mode,
        otherwise, just open in r+ mode."""

        # set the flags for existing data
        self._update_exist_flags()

        if not any(self._exist_flags):
            self._create_init(topology, data, units)
        else:
            self._read_write_init(topology, data, units)

    def _append_init(self):

        """Append mode initialization. Checks for given data and sets flags,
        and adds new data if given."""

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


    def _add_compound_traj_data(self, key, data, sparse_idxs=None):

        # create a group for this group of datasets
        cmpd_grp = self.h5.create_group(key)
        # make a dataset for each dataset in this group
        for sub_key, values in data.items():
            if (self._sparse_field_flags['/'.join([key, sub_key])]):
                assert sparse_idxs[sub_key] is not None, \
                    "sparse_idxs must be given if this is a sparse field"
                # the sparse idxs for this nested field
                sparse_field_idxs = sparse_idxs[sub_key]

                # create the group for this sparse group
                sparse_grp = cmpd_grp.create_group(sub_key)

                # add the data to this group
                sparse_grp.create_dataset('data', data=values, maxshape=(None, *values.shape[1:]))
                # add the sparse idxs
                sparse_grp.create_dataset('_sparse_idxs', data=sparse_field_idxs, maxshape=(None,))
            else:
                cmpd_grp.create_dataset(sub_key, data=values,
                                        maxshape=(None, *values.shape[1:]))

    def _add_traj_data(self, key, data, sparse_idxs=None):

        # if it is a sparse dataset we need to add the data and add
        # the idxs in a group
        if (self._sparse_field_flags[key]):
            assert sparse_idxs is not None, "sparse_idxs must be given if this is a sparse field"
            sparse_grp = self.h5.create_group(key)
            # add the data to this group
            sparse_grp.create_dataset('data', data=data, maxshape=(None, *data.shape[1:]))
            # add the sparse idxs
            sparse_grp.create_dataset('_sparse_idxs', data=sparse_idxs, maxshape=(None,))

        else:
            # create the dataset
            self.h5.create_dataset(key, data=data, maxshape=(None, *data.shape[1:]))


    @property
    def filename(self):
        return self._filename

    def open(self):
        if self.closed:
            self._h5 = h5py.File(self._filename, mode=self._h5py_mode)
            self.closed = False
        else:
            raise IOError("This file is already open")

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
        for compliance_type in COMPLIANCE_TAGS:
            # check if compliance for this type is met
            result = self._check_compliance_keys(compliance_type)
            # set the flag
            self._compliance_flags[compliance_type] = result

    def _check_compliance_keys(self, compliance_type):
        """Checks whether the flags for the datasets have been set to True."""
        results = []
        compliance_requirements = dict(COMPLIANCE_REQUIREMENTS)
        # go through each required dataset for this compliance
        # requirements and see if they exist
        for dataset_key in compliance_requirements[compliance_type]:
            results.append(self._exist_flags[dataset_key])
        return all(results)

    @property
    def h5(self):
        return self._h5

    @property
    def topology(self):
        return self._h5['topology'][()]

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


    def extend(self, **kwargs):
        """ append to the trajectory as a whole """


        if self._wepy_mode == 'c-':
            assert self._append_flags[dataset_key], "dataset is not available for appending to"

        # get trajectory data from the kwargs
        traj_data = _extract_dict(TRAJ_DATA_FIELDS, **kwargs)

        # sparse data idxs
        sparse_data_idxs = _extract_dict(SPARSE_IDX_KEYS, **kwargs)

        # warn about unknown kwargs
        test_fields = list(TRAJ_DATA_FIELDS) + list(TRAJ_UNIT_FIELDS) + list(SPARSE_IDX_KEYS)
        for key in kwargs.keys():
            if not (key in test_fields):
                warn("kwarg {} not recognized and was ignored".format(key), RuntimeWarning)

        # assess compliance types
        compliance_tags = _check_data_compliance(traj_data)
        # assert we meet minimum compliance, unless extra fields are
        # sparse
        assert 'COORDS' in compliance_tags, \
            "Appended data must minimally be COORDS compliant"

        # TODO check other compliances for this dataset


        # if any sparse data idxs were given we need to handle this.
        # add the idxs to the _sparse_idxs for this data
        for sparse_key, sparse_idxs in sparse_data_idxs.items():
            # assert that this is a sparse field
            assert self._sparse_field_flags[sparse_key], \
                "{} is not a sparse field".format(sparse_key)

        # number of frames to add
        n_new_frames = traj_data['positions'].shape[0]

        # add trajectory data for each field

        # get the dataset/group
        for key, value in traj_data.items():

            # if there is no value for a key just ignore it
            if value is None:
                continue

            # figure out if it is a dataset or a group (compound data like observables)
            thing = self._h5[key]
            if isinstance(thing, h5py.Group):

                # it is a group
                group = thing

                # go through the fields given
                for dset_key, dset_value in value.items():
                    # get that dataset
                    dset = group[dset_key]

                    # append to the dataset on the first dimension, keeping the
                    # others the same, if they exist
                    if len(dset.shape) > 1:
                        dset.resize( (dset.shape[0] + n_new_frames, *dset.shape[1:]) )
                    else:
                        dset.resize( (dset.shape[0] + n_new_frames, ) )

                    # add the new data
                    dset[-n_new_frames:, ...] = dset_value
            else:
                # it is just a dataset
                dset = thing

                # append to the dataset on the first dimension, keeping the
                # others the same, if they exist
                if len(dset.shape) > 1:
                    dset.resize( (dset.shape[0] + n_new_frames, *dset.shape[1:]) )
                else:
                    dset.resize( (dset.shape[0] + n_new_frames, ) )

                # add the new data
                dset[-n_new_frames:, ...] = value

    @property
    def positions(self):

        return self._h5['positions']

    @property
    def time(self):
        if self._exist_flags['time']:
            return self._h5['time']
        else:
            return None
    @property
    def box_volume(self):
        if self._exist_flags['box_volume']:
            return self._h5['box_volume']
        else:
            return None
    @property
    def kinetic_energy(self):
        if self._exist_flags['kinetic_energy']:
            return self._h5['kinetic_energy']
        else:
            return None

    @property
    def potential_energy(self):
        if self._exist_flags['potential_energy']:
            return self._h5['potential_energy']
        else:
            return None

    @property
    def box_vectors(self):
        if self._exist_flags['box_vectors']:
            return self._h5['box_vectors']
        else:
            return None

    @property
    def velocities(self):
        if self._exist_flags['velocities']:
            return self._h5['velocities']
        else:
            return None


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

    @property
    def parameters(self):
        if self._exist_flags['parameters']:
            return self._h5['parameters']
        else:
            return None


    @property
    def parameter_derivatives(self):
        if self._exist_flags['parameter_derivatives']:
            return self._h5['parameter_derivatives']
        else:
            return None


    @property
    def observables(self):
        if self._exist_flags['observables']:
            return self._h5['observables']
        else:
            return None

    def to_mdtraj(self):

        topology = _json_to_mdtraj_topology(self.topology)

        positions = self.h5['positions'][:]
        # reshape for putting into mdtraj
        time = self.h5['time'][:, 0]
        box_vectors = self.h5['box_vectors'][:]
        unitcell_lengths, unitcell_angles = _box_vectors_to_lengths_angles(box_vectors)

        traj = mdj.Trajectory(positions, topology,
                       time=time,
                       unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)

        return traj


class WepyHDF5(object):

    def __init__(self, filename, mode='x', topology=None, **kwargs):
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

        # set the number of dimensions to use
        self.n_dims = N_DIMS
        if 'n_dims' in kwargs:
            assert isinstance(kwargs['n_dims'], int), "n_dims must be an integer"
            assert kwargs['n_dims'] > 0, "n_dims must be a positive integer"
            self.n_dims = kwargs['n_dims']

        ### WepyHDF5 specific variables

        # units
        units = _extract_dict(TRAJ_UNIT_FIELDS, **kwargs)

        # warn about unknown kwargs
        for key in kwargs.keys():
            if not (key in TRAJ_DATA_FIELDS) and not (key in TRAJ_UNIT_FIELDS):
                warn("kwarg {} not recognized and was ignored".format(key), RuntimeWarning)

        # counter for the new runs, specific constructors will update
        # this if needed
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

        # the dtypes for the resampling auxiliary data
        self.resampling_aux_dtypes = {}
        self.resampling_aux_shapes = {}
        # whether or not the auxiliary datasets have been initialized
        self._resampling_aux_init = []

        # initialize the attribute for the dtype for the boundary
        # conditions warp records
        self.warp_dtype = None

        # the dtypes for warping auxiliary data
        self.warp_aux_dtypes = {}
        self.warp_aux_shapes = {}
        # whether or not the auxiliary datasets have been initialized
        self._warp_aux_init = []

        # the dtypes for the boundary conditions auxiliary data
        self.bc_aux_dtypes = {}
        self.bc_aux_shapes = {}
        # whether or not the auxiliary datasets have been initialized
        self._bc_aux_init = []

        ### HDF5 file wrapper specific variables

        # all the keys for the top-level items in this class
        self._keys = ['topology', 'runs', 'resampling', 'warping']

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
                self._create_init(topology, units)

            # read/write mode: in this mode we do not completely overwrite
            # the old file and start again but rather write over top of
            # values if requested
            elif self._wepy_mode in ['r+']:
                self._read_write_init()

            # add mode: read/write create if doesn't exist
            elif self._wepy_mode in ['a']:
                self._add_init(topology, units)

            # append mode
            elif self._wepy_mode in ['c', 'c-']:
                # use the hidden init function for appending data
                self._append_init()

            # read only mode
            elif self._wepy_mode == 'r':
                # if any data was given, warn the user
                if topology is not None:
                   warn("Cannot set topology on read only", RuntimeWarning)

                # then run the initialization process
                self._read_init()

            self._h5.flush()

            # set the h5py mode to the value in the actual h5py.File
            # object after creation
            self._h5py_mode = self._h5.mode


        # should be closed after initialization unless it is read and/or readwrite
        self.closed = True


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

    def _create_init(self, topology, units):
        """Completely overwrite the data in the file. Reinitialize the values
        and set with the new ones if given."""

        assert topology is not None, "Topology must be given"

        # assign the topology
        self.topology = topology

        # initialize the units group
        unit_grp = self._h5.create_group('units')

        # initialize the compound unit groups
        for field in COMPOUND_UNIT_FIELDS:
            unit_grp.create_group(field)

        # make a mapping of the unit kwarg keys to the keys used in
        # datastructure and the COMPOUND  key lists
        unit_key_map = {unit_key : field for field, unit_key in DATA_UNIT_MAP}

        # set the units
        for unit_key, unit_value in units.items():

            # get the field name for the unit
            field = unit_key_map[unit_key]

            # ignore the field if not given
            if unit_value is None:
                continue

            # if the units are compound then set compound units
            if field in COMPOUND_UNIT_FIELDS:
                # get the unit group
                cmp_grp = unit_grp[field]
                # set all the units in the dict for this compound key
                for cmp_key, cmp_value in unit_value.items():
                    cmp_grp.create_dataset(cmp_key, data=cmp_value)

            # its a simple data type
            else:
                unit_grp.create_dataset(field, data=unit_value)

    def _read_write_init(self):
        """Write over values if given but do not reinitialize any old ones. """

        self._read_init()

        # set the counter for runs based on the groups already present
        for run_grp in enumerate(self.h5['runs']):
            self._run_idx_counter += 1

    def _add_init(self, topology, units):
        """Create the dataset if it doesn't exist and put it in r+ mode,
        otherwise, just open in r+ mode."""

        # set the flags for existing data
        self._update_exist_flags()

        if not any(self._exist_flags):
            self._create_init(topology)
        else:
            self._read_write_init()

    def _append_init(self):
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

    def open(self):
        if self.closed:
            self._h5 = h5py.File(self._filename, self._h5py_mode)
            self.closed = False
        else:
            raise IOError("This file is already open")

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
    def run_keys(self):
        return self._h5['runs']

    @property
    def runs(self):
        return self.h5['runs'].values()

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

    def run_traj_idx_tuples(self):
        tups = []
        for run_idx in self.run_idxs:
            for traj_idx in self.run_traj_idxs(run_idx):
                tups.append((run_idx, traj_idx))

        return tups

    @property
    def n_trajs(self):
        return len(list(self.run_traj_idx_tuples()))


    @property
    def n_atoms(self):
        return self.positions.shape[1]

    @property
    def topology(self):
        return self._h5['topology'][()]

    @topology.setter
    def topology(self, topology):
        try:
            json_d = json.loads(topology)
            del json_d
        except json.JSONDecodeError:
            raise ValueError("topology must be a valid JSON string")

        # check to see if this is the initial setting of it
        if not self._exist_flags['topology']:
            self._h5.create_dataset('topology', data=topology)
            self._exist_flags['topology'] = True
            # if we are in strict append mode we cannot append after we create something
            if self._wepy_mode == 'c-':
                self._append_flags['topology'] = False

        # if not replace the old one if we are in a non-concatenate write mode
        elif self._wepy_mode in ['w', 'r+', 'x', 'w-', 'a']:
            self._h5['topology'][()] = topology
        else:
            raise IOError("In mode {} and cannot modify topology".format(self._wepy_mode))


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

    def init_run_resampling(self, run_idx, decision_enum, instruction_dtypes_tokens,
                            resampling_aux_dtypes=None, resampling_aux_shapes=None):

        run_grp = self.run(run_idx)

        # save the instruction_dtypes_tokens in the object
        self.instruction_dtypes_tokens[run_idx] = instruction_dtypes_tokens

        # init a resampling group
        # initialize the resampling group
        res_grp = run_grp.create_group('resampling')
        # initialize the records and data groups
        rec_grp = res_grp.create_group('records')


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


        # initialize the auxiliary data group
        aux_grp = res_grp.create_group('aux_data')
        # if dtypes and shapes for aux dtypes are given create them,
        # if only one is given raise an error, if neither are given
        # just ignore this, it can be set when the first batch of aux
        # data is given, which will have a dtype and shape already given
        if (resampling_aux_dtypes is not None) and (resampling_aux_shapes is not None):
            # initialize the dtype and shapes for them
            self.resampling_aux_dtypes = {}
            self.resampling_aux_shapes = {}
            for key, dtype in resampling_aux_dtypes.items():
                # the shape of the dataset is needed too, for setting the maxshape
                shape = resampling_aux_shapes[key]
                # set them in the attributes
                self.resampling_aux_dtypes[key] = dtype
                self.resampling_aux_shapes[key] = shape
                # create the dataset with nothing in it
                aux_grp.create_dataset(key, (0, *[0 for i in shape]), dtype=dtype,
                                       maxshape=(None, *shape))

            # set the flags for these as initialized
            self._resampling_aux_init.extend(resampling_aux_dtypes.keys())

        elif (resampling_aux_dtypes is None) and (resampling_aux_shapes is None):
            # if they are both not given just ignore this part
            pass
        else:
            if resampling_aux_dtypes is None:
                raise ValueError("shapes were given but not dtypes")
            else:
                raise ValueError("dtypes were given but not shapes")


    def init_run_warp(self, run_idx, warp_dtype, warp_aux_dtypes=None, warp_aux_shapes=None):

        run_grp = self.run(run_idx)

        # save the dtype
        self.warp_dtype = warp_dtype

        # initialize the groups
        warp_grp = run_grp.create_group('warping')

        # the records themselves are a dataset so we make the full
        # dtype with the cycle and walker idx
        dt = _make_numpy_warp_dtype(self.warp_dtype)
        # make the dataset to be resizable
        rec_dset = warp_grp.create_dataset('records', (0,), dtype=dt, maxshape=(None,))

        # initialize the auxiliary data group
        # the data group can hold compatible numpy arrays by key
        aux_grp = warp_grp.create_group('aux_data')

        # if dtypes and shapes for aux dtypes are given create them,
        # if only one is given raise an error, if neither are given
        # just ignore this, it can be set when the first batch of aux
        # data is given, which will have a dtype and shape already given
        if (warp_aux_dtypes is not None) and (warp_aux_shapes is not None):
            # initialize the dtype and shapes for them
            self.warp_aux_dtypes = {}
            self.warp_aux_shapes = {}
            for key, dtype in warp_aux_dtypes.items():
                # the shape of the dataset is needed too, for setting the maxshape
                shape = warp_aux_shapes[key]
                # set them in the attributes
                self.warp_aux_dtypes[key] = dtype
                self.warp_aux_shapes[key] = shape
                # create the dataset
                aux_grp.create_dataset(key, (0, *[0 for i in shape]), dtype=dtype,
                                           maxshape=(None, *shape))

            # set the flags for these as initialized
            self._warp_aux_init.extend(warp_aux_dtypes.keys())

        elif (warp_aux_dtypes is None) and (warp_aux_shapes is None):
            # if they are both not given just ignore this part
            pass
        else:
            if warp_aux_dtypes is None:
                raise ValueError("shapes were given but not dtypes")
            else:
                raise ValueError("dtypes were given but not shapes")

    def init_run_bc(self, run_idx, bc_aux_dtypes=None, bc_aux_shapes=None):

        run_grp = self.run(run_idx)

        # initialize the groups
        bc_grp = run_grp.create_group('boundary_conditions')

        # TODO no records yet, this would be for storing data about
        # the boundary conditions themselves, i.e. the starting data
        # or dynamic BCs

        # initialize the auxiliary data group
        # the data group can hold compatible numpy arrays by key
        aux_grp = bc_grp.create_group('aux_data')
        # if dtypes and shapes for aux dtypes are given create them,
        # if only one is given raise an error, if neither are given
        # just ignore this, it can be set when the first batch of aux
        # data is given, which will have a dtype and shape already given
        if (bc_aux_dtypes is not None) and (bc_aux_shapes is not None):
            # initialize the dtype and shapes for them
            self.bc_aux_dtypes = {}
            self.bc_aux_shapes = {}
            for key, dtype in bc_aux_dtypes.items():
                # the shape of the dataset is needed too, for setting the maxshape
                shape = bc_aux_shapes[key]
                # set them in the attributes
                self.bc_aux_dtypes[key] = dtype
                self.bc_aux_shapes[key] = shape
                # create the dataset
                aux_grp.create_dataset(key, (0, *[0 for i in shape]), dtype=dtype,
                                           maxshape=(None, *shape))

            # set the flags for these as initialized
            self._bc_aux_init.extend(bc_aux_dtypes.keys())


        elif (bc_aux_dtypes is None) and (bc_aux_shapes is None):
            # if they are both not given just ignore this part
            pass
        else:
            if bc_aux_dtypes is None:
                raise ValueError("shapes were given but not dtypes")
            else:
                raise ValueError("dtypes were given but not shapes")

    def add_traj(self, run_idx, weights=None, **kwargs):

        # get the data from the kwargs related to making a trajectory
        traj_data = _extract_dict(TRAJ_DATA_FIELDS, **kwargs)

        # warn about unknown kwargs
        for key in kwargs.keys():
            if not (key in TRAJ_DATA_FIELDS) and not (key in TRAJ_UNIT_FIELDS):
                warn("kwarg {} not recognized and was ignored".format(key), RuntimeWarning)


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


        # current traj_idx
        traj_idx = self._run_traj_idx_counter[run_idx]
        # make a group for this trajectory, with the current traj_idx
        # for this run
        traj_grp = self._h5.create_group(
                        'runs/{}/trajectories/{}'.format(run_idx, traj_idx))

        # add the run_idx as metadata
        traj_grp.attrs['run_idx'] = run_idx
        # add the traj_idx as metadata
        traj_grp.attrs['traj_idx'] = traj_idx

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
        traj_grp.create_dataset('positions', data=traj_data.pop('positions'),
                                maxshape=(None, n_atoms, self.n_dims))

        # add data depending whether it is compound or not
        for key, value in traj_data.items():
            if key in COMPOUND_DATA_FIELDS:
                self._add_compound_traj_data(run_idx, traj_idx, key, value)
            else:
                self._add_traj_data(run_idx, traj_idx, key, value)


        return traj_grp

    def _add_traj_data(self, run_idx, traj_idx, key, data):

        # get the traj group
        traj_grp = self._h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]
        # create the dataset
        traj_grp.create_dataset(key, data=data, maxshape=(None, *data.shape[1:]))

    def _add_compound_traj_data(self, run_idx, traj_idx, key, data):

        # get the traj group
        traj_grp = self._h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]

        # create a group for this group of datasets
        cmpd_grp = traj_grp.create_group(key)
        # make a dataset for each dataset in this group
        for key, values in data.items():
            cmpd_grp.create_dataset(key, data=values,
                                    maxshape=(None, *values.shape[1:]))


    def extend_traj(self, run_idx, traj_idx, weights=None, **kwargs):

        if self._wepy_mode == 'c-':
            assert self._append_flags[dataset_key], "dataset is not available for appending to"

        # get trajectory data from the kwargs
        traj_data = _extract_dict(TRAJ_DATA_FIELDS, **kwargs)

        # warn about unknown kwargs
        for key in kwargs.keys():
            if not (key in TRAJ_DATA_FIELDS) and not (key in TRAJ_UNIT_FIELDS):
                warn("kwarg {} not recognized and was ignored".format(key), RuntimeWarning)

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

        # get the dataset/group
        for key, value in traj_data.items():

            # if there is no value for a key just ignore it
            if value is None:
                continue

            # figure out if it is a dataset or a group (compound data like observables)
            thing = self._h5['runs/{}/trajectories/{}/{}'.format(run_idx, traj_idx, key)]
            if isinstance(thing, h5py.Group):
                # it is a group
                group = thing

                # go through the fields given
                for dset_key, dset_value in value.items():
                    # get that dataset
                    dset = group[dset_key]

                    # append to the dataset on the first dimension, keeping the
                    # others the same, if they exist
                    if len(dset.shape) > 1:
                        dset.resize( (dset.shape[0] + n_new_frames, *dset.shape[1:]) )
                    else:
                        dset.resize( (dset.shape[0] + n_new_frames, ) )

                    # add the new data
                    dset[-n_new_frames:, ...] = dset_value
            else:
                # it is just a dataset
                dset = thing

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


    def add_cycle_resampling_aux_data(self, run_idx, resampling_aux_data):

        data_grp = self._h5['runs/{}/resampling/aux_data'.format(run_idx)]

        # if the datasets were initialized just add the new data
        for key, aux_data in resampling_aux_data.items():

            if key in self._resampling_aux_init:

                # get the dataset
                dset = data_grp[key]
                # add the new data
                dset.resize( (dset.shape[0] + 1, *aux_data.shape) )
                dset[-1] = np.array([aux_data])

            # if the datasets were not initialized initialize them with
            # the incoming dataset
            else:
                for key, aux_data in resampling_aux_data.items():
                    data_grp.create_dataset(key, data=np.array([aux_data]), dtype=aux_data.dtype,
                                           maxshape=(None, *aux_data.shape))
                    # save the dtype and shape
                    self.resampling_aux_dtypes[key] = aux_data.dtype
                    self.resampling_aux_shapes[key] = aux_data.shape
                    self._resampling_aux_init.append(key)

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

    def add_cycle_warp_records(self, run_idx, warp_records):

        rec_dset = self._h5['runs/{}/warping/records'.format(run_idx)]
        cycle_idx = self._current_resampling_rec_cycle

        # make the records for what is stored in the dset
        cycle_records = [(cycle_idx, *warp_record) for warp_record in warp_records]

        # add them to the dset
        self._append_instruct_records(rec_dset, cycle_records)

    def add_cycle_warp_aux_data(self, run_idx, warp_aux_data):

        data_grp = self._h5['runs/{}/warping/aux_data'.format(run_idx)]

        # if the datasets were initialized just add the new data
        for key, aux_data in warp_aux_data.items():

            if key in self._warp_aux_init:

                # get the dataset
                dset = data_grp[key]
                # add the new data, this is not on a cycle basis like
                # bc or resampling and is lined up with the warp
                # records along their first axis, so we resize for all
                # new additions
                dset.resize( (dset.shape[0] + aux_data.shape[0], *aux_data.shape[1:]) )
                dset[-aux_data.shape[0]:] = aux_data

            # if the datasets were not initialized initialize them with
            # the incoming dataset
            else:
                for key, aux_data in warp_aux_data.items():
                    data_grp.create_dataset(key, data=np.array([aux_data]), dtype=aux_data.dtype,
                                           maxshape=(None, *aux_data.shape))
                    # save the dtype and shape
                    self.warp_aux_dtypes[key] = aux_data.dtype
                    self.warp_aux_shapes[key] = aux_data.shape
                    self._warp_aux_init.append(key)

    def add_cycle_bc_aux_data(self, run_idx, bc_aux_data):

        data_grp = self._h5['runs/{}/boundary_conditions/aux_data'.format(run_idx)]

        # if the datasets were initialized just add the new data
        for key, aux_data in bc_aux_data.items():

            if key in self._bc_aux_init:

                # get the dataset
                dset = data_grp[key]
                # add the new data
                dset.resize( (dset.shape[0] + 1, *aux_data.shape) )
                dset[-1] = np.array([aux_data])

            # if the datasets were not initialized initialize them with
            # the incoming dataset
            else:
                for key, aux_data in bc_aux_data.items():
                    data_grp.create_dataset(key, data=np.array([aux_data]), dtype=aux_data.dtype,
                                           maxshape=(None, *aux_data.shape))
                    # save the dtype and shape
                    self.bc_aux_dtypes[key] = aux_data.dtype
                    self.bc_aux_shapes[key] = aux_data.shape
                    self._bc_aux_init.append(key)

    def iter_runs(self, idxs=False, run_sel=None):
        """Iterate through runs.

        idxs : if True returns `(run_idx, run_group)`, False just `run_group`

        run_sel : if True will iterate over a subset of runs. Possible
        values are an iterable of indices of runs to iterate over.

        """

        if run_sel is None:
            run_sel = self.run_idxs

        for run_idx in self.run_idxs:
            if run_idx in run_sel:
                run = self.run(run_idx)
                if idxs:
                    yield run_idx, run
                else:
                    yield run

    def iter_trajs(self, idxs=False, traj_sel=None):
        """Generator for all of the trajectories in the dataset across all
        runs. If idxs=True will return a tuple of (run_idx, traj_idx).

        run_sel : if True will iterate over a subset of
        trajectories. Possible values are an iterable of `(run_idx,
        traj_idx)` tuples.

        """


        # set the selection of trajectories to iterate over
        if traj_sel is None:
            idx_tups = self.run_traj_idx_tuples()
        else:
            idx_tups = traj_sel

        # get each traj for each idx_tup and yield them for the generator
        for run_idx, traj_idx in idx_tups:
            traj = self.traj(run_idx, traj_idx)
            if idxs:
                yield (run_idx, traj_idx), traj
            else:
                yield traj

    def iter_trajs_fields(self, fields, idxs=False, traj_sel=None, debug_prints=False):
        """Generator for all of the specified non-compound fields
        h5py.Datasets for all trajectories in the dataset across all
        runs. Fields is a list of valid relative paths to datasets in
        the trajectory groups.

        """

        for idx_tup, traj in self.iter_trajs(idxs=True, traj_sel=traj_sel):
            run_idx, traj_idx = idx_tup

            dsets = {}

            # DEBUG if we ask for debug prints send in the run and
            # traj index so the function can print this out
            if debug_prints:
                dsets['run_idx'] = run_idx
                dsets['traj_idx'] = traj_idx

            for field in fields:
                try:
                    dset = traj[field][:]
                except KeyError:
                    warn("field \"{}\" not found in \"{}\"".format(field, traj.name), RuntimeWarning)
                    dset = None

                dsets[field] = dset

            if idxs:
                yield (run_idx, traj_idx), dsets
            else:
                yield dsets


    def run_map(self, func, *args, map_func=map, idxs=False, run_sel=None):
        """Function for mapping work onto trajectories in the WepyHDF5 file
           object. The call to iter_runs is run with `idxs=False`.

        func : the function that will be mapped to trajectory groups

        map_func : the function that maps the function. This is where
                        parallelization occurs if desired.  Defaults to
                        the serial python map function.

        traj_sel : a trajectory selection. This is a valid `traj_sel`
        argument for the `iter_trajs` function.

        idxs : if True results contain [(run_idx, result),...], if False
        returns [result,...]

        *args : additional arguments to the function. If this is an
                 iterable it will be assumed that it is the appropriate
                 length for the number of trajectories, WARNING: this will
                 not be checked and could result in a run time
                 error. Otherwise single values will be automatically
                 mapped to all trajectories.

        **kwargs : same as *args, but will pass all kwargs to the func.

        """

        # check the args and kwargs to see if they need expanded for
        # mapping inputs
        mapped_args = []
        for arg in args:
            # if it is a sequence or generator we keep just pass it to the mapper
            if isinstance(arg, Sequence) and not isinstance(arg, str):
                assert len(arg) == self.n_runs, \
                    "argument Sequence has fewer number of args then trajectories"
                mapped_args.append(arg)
            # if it is not a sequence or generator we make a generator out
            # of it to map as inputs
            else:
                mapped_arg = (arg for i in range(self.n_runs))
                mapped_args.append(mapped_arg)


        results = map_func(func, self.iter_runs(idxs=False, run_sel=run_sel),
                           *mapped_args)

        if idxs:
            if run_sel is None:
                run_sel = self.run_idxs
            return zip(run_sel, results)
        else:
            return results


    def traj_map(self, func, *args, map_func=map, idxs=False, traj_sel=None):
        """Function for mapping work onto trajectories in the WepyHDF5 file object.

        func : the function that will be mapped to trajectory groups

        map_func : the function that maps the function. This is where
                        parallelization occurs if desired.  Defaults to
                        the serial python map function.

        traj_sel : a trajectory selection. This is a valid `traj_sel`
        argument for the `iter_trajs` function.

        *args : additional arguments to the function. If this is an
                 iterable it will be assumed that it is the appropriate
                 length for the number of trajectories, WARNING: this will
                 not be checked and could result in a run time
                 error. Otherwise single values will be automatically
                 mapped to all trajectories.

        """

        # check the args and kwargs to see if they need expanded for
        # mapping inputs
        mapped_args = []
        for arg in args:
            # if it is a sequence or generator we keep just pass it to the mapper
            if isinstance(arg, Sequence) and not isinstance(arg, str):
                assert len(arg) == self.n_trajs, "Sequence has fewer"
                mapped_args.append(arg)
            # if it is not a sequence or generator we make a generator out
            # of it to map as inputs
            else:
                mapped_arg = (arg for i in range(self.n_trajs))
                mapped_args.append(mapped_arg)

        results = map_func(func, self.iter_trajs(traj_sel=traj_sel), *mapped_args)

        if idxs:
            if traj_sel is None:
                traj_sel = self.run_traj_idx_tuples()
            return zip(traj_sel, results)
        else:
            return results

    def traj_fields_map(self, func, fields, *args, map_func=map, idxs=False, traj_sel=None,
                        debug_prints=False):
        """Function for mapping work onto field of trajectories in the
        WepyHDF5 file object. Similar to traj_map, except `h5py.Group`
        objects cannot be pickled for message passing. So we select
        the fields to serialize instead and pass the `numpy.ndarray`s
        to have the work mapped to them.

        func : the function that will be mapped to trajectory groups

        fields : list of fields that will be serialized into a dictionary
                 and passed to the map function. These must be valid
                 `h5py` path strings relative to the trajectory
                 group. These include the standard fields like
                 'positions' and 'weights', as well as compound paths
                 e.g. 'observables/sasa'.

        map_func : the function that maps the function. This is where
                        parallelization occurs if desired.  Defaults to
                        the serial python map function.

        traj_sel : a trajectory selection. This is a valid `traj_sel`
        argument for the `iter_trajs` function.

        *args : additional arguments to the function. If this is an
                 iterable it will be assumed that it is the appropriate
                 length for the number of trajectories, WARNING: this will
                 not be checked and could result in a run time
                 error. Otherwise single values will be automatically
                 mapped to all trajectories.

        """

        # check the args and kwargs to see if they need expanded for
        # mapping inputs
        mapped_args = []
        for arg in args:
            # if it is a sequence or generator we keep just pass it to the mapper
            if isinstance(arg, Sequence) and not isinstance(arg, str):
                assert len(arg) == self.n_trajs, "Sequence has fewer"
                mapped_args.append(arg)
            # if it is not a sequence or generator we make a generator out
            # of it to map as inputs
            else:
                mapped_arg = (arg for i in range(self.n_trajs))
                mapped_args.append(mapped_arg)

        results = map_func(func, self.iter_trajs_fields(fields, traj_sel=traj_sel, idxs=False,
                                                        debug_prints=debug_prints),
                           *mapped_args)

        if idxs:
            if traj_sel is None:
                traj_sel = self.run_traj_idx_tuples()
            return zip(traj_sel, results)
        else:
            return results


    def compute_observable(self, func, fields, *args,
                           map_func=map, traj_sel=None, save_to_hdf5=None, idxs=False,
                           debug_prints=False):
        """Compute an observable on the trajectory data according to a
        function. Optionally save that data in the observables data group for
        the trajectory.
        """

        if save_to_hdf5 is not None:
            assert self.mode in ['w', 'w-', 'x', 'r+', 'c', 'c-'],\
                "File must be in a write mode"
            assert isinstance(save_to_hdf5, str),\
                "`save_to_hdf5` should be the field name to save the data in the `observables` group in each trajectory"
            field_name=save_to_hdf5

            # DEBUG enforce this until sparse trajectories are implemented
            # assert traj_sel is None, "no selections until sparse trajectory data is implemented"

        results = self.traj_fields_map(func, fields, *args,
                                       map_func=map_func, traj_sel=traj_sel, idxs=True,
                                       debug_prints=debug_prints)

        # if we are saving this to the trajectories observables add it as a dataset
        if save_to_hdf5:

            if debug_prints:
                print("Finished calculations saving them to HDF5")

            for idx_tup, obs_values in results:
                run_idx, traj_idx = idx_tup

                if debug_prints:
                    print("Saving run {} traj {} observables/{}".format(run_idx, traj_idx, field_name))

                # try to get the observables group or make it if it doesn't exist
                try:
                    obs_grp = self.traj(run_idx, traj_idx)['observables']
                except KeyError:

                    if debug_prints:
                        print("Group uninitialized. Initializing.")

                    obs_grp = self.traj(run_idx, traj_idx).create_group('observables')

                # try to create the dataset
                try:
                    obs_grp.create_dataset(field_name, data=obs_values)
                # if it fails we either overwrite or raise an error
                except RuntimeError:
                    # if we are in a permissive write mode we delete the
                    # old dataset and add the new one, overwriting old data
                    if self.mode in ['w', 'w-', 'x', 'r+']:

                        if debug_prints:
                            print("Dataset already present. Overwriting.")

                        del obs_grp[field_name]
                        obs_grp.create_dataset(field_name, data=obs_values)
                    # this will happen in 'c' and 'c-' modes
                    else:
                        raise RuntimeError(
                            "Dataset already exists and file is in concatenate mode ('c' or 'c-')")

        # otherwise just return it
        else:
            if idxs:
                return results
            else:
                return (obs_values for idxs, obs_values in results)

    def join(self, other_h5):
        """Given another WepyHDF5 file object does a left join on this
        file. Renumbering the runs starting from this file.
        """

        with other_h5 as h5:
            for run_idx in h5.run_idxs:
                # the other run group handle
                other_run = h5.run(run_idx)
                # copy this run to this file in the next run_idx group
                self.h5.copy(other_run, 'runs/{}'.format(self._run_idx_counter))
                # increment the run_idx counter
                self._run_idx_counter += 1

    def export_traj(self, run_idx, traj_idx, filepath, mode='x'):
        """Write a single trajectory from the WepyHDF5 container to a TrajHDF5
        file object. Returns the handle to the new file."""

        # get the traj group
        traj_grp = self.h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]

        # get the traj data as a dictionary
        traj_data = {}
        for thing_key in list(traj_grp):
            # if it is a standard traj data field we keep it to put in the traj
            if thing_key in TRAJ_DATA_FIELDS:
                # handle it if it is compound
                if thing_key in COMPOUND_DATA_FIELDS:
                    traj_data[thing_key] = {}
                    for cmp_key in list(traj_grp[thing_key]):
                        traj_data[thing_key][cmp_key] = traj_grp[thing_key][cmp_key]
                # otherwise just copy the dataset
                else:
                    traj_data[thing_key] = traj_grp[thing_key][:]

        # the mapping of common keys to the transfer keys
        unit_key_map = dict(DATA_UNIT_MAP)

        unit_grp = self.h5['units']
        units = {}
        # go through each unit and add them to the dictionary changing
        # the name back to the unit key
        for unit_key in list(unit_grp):
            unit_transfer_key = unit_key_map[unit_key]
            if unit_key in TRAJ_DATA_FIELDS:
                # handle compound units
                if unit_key in COMPOUND_UNIT_FIELDS:
                    units[unit_transfer_key] = {}
                    for cmp_key in list(unit_grp[unit_key]):
                        units[unit_transfer_key][cmp_key] = unit_grp[unit_key][cmp_key]
                # simple units
                else:
                    units[unit_transfer_key] = unit_grp[unit_key][()]

        traj_h5 = TrajHDF5(filepath, mode=mode, topology=self.topology, **traj_data, **units)

        return traj_h5


    def to_mdtraj(self, run_idx, traj_idx, frames=None):

        topology = _json_to_mdtraj_topology(self.topology)

        traj_grp = self.traj(run_idx, traj_idx)

        # get the data for all or for the frames specified
        if frames is None:
            positions = traj_grp['positions'][:]
            time = traj_grp['time'][:, 0]
            box_vectors = traj_grp['box_vectors'][:]
        else:
            positions = traj_grp['positions'][frames]
            time = traj_grp['time'][frames][:, 0]
            box_vectors = traj_grp['box_vectors'][frames]

        unitcell_lengths, unitcell_angles = _box_vectors_to_lengths_angles(box_vectors)

        traj = mdj.Trajectory(positions, topology,
                       time=time,
                       unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)

        return traj


def _extract_dict(keys, **kwargs):
    traj_data = {}
    for field in keys:
        try:
            traj_data[field] = kwargs[field]
        except KeyError:
            pass

    return traj_data

# this is just a prototype, code was copied into where it needs to be
# as it is difficult to figure out which function received the kwargs,
# i.e. passing the function being called to a function it is calling.
def _warn_unknown_kwargs(**kwargs):
    for key in kwargs.keys():
        if not (key in TRAJ_DATA_FIELDS) and not (key in TRAJ_UNIT_FIELDS):
            warn("kwarg {} not recognized and was ignored".format(key), RuntimeWarning)


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

def _make_numpy_warp_dtype(instruct_dtype):

    dtype_map = [('cycle_idx', np.int), ('walker_idx', np.int),
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

def _json_to_mdtraj_topology(json_string):

    topology_dict = json.loads(json_string)

    topology = mdj.Topology()

    for chain_dict in sorted(topology_dict['chains'], key=operator.itemgetter('index')):
        chain = topology.add_chain()
        for residue_dict in sorted(chain_dict['residues'], key=operator.itemgetter('index')):
            try:
                resSeq = residue_dict["resSeq"]
            except KeyError:
                resSeq = None
                warnings.warn(
                    'No resSeq information found in HDF file, defaulting to zero-based indices')
            try:
                segment_id = residue_dict["segmentID"]
            except KeyError:
                segment_id = ""
            residue = topology.add_residue(residue_dict['name'], chain,
                                           resSeq=resSeq, segment_id=segment_id)
            for atom_dict in sorted(residue_dict['atoms'], key=operator.itemgetter('index')):
                try:
                    element = elem.get_by_symbol(atom_dict['element'])
                except KeyError:
                    element = elem.virtual
                topology.add_atom(atom_dict['name'], element, residue)

    atoms = list(topology.atoms)
    for index1, index2 in topology_dict['bonds']:
        topology.add_bond(atoms[index1], atoms[index2])

    return topology

def _box_vectors_to_lengths_angles(box_vectors):

    unitcell_lengths = []
    for basis in box_vectors:
        unitcell_lengths.append(np.array([np.linalg.norm(frame_v) for frame_v in basis]))

    unitcell_lengths = np.array(unitcell_lengths)

    unitcell_angles = []
    for vs in box_vectors:

        angles = np.array([np.degrees(
                            np.arccos(np.dot(vs[i], vs[j])/
                                      (np.linalg.norm(vs[i]) * np.linalg.norm(vs[j]))))
                           for i, j in [(0,1), (1,2), (2,0)]])

        unitcell_angles.append(angles)

    unitcell_angles = np.array(unitcell_angles)

    return unitcell_lengths, unitcell_angles


def _check_data_compliance(traj_data, compliance_requirements=COMPLIANCE_REQUIREMENTS):
    """Given a dictionary of trajectory data it returns the
       COMPLIANCE_TAGS that the data satisfies. """

    # cast the nested tuples to a dictionary if necessary
    compliance_dict = dict(compliance_requirements)

    fields = set()
    for field, value in traj_data.keys():
        # check to make sure the value actually has something in it
        if (value is not None) and len(value > 0):
            fields.update(field)

    compliances = []
    for compliance_tag, compliance_fields in compliance_dict.items():
        compliance_fields = set(compliance_fields)
        # if the fields are a superset of the compliance fields for
        # this compliance type then it satisfies it
        if fields.issuperset(compliance_fields):
            compliances.append(compliance_tag)

    return compliances

# see TODO
def concat(wepy_h5s):
    pass
