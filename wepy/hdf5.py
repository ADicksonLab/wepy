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

# defaults for the rank (length of shape vector) for certain
# unchanging data fields. This is the rank of the feawture not the
# array that will acutally be saved int he hdf5. That will always be
# one more than the rank of the feature.
FIELD_FEATURE_RANKS = (('positions', 2),
                 ('time', 1),
                 ('box_vectors', 2),
                 ('velocities', 2),
                 ('forces', 2),
                 ('box_volume', 1),
                 ('kinetic_energy', 1),
                 ('potential_energy', 1),
                )

# defaults for the shapes for those fields they can be given.
FIELD_FEATURE_SHAPES = (('time', (1,)),
                             ('box_vectors', (3,3)),
                             ('box_volume', (1,)),
                             ('kinetic_energy', (1,)),
                             ('potential_energy', (1,)),
                            )

FIELD_FEATURE_DTYPES = (('positions', np.dtype(np.float)),
                        ('velocities', np.dtype(np.float)),
                        ('forces', np.dtype(np.float)),
                        ('time', np.dtype(np.float)),
                        ('box_vectors', np.dtype(np.float)),
                        ('box_volume', np.dtype(np.float)),
                        ('kinetic_energy', np.dtype(np.float)),
                        ('potential_energy', np.dtype(np.float)),
                        )


# Positions (and thus velocities and forces) are determined by the
# N_DIMS (which can be customized) and more importantly the number of
# particles which is always different. All the others are always wacky
# and different.
POSITIONS_LIKE_FIELDS = ('velocities', 'forces')

WEIGHT_SHAPE = (1,)


# some fields have more than one dataset associated with them
COMPOUND_DATA_FIELDS = ('parameters', 'parameter_derivatives', 'observables')
COMPOUND_UNIT_FIELDS = ('parameters', 'parameter_derivatives', 'observables')

# some fields can have sparse data, non-sparse data must be all the
# same shape and larger than sparse arrays. Sparse arrays have an
# associated index with them aligning them to the non-sparse datasets
SPARSE_DATA_FIELDS = ('velocities', 'forces', 'kinetic_energy', 'potential_energy',
                      'box_volume', 'parameters', 'parameter_derivatives', 'observables')

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

    def __init__(self, filename, topology=None, mode='x',
                 data=None, units=None, sparse_idxs=None,
                 sparse_fields=None,
                 feature_shapes=None, feature_dtypes=None):
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

        # just an alias that makes things more semantic in this code
        traj_data = data

        # set hidden feature shapes and dtype, which are only
        # referenced if needed in the create constructor
        self._field_feature_shapes_kwarg = feature_shapes
        self._field_feature_dtypes_kwarg = feature_dtypes

        # all the keys for the datasets and groups
        self._keys = ['topology', 'positions', 'velocities',
                      'box_vectors',
                      'time', 'box_volume', 'kinetic_energy', 'potential_energy',
                      'forces', 'parameters', 'parameter_derivatives', 'observables']

        # set the flags for sparse data that will be allowed in this
        # object from the sparse field flags or recognize it from the
        # idxs passed in
        if sparse_fields is not None:
            # make a set of the defined sparse fields, if given
            self._sparse_fields = set(sparse_fields)
        else:
            self._sparse_fields = set([])

        # go through the idxs given to the constructor for different
        # fields, and add them to the sparse fields list (if they
        # aren't already there)
        if sparse_idxs is not None:
            for field_path in sparse_idxs.keys():
                self._sparse_fields.add(field_path)

        # append the exist flags
        self._exist_flags = {key : False for key in TRAJ_DATA_FIELDS}
        self._compound_exist_flags = {key : {} for key in COMPOUND_DATA_FIELDS}

        # initialize the append flags
        self._append_flags = {key : True for key in TRAJ_DATA_FIELDS}
        self._compound_append_flags = {key : {} for key in COMPOUND_DATA_FIELDS}

        # initialize the compliance types
        self._compliance_flags = {tag : False for tag, requirements in COMPLIANCE_REQUIREMENTS}

        # open the file in a context and initialize
        self.closed = True
        with h5py.File(filename, mode=self._h5py_mode) as h5:
            self._h5 = h5


            # create file mode: 'w' will create a new file or overwrite,
            # 'w-' and 'x' will not overwrite but will create a new file
            if self._wepy_mode in ['w', 'w-', 'x']:

                self._create_init(topology, traj_data, units, sparse_idxs=sparse_idxs)
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
                self._add_init(topology, traj_data, units, sparse_idxs=sparse_idxs)

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

        # initialize the settings group
        settings_grp = self._h5.create_group('_settings')
        self._h5.create_group('observables')

        # assign the topology
        self.topology = topology

        # make a dataset for the sparse fields allowed.  this requires
        # a 'special' datatype for variable length strings. This is
        # supported by HDF5 but not numpy.
        vlen_str_dt = h5py.special_dtype(vlen=str)

        # create the dataset with empty values for the length of the
        # sparse fields given
        sparse_fields_ds = settings_grp.create_dataset('sparse_fields',
                                                       (len(self._sparse_fields),),
                                                       dtype=vlen_str_dt,
                                                       maxshape=(None,))

        # set the flags
        for i, sparse_field in enumerate(self._sparse_fields):
            sparse_fields_ds[i] = sparse_field

        # TODO check to make sure all the fields of data are the same
        # length as the positions, unless they have sparse idxs
        # if the field has not been marked as sparse then it can't
        # have a different number of frames

        # attributes needed just for construction
        self._n_frames = data['positions'].shape[0]

        ## TODO set in the _set_default_init_field_attributes function for now
        # get the number of coordinates of positions, i.e. n_atoms
        # self._n_coords = data['positions'].shape[1]
        # get the number of dimensions
        # self._n_dims = data['positions'].shape[2]

        # make a list of the field paths from the trajectory data
        field_paths = list(data.keys())

        # check each other field for the correct number of frames
        # unless it is a sparse field
        for field_path in field_paths:
            # get the field data from data
            if not field_path in self.sparse_fields:
                field_data = data[field_path]

                # test the shape of it to make sure it is okay
                assert field_data.shape[0] == self._n_frames, \
                    "field data for {} has different number of frames {} and is not sparse".format(
                        field_name, field_data.shape[0])

        # initialize to the defaults
        self._set_default_init_field_attributes()

        # save the number of dimensions and number of atoms in settings
        settings_grp.create_dataset('n_dims', data=np.array(self._n_dims))
        settings_grp.create_dataset('n_atoms', data=np.array(self._n_coords))

        # if both feature shapes and dtypes were specified overwrite
        # (or initialize if not set by defaults) the defaults
        if (self._field_feature_shapes_kwarg is not None) and\
           (self._field_feature_dtypes_kwarg is not None):

            self._field_feature_shapes.update(self._field_feature_shapes_kwarg)
            self._field_feature_dtypes.update(self._field_feature_dtypes_kwarg)

        # any sparse field with unspecified shape and dtype must be
        # set to None so that it will be set at runtime
        for sparse_field in self.sparse_fields:
            if (not sparse_field in self._field_feature_shapes) or \
               (not sparse_field in self._field_feature_dtypes):
                self._field_feature_shapes[sparse_field] = None
                self._field_feature_dtypes[sparse_field] = None


        # save the field feature shapes and dtypes in the settings group
        shapes_grp = settings_grp.create_group('field_feature_shapes')
        for field_path, field_shape in self._field_feature_shapes.items():
            if field_shape is None:
                # set it as a dimensionless array of NaN
                field_shape = np.array(np.nan)

            shapes_grp.create_dataset(field_path, data=field_shape)

        dtypes_grp = settings_grp.create_group('field_feature_dtypes')
        for field_path, field_dtype in self._field_feature_dtypes.items():
            if field_dtype is None:
                dt_str = 'None'
            else:
                # make a json string of the datatype that can be read in again
                dt_str = json.dumps(field_dtype.descr)

            dtypes_grp.create_dataset(field_path, data=dt_str)


        # create the datasets for the actual data

        # positions
        positions_shape = data['positions'].shape

        # add the rest of the fields of data to the trajectory
        for field_path, field_data in data.items():

            # if there were sparse idxs for this field pass them in
            if field_path in sparse_idxs:
                field_sparse_idxs = sparse_idxs[field_path]
            # if this is a sparse field and no sparse_idxs were given
            # we still need to initialize it as a sparse field so it
            # can be extended properly so we make sparse_idxs to match
            # the full length of this initial trajectory data
            elif field_path in self.sparse_fields:
                field_sparse_idxs = np.arange(positions_shape[0])
            # otherwise it is not a sparse field so we just pass in None
            else:
                field_sparse_idxs = None


            self._add_traj_field_data(field_path, field_data, sparse_idxs=field_sparse_idxs)

        ## initialize empty sparse fields
        # get the sparse field datasets that haven't been initialized
        init_fields = list(sparse_idxs.keys()) + list(traj_data.keys())
        uninit_sparse_fields = set(self.sparse_fields).difference(init_fields)
        # the shapes
        uninit_sparse_shapes = [self.field_feature_shapes[field] for field in uninit_sparse_fields]
        # the dtypes
        uninit_sparse_dtypes = [self.field_feature_dtypes[field] for field in uninit_sparse_fields]
        # initialize the sparse fields in the hdf5
        self._init_fields(uninit_sparse_fields, uninit_sparse_shapes, uninit_sparse_dtypes)

        ## UNITS
        # initialize the units group
        unit_grp = self._h5.create_group('units')

        # set the units
        for field_path, unit_value in units.items():

            # ignore the field if not given
            if unit_value is None:
                continue

            unit_path = '/units/{}'.format(field_path)

            unit_grp.create_dataset(unit_path, data=unit_value)



    def _read_write_init(self):
        """Write over values if given but do not reinitialize any old ones. """

        self._read_init()

    def _add_init(self, topology, data, units, sparse_idxs):
        """Create the dataset if it doesn't exist and put it in r+ mode,
        otherwise, just open in r+ mode."""

        # set the flags for existing data
        self._update_exist_flags()

        if not any(self._exist_flags):
            self._create_init(topology, data, units)
        else:
            self._read_write_init()

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

    def _get_field_path_grp(self, field_path):
        """Given a field path for the trajectory returns the group the field's
        dataset goes in and the key for the field name in that group.

        The field path for a simple field is just the name of the
        field and for a compound field it is the compound field group
        name with the subfield separated by a '/' like
        'observables/observable1' where 'observables' is the compound
        field group and 'observable1' is the subfield name.

        """

        # check if it is compound
        if '/' in field_path:
            # split it
            grp_name, field_name = field_path.split('/')
            # get the hdf5 group
            grp = self.h5[grp_name]
        # its simple so just return the root group and the original path
        else:
            grp = self.h5
            field_name = field_path

        return grp, field_name

    def _init_field(self, field_path, feature_shape, dtype):
        """Initialize a data field in the trajectory to be empty but
        resizeable."""

        # check whether this is a sparse field and create it
        # appropriately
        if field_path in self.sparse_fields:
            # it is a sparse field
            self._init_sparse_field(field_path, feature_shape, dtype)
        else:
            # it is not a sparse field (AKA simple)
            self._init_contiguous_field(field_path, feature_shape, dtype)

    def _init_contiguous_field(self, field_path, feature_shape, dtype):

        # create the empty dataset in the correct group, setting
        # maxshape so it can be resized for new feature vectors to be added
        self._h5.create_dataset(field_path, (0, *[0 for i in feature_shape]), dtype=dtype,
                           maxshape=(None, *feature_shape))


    def _init_sparse_field(self, field_path, feature_shape, dtype):

        sparse_grp = self.h5.create_group(field_path)

        # check to see that neither the feature_shape and dtype are
        # None which indicates it is a runtime defined value and
        # should be ignored here
        if (feature_shape is None) or (dtype is None):
            # do nothing
            pass
        else:
            # create the dataset for the feature data
            sparse_grp.create_dataset('data', (0, *[0 for i in feature_shape]), dtype=dtype,
                               maxshape=(None, *feature_shape))

            # create the dataset for the sparse indices
            sparse_grp.create_dataset('_sparse_idxs', (0,), dtype=np.int, maxshape=(None,))


    def _init_fields(self, field_paths, field_feature_shapes, field_feature_dtypes):
        for i, field_path in enumerate(field_paths):
            self._init_field(field_path, field_feature_shapes[i], field_feature_dtypes[i])


    def _set_default_init_field_attributes(self):
        """Sets the feature_shapes and feature_dtypes to be the default for
        this module. These will be used to initialize field datasets when no
        given during construction (i.e. for sparse values)"""

        # we use the module defaults for the datasets to initialize them
        field_feature_shapes = dict(FIELD_FEATURE_SHAPES)
        field_feature_dtypes = dict(FIELD_FEATURE_DTYPES)

        # get the number of coordinates of positions, i.e. n_atoms
        # from the topology
        self._n_coords = _json_top_atom_count(self.topology)
        # get the number of dimensions as a default
        self._n_dims = N_DIMS

        # feature shapes for positions and positions-like fields are
        # not known at the module level due to different number of
        # coordinates (number of atoms) and number of dimensions
        # (default 3 spatial). We set them now that we know this
        # information.
        # add the postitions shape
        field_feature_shapes['positions'] = (self._n_coords, self._n_dims)
        # add the positions-like field shapes (velocities and forces) as the same
        for poslike_field in POSITIONS_LIKE_FIELDS:
            field_feature_shapes[poslike_field] = (self._n_coords, self._n_dims)

        # set the attributes
        self._field_feature_shapes = field_feature_shapes
        self._field_feature_dtypes = field_feature_dtypes

    def _add_traj_field_data(self, field_path, field_data, sparse_idxs=None):

        # if it is a sparse dataset we need to add the data and add
        # the idxs in a group
        if sparse_idxs is None:
            # create the dataset
            self.h5.create_dataset(field_path, data=field_data, maxshape=(None, *field_data.shape[1:]))
        else:
            sparse_grp = self.h5.create_group(field_path)
            # add the data to this group
            sparse_grp.create_dataset('data', data=field_data, maxshape=(None, *field_data.shape[1:]))
            # add the sparse idxs
            sparse_grp.create_dataset('_sparse_idxs', data=sparse_idxs, maxshape=(None,))



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
        else:
            warn("File already closed")

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
    def sparse_fields(self):
        return self.h5['_settings/sparse_fields'][:]

    @property
    def field_feature_shapes(self):
        shapes_grp = self.h5['_settings/field_feature_shapes']
        return {field_path : shape_ds[()] for field_path, shape_ds in shapes_grp.items()}

    @property
    def field_feature_dtypes(self):
        dtypes_grp = self.h5['_settings/field_feature_dtypes']
        return {field_path : np.dtype(json.loads(dtype_ds[()])) for
                field_path, dtype_ds in dtypes_grp.items()}

    @property
    def n_frames(self):
        return self.positions.shape[0]

    @property
    def n_atoms(self):
        return self.h5['_settings/n_atoms'][()]

    @property
    def n_dims(self):
        return self.h5['_settings/n_dims'][()]

    @property
    def fields(self):
        field_names = []
        for field in self.h5:
            if field in COMPOUND_DATA_FIELDS:
                for subfield in self.h5[field]:
                    field_path = field + '/' + subfield
                    field_names.append(field_path)
            else:
                field_names.append(field)

        return fields

    def _extend_contiguous_field(self, field_path, values):

        field = self.h5[field_path]

        # make sure this is a feature vector
        assert len(values.shape) > 1, \
            "values must be a feature vector with the same number of dimensions as the number"

        # of datase new frames
        n_new_frames = values.shape[0]

        # check the field to make sure it is not empty
        if all([i == 0 for i in field.shape]):

            # check the feature shape against the maxshape which gives
            # the feature dimensions for an empty dataset
            assert values.shape[1:] == field.maxshape[1:], \
                "field feature dimensions must be the same, i.e. all but the first dimension"

            # if it is empty resize it to make an array the size of
            # the new values with the maxshape for the feature
            # dimensions
            feature_dims = field.maxshape[1:]
            field.resize( (n_new_frames, *feature_dims) )

            # set the new data to this
            field[0:, ...] = values

        else:
            # make sure the new data has the right dimensions against
            # the shape it already has
            assert values.shape[1:] == field.shape[1:], \
                "field feature dimensions must be the same, i.e. all but the first dimension"


            # append to the dataset on the first dimension, keeping the
            # others the same, these must be feature vectors and therefore
            # must exist
            field.resize( (field.shape[0] + n_new_frames, *field.shape[1:]) )
            # add the new data
            field[-n_new_frames:, ...] = values

    def _extend_sparse_field(self, field_path, values, sparse_idxs):

        field = self.h5[field_path]

        field_data = field['data']
        field_sparse_idxs = field['_sparse_idxs']

        # make sure this is a feature vector
        assert len(values.shape) > 1, \
            "values must be a feature vector with the same number of dimensions as the dataset"

        # number of new frames
        n_new_frames = values.shape[0]

        if all([i == 0 for i in field_data.shape]):

            # check the feature shape against the maxshape which gives
            # the feature dimensions for an empty dataset
            assert values.shape[1:] == field_data.maxshape[1:], \
                "field feature dimensions must be the same, i.e. all but the first dimension"

            # if it is empty resize it to make an array the size of
            # the new values with the maxshape for the feature
            # dimensions
            feature_dims = field_data.maxshape[1:]
            field_data.resize( (n_new_frames, *feature_dims) )

            # set the new data to this
            field_data[0:, ...] = values

        else:

            # make sure the new data has the right dimensions
            assert values.shape[1:] == field_data.shape[1:], \
                "field feature dimensions must be the same, i.e. all but the first dimension"

            # append to the dataset on the first dimension, keeping the
            # others the same, these must be feature vectors and therefore
            # must exist
            field_data.resize( (field_data.shape[0] + n_new_frames, *field_data.shape[1:]) )
            # add the new data
            field_data[-n_new_frames:, ...] = values

        # add the sparse idxs in the same way
        field_sparse_idxs.resize( (field_sparse_idxs.shape[0] + n_new_frames,
                                   *field_sparse_idxs.shape[1:]) )
        # add the new data
        field_sparse_idxs[-n_new_frames:, ...] = sparse_idxs


    def extend(self, data):
        """ append to the trajectory as a whole """

        # nicer alias
        traj_data = data

        if self._wepy_mode == 'c-':
            assert self._append_flags[dataset_key], "dataset is not available for appending to"

        # assess compliance types
        compliance_tags = _check_data_compliance(traj_data)
        # assert we meet minimum compliance, unless extra fields are
        # sparse
        assert 'COORDS' in compliance_tags, \
            "Appended data must minimally be COORDS compliant"

        # TODO check other compliances for this dataset

        # number of frames to add
        n_new_frames = traj_data['positions'].shape[0]

        # calculate the new sparse idxs for sparse fields that may be
        # being added
        sparse_idxs = np.array(range(self.n_frames, self.n_frames + n_new_frames))

        # add trajectory data for each field
        for field_path, field_data in traj_data.items():

            # if the field hasn't been initialized yet initialize it
            if not field_path in self._h5:
                feature_shape = field_data.shape[1:]
                feature_dtype = field_data.dtype

                # not specified as sparse_field, no settings
                if (not field_path in self.field_feature_shapes) and \
                     (not field_path in self.field_feature_dtypes) and \
                     not field_path in self.sparse_fields:
                    ## only save if it is an observable
                    is_observable = False
                    if '/' in field_path:
                        group_name = field_path.split('/')[0]
                        if group_name == 'observables':
                            is_observable = True
                    if is_observable:
                          warn("the field '{}' was received but not previously specified"
                               " but is being added because it is in observables.".format(field_path))
                          ## save sparse_field flag, shape, and dtype
                          self._add_sparse_field_flag(field_path)
                          self._add_field_feature_shape(field_path, feature_shape)
                          feature_dtype_str = json.dumps(feature_dtype.descr)
                          self._add_field_feature_dtype(field_path, feature_dtype_str)
                    else:
                        raise ValueError("the field '{}' was received but not previously specified"
                            "it is being ignored because it is not an observable.".format(field_path))
                # specified as sparse_field but no settings given
                elif (self.field_feature_shapes[field_path] is None and
                   self.field_feature_dtypes[field_path] is None) and \
                   field_path in self.sparse_fields:
                    ## save shape and dtype
                    # add the shape and dtype
                    self._add_field_feature_shape(field_path, feature_shape)
                    feature_dtype_str = json.dumps(feature_dtype.descr)
                    self._add_field_feature_dtype(field_path, feature_dtype_str)


                # initialize
                self._init_field(field_path, feature_shape, feature_dtype)

            # extend it either as a sparse field or a contiguous field
            if field_path in self.sparse_fields:
                self._extend_sparse_field(field_path, field_data, sparse_idxs)
            else:
                self._extend_contiguous_field(field_path, field_data)


    def _add_sparse_field_flag(self, field_path):

        sparse_fields_ds = self._h5['_settings/sparse_fields']

        # make sure it isn't already in the sparse_fields
        if field_path in sparse_fields_ds[:]:
            warn("sparse field {} already a sparse field, ignoring".format(field_path))

        sparse_fields_ds.resize( (sparse_fields_ds.shape[0] + 1,) )
        sparse_fields_ds[sparse_fields_ds.shape[0] - 1] = field_path

    def _add_field_feature_shape(self, field_path, field_feature_shape):
        shapes_grp = self._h5['_settings/field_feature_shapes']
        shapes_grp.create_dataset(field_path, data=np.array(field_feature_shape))

    def _add_field_feature_dtype(self, field_path, field_feature_dtype):
        dtypes_grp = self._h5['_settings/field_feature_dtypes']
        dtypes_grp.create_dataset(field_path, data=field_feature_dtype)

    def _get_contiguous_field(self, field_path):
        return self._h5[field_path]

    def _get_sparse_field(self, field_path):

        field = self._h5[field_path]
        data = field['data'][:]
        sparse_idxs = field['_sparse_idxs'][:]

        filled_data = np.full( (self.n_frames, *data.shape[1:]), np.nan)
        filled_data[sparse_idxs] = data

        mask = np.full( (self.n_frames, *data.shape[1:]), True)
        mask[sparse_idxs] = False

        masked_array = np.ma.masked_array(filled_data, mask=mask)

        return masked_array

    def get_field(self, field_path):
        """Returns a numpy array for the given field."""

        assert isinstance(field_path, str), "field_path must be a string"

        # if the field doesn't exist return None
        if not field_path in self._h5:
            return None

        # get the field depending on whether it is sparse or not
        if field_path in self.sparse_fields:
            return self._get_sparse_field(field_path)
        else:
            return self._get_contiguous_field(field_path)


    @property
    def positions(self):
        return self.get_field('positions')
    @property
    def time(self):
        return self.get_field('time')
    @property
    def box_volume(self):
        return self.get_field('box_volume')
    @property
    def kinetic_energy(self):
        return self.get_field('kinetic_energy')
    @property
    def potential_energy(self):
        return self.get_field('potential_energy')
    @property
    def box_vectors(self):
        return self.get_field('box_vectors')
    @property
    def velocities(self):
        return self.get_field('velocities')
    @property
    def forces(self):
        return self.get_field('forces')
    @property
    def parameters(self):
        return self.get_field('parameters')
    @property
    def parameter_derivatives(self):
        return self.get_field('parameter_derivatives')
    @property
    def observables(self):
        return self.get_field('observables')

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

    def __init__(self, filename, topology=None, mode='x',
                 units=None,
                 sparse_fields=None,
                 feature_shapes=None, feature_dtypes=None,
                 n_dims=None):
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

        self._n_dims = n_dims


        # set hidden feature shapes and dtype, which are only
        # referenced if needed when trajectories are created. These
        # will be saved in the settings section in the actual HDF5
        # file
        self._field_feature_shapes_kwarg = feature_shapes
        self._field_feature_dtypes_kwarg = feature_dtypes

        # save the sparse fields as a private variable for use in the
        # create constructor
        if sparse_fields is None:
            self._sparse_fields = []
        else:
            self._sparse_fields = sparse_fields

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

    def _create_init(self, topology, units):
        """Completely overwrite the data in the file. Reinitialize the values
        and set with the new ones if given."""

        assert topology is not None, "Topology must be given"

        # assign the topology
        self.topology = topology

        # attributes needed just for construction

        # initialize the settings group
        settings_grp = self._h5.create_group('_settings')

        # sparse fields
        if self._sparse_fields is not None:

            # make a dataset for the sparse fields allowed.  this requires
            # a 'special' datatype for variable length strings. This is
            # supported by HDF5 but not numpy.
            vlen_str_dt = h5py.special_dtype(vlen=str)

            # create the dataset with empty values for the length of the
            # sparse fields given
            sparse_fields_ds = settings_grp.create_dataset('sparse_fields',
                                                           (len(self._sparse_fields),),
                                                           dtype=vlen_str_dt,
                                                           maxshape=(None,))

            # set the flags
            for i, sparse_field in enumerate(self._sparse_fields):
                sparse_fields_ds[i] = sparse_field


        # field feature shapes and dtypes

        # initialize to the defaults
        if self._n_dims is None:
            self._set_default_init_field_attributes(n_dims=self._n_dims)


        # save the number of dimensions and number of atoms in settings
        settings_grp.create_dataset('n_dims', data=np.array(self._n_dims))
        settings_grp.create_dataset('n_atoms', data=np.array(self._n_coords))

        # if both feature shapes and dtypes were specified overwrite
        # (or initialize if not set by defaults) the defaults
        if (self._field_feature_shapes_kwarg is not None) and\
           (self._field_feature_dtypes_kwarg is not None):

            self._field_feature_shapes.update(self._field_feature_shapes_kwarg)
            self._field_feature_dtypes.update(self._field_feature_dtypes_kwarg)

        # any sparse field with unspecified shape and dtype must be
        # set to None so that it will be set at runtime
        for sparse_field in self.sparse_fields:
            if (not sparse_field in self._field_feature_shapes) or \
               (not sparse_field in self._field_feature_dtypes):
                self._field_feature_shapes[sparse_field] = None
                self._field_feature_dtypes[sparse_field] = None


        # save the field feature shapes and dtypes in the settings group
        shapes_grp = settings_grp.create_group('field_feature_shapes')
        for field_path, field_shape in self._field_feature_shapes.items():
            if field_shape is None:
                # set it as a dimensionless array of NaN
                field_shape = np.array(np.nan)

            shapes_grp.create_dataset(field_path, data=field_shape)

        dtypes_grp = settings_grp.create_group('field_feature_dtypes')
        for field_path, field_dtype in self._field_feature_dtypes.items():
            if field_dtype is None:
                dt_str = 'None'
            else:
                # make a json string of the datatype that can be read in again
                dt_str = json.dumps(field_dtype.descr)

            dtypes_grp.create_dataset(field_path, data=dt_str)

        # initialize the units group
        unit_grp = self._h5.create_group('units')

        # set the units
        for field_path, unit_value in units.items():

            # ignore the field if not given
            if unit_value is None:
                continue

            unit_path = '/units/{}'.format(field_path)

            unit_grp.create_dataset(unit_path, data=unit_value)


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

    def _get_field_path_grp(self, run_idx, traj_idx, field_path):
        """Given a field path for the trajectory returns the group the field's
        dataset goes in and the key for the field name in that group.

        The field path for a simple field is just the name of the
        field and for a compound field it is the compound field group
        name with the subfield separated by a '/' like
        'observables/observable1' where 'observables' is the compound
        field group and 'observable1' is the subfield name.

        """

        # check if it is compound
        if '/' in field_path:
            # split it
            grp_name, field_name = field_path.split('/')
            # get the hdf5 group
            grp = self.h5['runs/{}/trajectories/{}/{}'.format(run_idx, traj_idx, grp_name)]
        # its simple so just return the root group and the original path
        else:
            grp = self.h5
            field_name = field_path

        return grp, field_name

    def _set_default_init_field_attributes(self, n_dims=None):
        """Sets the feature_shapes and feature_dtypes to be the default for
        this module. These will be used to initialize field datasets when no
        given during construction (i.e. for sparse values)"""

        # we use the module defaults for the datasets to initialize them
        field_feature_shapes = dict(FIELD_FEATURE_SHAPES)
        field_feature_dtypes = dict(FIELD_FEATURE_DTYPES)


        # get the number of coordinates of positions, i.e. n_atoms
        # from the topology
        self._n_coords = _json_top_atom_count(self.topology)
        # get the number of dimensions as a default
        if n_dims is None:
            self._n_dims = N_DIMS

        # feature shapes for positions and positions-like fields are
        # not known at the module level due to different number of
        # coordinates (number of atoms) and number of dimensions
        # (default 3 spatial). We set them now that we know this
        # information.
        # add the postitions shape
        field_feature_shapes['positions'] = (self._n_coords, self._n_dims)
        # add the positions-like field shapes (velocities and forces) as the same
        for poslike_field in POSITIONS_LIKE_FIELDS:
            field_feature_shapes[poslike_field] = (self._n_coords, self._n_dims)

        # set the attributes
        self._field_feature_shapes = field_feature_shapes
        self._field_feature_dtypes = field_feature_dtypes

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
    def n_trajs(self):
        return len(list(self.run_traj_idx_tuples()))

    @property
    def settings(self):
        return NotImplementedError

    @property
    def n_atoms(self):
        return self.h5['_settings/n_atoms'][()]

    @property
    def n_dims(self):
        return self.h5['_settings/n_dims'][()]

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

    @property
    def n_atoms(self):
        return self.h5['_settings/n_atoms'][()]

    @property
    def n_dims(self):
        return self.h5['_settings/n_dims'][()]

    @property
    def sparse_fields(self):
        return self.h5['_settings/sparse_fields'][:]

    @property
    def field_feature_shapes(self):
        shapes_grp = self.h5['_settings/field_feature_shapes']

        field_paths = iter_field_paths(shapes_grp)

        shapes = {}
        for field_path in field_paths:
            shape = shapes_grp[field_path][()]
            if np.isnan(shape).all():
                shapes[field_path] = None
            else:
                shapes[field_path] = shape

        return shapes

    @property
    def field_feature_dtypes(self):

        dtypes_grp = self.h5['_settings/field_feature_dtypes']

        field_paths = iter_field_paths(dtypes_grp)

        dtypes = {}
        for field_path in field_paths:
            dtype_str = dtypes_grp[field_path][()]
            # if there is 'None' flag for the dtype then return None
            if dtype_str == 'None':
                dtypes[field_path] = None
            else:
                dtype_obj = json.loads(dtype_str)
                dtype_obj = [tuple(d) for d in dtype_obj]
                dtype = np.dtype(dtype_obj)
                dtypes[field_path] = dtype

        return dtypes

    @property
    def metadata(self):
        return dict(self._h5.attrs)

    def add_metadata(self, key, value):
        self._h5.attrs[key] = value

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

    def _init_traj_field(self, run_idx, traj_idx, field_path, feature_shape, dtype):
        """Initialize a data field in the trajectory to be empty but
        resizeable."""

        # check whether this is a sparse field and create it
        # appropriately
        if field_path in self.sparse_fields:
            # it is a sparse field
            self._init_sparse_traj_field(run_idx, traj_idx, field_path, feature_shape, dtype)
        else:
            # it is not a sparse field (AKA simple)
            self._init_contiguous_traj_field(run_idx, traj_idx, field_path, feature_shape, dtype)

    def _init_contiguous_traj_field(self, run_idx, traj_idx, field_path, shape, dtype):

        traj_grp = self._h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]

        # create the empty dataset in the correct group, setting
        # maxshape so it can be resized for new feature vectors to be added
        traj_grp.create_dataset(field_path, (0, *[0 for i in shape]), dtype=dtype,
                           maxshape=(None, *shape))


    def _init_sparse_traj_field(self, run_idx, traj_idx, field_path, shape, dtype):

        traj_grp = self._h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]

        # check to see that neither the shape and dtype are
        # None which indicates it is a runtime defined value and
        # should be ignored here
        if (shape is None) or (dtype is None):
            # do nothing
            pass
        else:

            # only create the group if you are going to add the
            # datasets so the extend function can know if it has been
            # properly initialized easier
            sparse_grp = traj_grp.create_group(field_path)

            # create the dataset for the feature data
            sparse_grp.create_dataset('data', (0, *[0 for i in shape]), dtype=dtype,
                               maxshape=(None, *shape))

            # create the dataset for the sparse indices
            sparse_grp.create_dataset('_sparse_idxs', (0,), dtype=np.int, maxshape=(None,))


    def _init_traj_fields(self, run_idx, traj_idx,
                          field_paths, field_feature_shapes, field_feature_dtypes):
        for i, field_path in enumerate(field_paths):
            self._init_traj_field(run_idx, traj_idx,
                                  field_path, field_feature_shapes[i], field_feature_dtypes[i])


    def add_traj(self, run_idx, data, weights=None, sparse_idxs=None, metadata=None):

        # convenient alias
        traj_data = data

        # initialize None kwargs
        if sparse_idxs is None:
            sparse_idxs = {}
        if metadata is None:
            metadata = {}

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

        # check to make sure the positions are the right shape
        assert traj_data['positions'].shape[1] == self.n_atoms, \
            "positions given have different number of atoms: {}, should be {}".format(
                pos_n_atoms, self.n_atoms)
        assert traj_data['positions'].shape[2] == self.n_dims, \
            "positions given have different number of dims: {}, should be {}".format(
                pos_n_dims, self.n_dims)

        # add datasets to the traj group

        # weights
        traj_grp.create_dataset('weights', data=weights, maxshape=(None, *WEIGHT_SHAPE))
        # positions

        positions_shape = traj_data['positions'].shape

        # add the rest of the traj_data
        for field_path, field_data in traj_data.items():

            # if there were sparse idxs for this field pass them in
            if field_path in sparse_idxs:
                field_sparse_idxs = sparse_idxs[field_path]
            # if this is a sparse field and no sparse_idxs were given
            # we still need to initialize it as a sparse field so it
            # can be extended properly so we make sparse_idxs to match
            # the full length of this initial trajectory data
            elif field_path in self.sparse_fields:
                field_sparse_idxs = np.arange(positions_shape[0])
            # otherwise it is not a sparse field so we just pass in None
            else:
                field_sparse_idxs = None


            self._add_traj_field_data(run_idx, traj_idx, field_path, field_data,
                                      sparse_idxs=field_sparse_idxs)

        ## initialize empty sparse fields
        # get the sparse field datasets that haven't been initialized
        traj_init_fields = list(sparse_idxs.keys()) + list(traj_data.keys())
        uninit_sparse_fields = set(self.sparse_fields).difference(traj_init_fields)
        # the shapes
        uninit_sparse_shapes = [self.field_feature_shapes[field] for field in uninit_sparse_fields]
        # the dtypes
        uninit_sparse_dtypes = [self.field_feature_dtypes[field] for field in uninit_sparse_fields]
        # initialize the sparse fields in the hdf5
        self._init_traj_fields(run_idx, traj_idx,
                               uninit_sparse_fields, uninit_sparse_shapes, uninit_sparse_dtypes)

        # # if a sparse_field has been specified but has not been given
        # # and shapes and dtypes were provided it must be initialized
        # # so this trajectory can be extended
        # for sparse_field in self.sparse_fields:
        #     if (not sparse_field in traj_data) and (not sparse_field in traj_grp):
        #         # get the shape and dtype for this field
        #         shape = self.field_feature_shapes[sparse_field]
        #         dtype = self.field_feature_dtypes[sparse_field]
        #         # initialize it
        #         self._init_sparse_traj_field(run_idx, traj_idx, sparse_field,
        #                                      shape, dtype)

        return traj_grp

    def _add_traj_field_data(self, run_idx, traj_idx, field_path, field_data, sparse_idxs=None):

        # get the traj group
        traj_grp = self._h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]

        # if it is a sparse dataset we need to add the data and add
        # the idxs in a group
        if sparse_idxs is None:
            traj_grp.create_dataset(field_path, data=field_data,
                                    maxshape=(None, *field_data.shape[1:]))
        else:
            sparse_grp = traj_grp.create_group(field_path)
            # add the data to this group
            sparse_grp.create_dataset('data', data=field_data,
                                      maxshape=(None, *field_data.shape[1:]))
            # add the sparse idxs
            sparse_grp.create_dataset('_sparse_idxs', data=sparse_idxs,
                                      maxshape=(None,))

    def _extend_contiguous_traj_field(self, run_idx, traj_idx, field_path, field_data):

        traj_grp = self.h5['/runs/{}/trajectories/{}'.format(run_idx, traj_idx)]
        field = traj_grp[field_path]

        # make sure this is a feature vector
        assert len(field_data.shape) > 1, \
            "field_data must be a feature vector with the same number of dimensions as the number"

        # of datase new frames
        n_new_frames = field_data.shape[0]

        # check the field to make sure it is not empty
        if all([i == 0 for i in field.shape]):

            # check the feature shape against the maxshape which gives
            # the feature dimensions for an empty dataset
            assert field_data.shape[1:] == field.maxshape[1:], \
                "field feature dimensions must be the same, i.e. all but the first dimension"

            # if it is empty resize it to make an array the size of
            # the new field_data with the maxshape for the feature
            # dimensions
            feature_dims = field.maxshape[1:]
            field.resize( (n_new_frames, *feature_dims) )

            # set the new data to this
            field[0:, ...] = field_data

        else:
            # make sure the new data has the right dimensions against
            # the shape it already has
            assert field_data.shape[1:] == field.shape[1:], \
                "field feature dimensions must be the same, i.e. all but the first dimension"


            # append to the dataset on the first dimension, keeping the
            # others the same, these must be feature vectors and therefore
            # must exist
            field.resize( (field.shape[0] + n_new_frames, *field.shape[1:]) )
            # add the new data
            field[-n_new_frames:, ...] = field_data

    def _extend_sparse_traj_field(self, run_idx, traj_idx, field_path, values, sparse_idxs):

        field = self.h5['/runs/{}/trajectories/{}/{}'.format(run_idx, traj_idx, field_path)]

        field_data = field['data']
        field_sparse_idxs = field['_sparse_idxs']

        # number of new frames
        n_new_frames = values.shape[0]

        # if this sparse_field has been initialized empty we need to resize
        if all([i == 0 for i in field_data.shape]):


            # check the feature shape against the maxshape which gives
            # the feature dimensions for an empty dataset
            assert values.shape[1:] == field_data.maxshape[1:], \
                "field feature dimensions must be the same, i.e. all but the first dimension"

            # if it is empty resize it to make an array the size of
            # the new values with the maxshape for the feature
            # dimensions
            feature_dims = field_data.maxshape[1:]
            field_data.resize( (n_new_frames, *feature_dims) )

            # set the new data to this
            field_data[0:, ...] = values

        else:

            # make sure the new data has the right dimensions
            assert values.shape[1:] == field_data.shape[1:], \
                "field feature dimensions must be the same, i.e. all but the first dimension"

            # append to the dataset on the first dimension, keeping the
            # others the same, these must be feature vectors and therefore
            # must exist
            field_data.resize( (field_data.shape[0] + n_new_frames, *field_data.shape[1:]) )
            # add the new data
            field_data[-n_new_frames:, ...] = values

        # add the sparse idxs in the same way
        field_sparse_idxs.resize( (field_sparse_idxs.shape[0] + n_new_frames,
                                   *field_sparse_idxs.shape[1:]) )
        # add the new data
        field_sparse_idxs[-n_new_frames:, ...] = sparse_idxs


    def extend_traj(self, run_idx, traj_idx, data, weights=None):

        if self._wepy_mode == 'c-':
            assert self._append_flags[dataset_key], "dataset is not available for appending to"

        # convenient alias
        traj_data = data

        # number of frames to add
        n_new_frames = traj_data['positions'].shape[0]

        n_frames = self.traj_n_frames(run_idx, traj_idx)

        # calculate the new sparse idxs for sparse fields that may be
        # being added
        sparse_idxs = np.array(range(n_frames, n_frames + n_new_frames))

        # get the trajectory group
        traj_grp = self._h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]

        ## weights

        # if weights are None then we assume they are 1.0
        if weights is None:
            weights = np.ones(n_new_frames, dtype=float)
        else:
            assert isinstance(weights, np.ndarray), "weights must be a numpy.ndarray"
            assert weights.shape[0] == n_new_frames,\
                "weights and the number of frames must be the same length"

        # add the weights
        weights_ds = traj_grp['weights']

        # append to the dataset on the first dimension, keeping the
        # others the same, if they exist
        if len(weights_ds.shape) > 1:
            weights_ds.resize( (weights_ds.shape[0] + n_new_frames, *weights_ds.shape[1:]) )
        else:
            weights_ds.resize( (weights_ds.shape[0] + n_new_frames, ) )

        # add the new data
        weights_ds[-n_new_frames:, ...] = weights


        # add the other fields
        for field_path, field_data in traj_data.items():

            # if the field hasn't been initialized yet initialize it
            if not field_path in traj_grp:
                feature_shape = field_data.shape[1:]
                feature_dtype = field_data.dtype

                # not specified as sparse_field, no settings
                if (not field_path in self.field_feature_shapes) and \
                     (not field_path in self.field_feature_dtypes) and \
                     not field_path in self.sparse_fields:
                    # only save if it is an observable
                    is_observable = False
                    if '/' in field_path:
                        group_name = field_path.split('/')[0]
                        if group_name == 'observables':
                            is_observable = True
                    if is_observable:
                          warn("the field '{}' was received but not previously specified"
                               " but is being added because it is in observables.".format(field_path))
                          # save sparse_field flag, shape, and dtype
                          self._add_sparse_field_flag(field_path)
                          self._set_field_feature_shape(field_path, feature_shape)
                          self._set_field_feature_dtype(field_path, feature_dtype)
                    else:
                        raise ValueError("the field '{}' was received but not previously specified"
                            "it is being ignored because it is not an observable.".format(field_path))

                # specified as sparse_field but no settings given
                elif (self.field_feature_shapes[field_path] is None and
                   self.field_feature_dtypes[field_path] is None) and \
                   field_path in self.sparse_fields:
                    # set the feature shape and dtype since these
                    # should be 0 in the settings
                    self._set_field_feature_shape(field_path, feature_shape)

                    self._set_field_feature_dtype(field_path, feature_dtype)

                # initialize
                self._init_traj_field(run_idx, traj_idx, field_path, feature_shape, feature_dtype)

            # extend it either as a sparse field or a contiguous field
            if field_path in self.sparse_fields:
                self._extend_sparse_traj_field(run_idx, traj_idx, field_path, field_data, sparse_idxs)
            else:
                self._extend_contiguous_traj_field(run_idx, traj_idx, field_path, field_data)

    def traj_n_frames(self, run_idx, traj_idx):
        return self.traj(run_idx, traj_idx)['positions'].shape[0]


    def _add_sparse_field_flag(self, field_path):

        sparse_fields_ds = self._h5['_settings/sparse_fields']

        # make sure it isn't already in the sparse_fields
        if field_path in sparse_fields_ds[:]:
            warn("sparse field {} already a sparse field, ignoring".format(field_path))

        sparse_fields_ds.resize( (sparse_fields_ds.shape[0] + 1,) )
        sparse_fields_ds[sparse_fields_ds.shape[0] - 1] = field_path

    def _add_field_feature_shape(self, field_path, field_feature_shape):
        shapes_grp = self._h5['_settings/field_feature_shapes']
        shapes_grp.create_dataset(field_path, data=np.array(field_feature_shape))

    def _add_field_feature_dtype(self, field_path, field_feature_dtype):
        feature_dtype_str = json.dumps(field_feature_dtype.descr)
        dtypes_grp = self._h5['_settings/field_feature_dtypes']
        dtypes_grp.create_dataset(field_path, data=feature_dtype_str)

    def _set_field_feature_shape(self, field_path, field_feature_shape):
        # check if the field_feature_shape is already set
        if field_path in self.field_feature_shapes:
            # check that the shape was previously saved as "None" as we
            # won't overwrite anything else
            if self.field_feature_shapes[field_path] is None:
                full_path = '_settings/field_feature_shapes/{}'.format(field_path)
                # we have to delete the old data and set new data
                del self.h5[full_path]
                self.h5.create_dataset(full_path, data=field_feature_shape)
            else:
                raise AttributeError(
                    "Cannot overwrite feature shape for {} with {} because it is {} not 'None'".format(
                        field_path, field_feature_shape, self.field_feature_shapes[field_path]))
        # it was not previously set so we must create then save it
        else:
            self._add_field_feature_shape(field_path, field_feature_shape)

    def _set_field_feature_dtype(self, field_path, field_feature_dtype):
        feature_dtype_str = json.dumps(field_feature_dtype.descr)
        # check if the field_feature_dtype is already set
        if field_path in self.field_feature_dtypes:
            # check that the dtype was previously saved as "None" as we
            # won't overwrite anything else
            if self.field_feature_dtypes[field_path] is None:
                full_path = '_settings/field_feature_dtypes/{}'.format(field_path)
                # we have to delete the old data and set new data
                del self.h5[full_path]
                self.h5.create_dataset(full_path, data=feature_dtype_str)
            else:
                raise AttributeError(
                    "Cannot overwrite feature dtype for {} with {} because it is {} not 'None'".format(
                        field_path, field_feature_dtype, self.field_feature_dtypes[field_path]))
        # it was not previously set so we must create then save it
        else:
            self._add_field_feature_dtype(field_path, field_feature_dtype)



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
            enum[decision_name] = dset[()]

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

    def _get_contiguous_traj_field(self, run_idx, traj_idx, field_path):

        full_path = "/runs/{}/trajectories/{}/{}".format(run_idx, traj_idx, field_path)
        return self._h5[full_path][:]

    def _get_sparse_traj_field(self, run_idx, traj_idx, field_path):

        traj_path = "/runs/{}/trajectories/{}".format(run_idx, traj_idx)
        traj_grp = self.h5[traj_path]
        field = traj_grp[field_path]
        data = field['data'][:]
        sparse_idxs = field['_sparse_idxs'][:]

        n_frames = traj_grp['positions'].shape[0]

        filled_data = np.full( (n_frames, *data.shape[1:]), np.nan)
        filled_data[sparse_idxs] = data

        mask = np.full( (n_frames, *data.shape[1:]), True)
        mask[sparse_idxs] = False

        masked_array = np.ma.masked_array(filled_data, mask=mask)

        return masked_array

    def get_traj_field(self, run_idx, traj_idx, field_path):
        """Returns a numpy array for the given field."""

        traj_path = "/runs/{}/trajectories/{}".format(run_idx, traj_idx)

        # if the field doesn't exist return None
        if not field_path in self._h5[traj_path]:
            return None

        # get the field depending on whether it is sparse or not
        if field_path in self.sparse_fields:
            return self._get_sparse_traj_field(run_idx, traj_idx, field_path)
        else:
            return self._get_contiguous_traj_field(run_idx, traj_idx, field_path)

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
                           map_func=map,
                           traj_sel=None,
                           save_to_hdf5=None, idxs=False, return_results=True,
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

        if return_results:
            results = []

        for result in self.traj_fields_map(func, fields, *args,
                                       map_func=map_func, traj_sel=traj_sel, idxs=True,
                                       debug_prints=debug_prints):

            idx_tup, obs_features = result
            run_idx, traj_idx = idx_tup

            # if we are saving this to the trajectories observables add it as a dataset
            if save_to_hdf5:

                if debug_prints:
                    print("Saving run {} traj {} observables/{}".format(
                        run_idx, traj_idx, field_name))

                # try to get the observables group or make it if it doesn't exist
                try:
                    obs_grp = self.traj(run_idx, traj_idx)['observables']
                except KeyError:

                    if debug_prints:
                        print("Group uninitialized. Initializing.")

                    obs_grp = self.traj(run_idx, traj_idx).create_group('observables')

                # try to create the dataset
                try:
                    obs_grp.create_dataset(field_name, data=obs_features)
                # if it fails we either overwrite or raise an error
                except RuntimeError:
                    # if we are in a permissive write mode we delete the
                    # old dataset and add the new one, overwriting old data
                    if self.mode in ['w', 'w-', 'x', 'r+']:

                        if debug_prints:
                            print("Dataset already present. Overwriting.")

                        del obs_grp[field_name]
                        obs_grp.create_dataset(field_name, data=obs_features)
                    # this will happen in 'c' and 'c-' modes
                    else:
                        raise RuntimeError(
                            "Dataset already exists and file is in concatenate mode ('c' or 'c-')")

            # also return it if requested
            if return_results:
                if idxs:
                    results.append(( idx_tup, obs_features))
                else:
                    results.append(obs_features)

        if return_results:
            return results

    def resampling_records(self, run_idx):

        res_grp = self.resampling_records_grp(run_idx)
        decision_enum = self.decision_enum(run_idx)

        res_recs = []
        for dec_name, dec_id in self.decision_enum(run_idx).items():

            # if this is a decision with variable length instructions
            if self._instruction_varlength_flags(run_idx)[dec_name][()]:
                dec_grp = res_grp[dec_name]
                # go through each dataset of different records
                for init_length in dec_grp['_initialized'][:]:
                    rec_ds = dec_grp['{}'.format(init_length)]

                    # make tuples for the decision and the records
                    tups = zip((dec_id for i in range(rec_ds.shape[0])), rec_ds)

                    # save the tuples
                    res_recs.extend(list(tups))
            else:
                rec_ds = res_grp[dec_name]
                # make tuples for the decision and the records
                tups = zip((dec_id for i in range(rec_ds.shape[0])), rec_ds)

                # save the tuples
                res_recs.extend(list(tups))

        return res_recs

    def resampling_records_dataframe(self, run_idx):
        records = self.resampling_records(run_idx)

        records = [(tup[0], *tup[1]) for tup in records]

        colnames = ['decision_id', 'cycle_idx', 'step_idx', 'walker_idx', 'instruction_record']

        df = pd.DataFrame(data=records, columns=colnames)
        return df


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

def _json_top_atom_count(json_str):
    top_d = json.loads(json_str)
    atom_count = 0
    atom_count = 0
    for chain in top_d['chains']:
        for residue in chain['residues']:
            atom_count += len(residue['atoms'])

    return atom_count

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
    for field, value in traj_data.items():

        # don't check observables
        if field in ['observables']:
            continue

        # check to make sure the value actually has something in it
        if (value is not None) and len(value) > 0:
            fields.update([field])

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

def iter_field_paths(grp):
    field_paths = []
    for field_name in grp:
        if isinstance(grp[field_name], h5py.Group):
            for subfield in grp[field_name]:
                field_paths.append(field_name + '/' + subfield)
        else:
            field_paths.append(field_name)
    return field_paths
