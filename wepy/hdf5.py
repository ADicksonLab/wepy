from collections import Sequence
import itertools as it
import json
from warnings import warn
from copy import copy

import numpy as np
import h5py

from wepy.util.mdtraj import mdtraj_to_json_topology, json_to_mdtraj_topology
from wepy.util.util import traj_box_vectors_to_lengths_angles, json_top_atom_count

# optional dependencies
try:
    import mdtraj as mdj
except ModuleNotFoundError:
    warn("mdtraj is not installed and that functionality will not work", RuntimeWarning)

try:
    import pandas as pd
except ModuleNotFoundError:
    warn("pandas is not installed and that functionality will not work", RuntimeWarning)


## Constants for the main trajectories data group
# Constants
N_DIMS = 3

TRAJECTORIES = 'trajectories'

# strings for trajectory fields
POSITIONS = 'positions'
BOX_VECTORS = 'box_vectors'
VELOCITIES = 'velocities'
FORCES = 'forces'
TIME = 'time'
KINETIC_ENERGY = 'kinetic_energy'
POTENTIAL_ENERGY = 'potential_energy'
BOX_VOLUME = 'box_volume'
OBSERVABLES = 'observables'

PARAMETERS = 'parameters'
PARAMETER_DERIVATIVES = 'parameter_derivatives'

WEIGHTS = 'weights'

# lists of keys etc.
TRAJ_DATA_FIELDS = (POSITIONS, TIME, BOX_VECTORS, VELOCITIES,
                    FORCES, KINETIC_ENERGY, POTENTIAL_ENERGY,
                    BOX_VOLUME, PARAMETERS, PARAMETER_DERIVATIVES,
                    OBSERVABLES)



# defaults for the rank (length of shape vector) for certain
# unchanging data fields. This is the rank of the feawture not the
# array that will acutally be saved int he hdf5. That will always be
# one more than the rank of the feature.
FIELD_FEATURE_RANKS = ((POSITIONS, 2),
                       (TIME, 1),
                       (BOX_VECTORS, 2),
                       (VELOCITIES, 2),
                       (FORCES, 2),
                       (BOX_VOLUME, 1),
                       (KINETIC_ENERGY, 1),
                       (POTENTIAL_ENERGY, 1),
                      )

# defaults for the shapes for those fields they can be given to.
FIELD_FEATURE_SHAPES = ((TIME, (1,)),
                        (BOX_VECTORS, (3,3)),
                        (BOX_VOLUME, (1,)),
                        (KINETIC_ENERGY, (1,)),
                        (POTENTIAL_ENERGY, (1,)),
                        )

WEIGHT_SHAPE = (1,)

FIELD_FEATURE_DTYPES = ((POSITIONS, np.float),
                        (VELOCITIES, np.float),
                        (FORCES, np.float),
                        (TIME, np.float),
                        (BOX_VECTORS, np.float),
                        (BOX_VOLUME, np.float),
                        (KINETIC_ENERGY, np.float),
                        (POTENTIAL_ENERGY, np.float),
                        )


# Positions (and thus velocities and forces) are determined by the
# N_DIMS (which can be customized) and more importantly the number of
# particles which is always different. All the others are always wacky
# and different.
POSITIONS_LIKE_FIELDS = (VELOCITIES, FORCES)

# some fields have more than one dataset associated with them
COMPOUND_DATA_FIELDS = (PARAMETERS, PARAMETER_DERIVATIVES, OBSERVABLES)
COMPOUND_UNIT_FIELDS = (PARAMETERS, PARAMETER_DERIVATIVES, OBSERVABLES)

# some fields can have sparse data, non-sparse data must be all the
# same shape and larger than sparse arrays. Sparse arrays have an
# associated index with them aligning them to the non-sparse datasets
SPARSE_DATA_FIELDS = (VELOCITIES, FORCES, KINETIC_ENERGY, POTENTIAL_ENERGY,
                      BOX_VOLUME, PARAMETERS, PARAMETER_DERIVATIVES, OBSERVABLES)


## Run data records

# the groups of run records
RESAMPLING = 'resampling'
RESAMPLER = 'resampler'
WARPING = 'warping'
PROGRESS = 'progress'
BC = 'boundary_conditions'

CYCLE_IDXS = '_cycle_idxs'

# records can be sporadic or continual. Continual records are
# generated every cycle and are saved every cycle and are for all
# walkers.  Sporadic records are generated conditional on specific
# events taking place and thus may or may not be produced each
# cycle. There also is not a single record for each (cycle, step) like
# there would be for continual ones because they can occur for single
# walkers, boundary conditions, or resamplers.
SPORADIC_RECORDS = (RESAMPLER, WARPING, RESAMPLING)

## Dataset Compliances
# a file which has different levels of keys can be used for
# different things, so we define these collections of keys,
# and flags to keep track of which ones this dataset
# satisfies, ds here stands for "dataset"
COMPLIANCE_TAGS = ['COORDS', 'TRAJ', 'RESTART']

# the minimal requirement (and need for this class) is to associate
# a collection of coordinates to some molecular structure (topology)
COMPLIANCE_REQUIREMENTS = (('COORDS',  (POSITIONS,)),
                           ('TRAJ',    (POSITIONS, TIME, BOX_VECTORS)),
                           ('RESTART', (POSITIONS, TIME, BOX_VECTORS,
                                        VELOCITIES)),
                          )



class WepyHDF5(object):

    def __init__(self, filename, topology=None, mode='x',
                 units=None,
                 sparse_fields=None,
                 feature_shapes=None, feature_dtypes=None,
                 n_dims=None,
                 alt_reps=None, main_rep_idxs=None
    ):
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

        # initialize the list of run record fields that are variable length
        self._run_records_fields_vlen = []


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

        # if we specify an atom subset of the main POSITIONS field
        # we must save them
        self._main_rep_idxs = main_rep_idxs

        # a dictionary specifying other alt_reps to be saved
        if alt_reps is not None:
            self._alt_reps = alt_reps
            # all alt_reps are sparse
            alt_rep_keys = ['alt_reps/{}'.format(key) for key in self._alt_reps.keys()]
            self._sparse_fields.extend(alt_rep_keys)
        else:
            self._alt_reps = {}

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


        # the dtypes for the record group instruction records
        self.run_record_dtypes = {}


        ## OLD
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
        ##

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
        self._set_default_init_field_attributes(n_dims=self._n_dims)

        # save the number of dimensions and number of atoms in settings
        settings_grp.create_dataset('n_dims', data=np.array(self._n_dims))
        settings_grp.create_dataset('n_atoms', data=np.array(self._n_coords))

        # the main rep atom idxs
        settings_grp.create_dataset('main_rep_idxs', data=self._main_rep_idxs, dtype=np.int)

        # alt_reps settings
        alt_reps_idxs_grp = settings_grp.create_group("alt_reps_idxs")
        for alt_rep_name, idxs in self._alt_reps.items():
            alt_reps_idxs_grp.create_dataset(alt_rep_name, data=idxs, dtype=np.int)

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
                # make a json string of the datatype that can be read
                # in again, we call np.dtype again because there is no
                # np.float.descr attribute
                dt_str = json.dumps(np.dtype(field_dtype).descr)

            dtypes_grp.create_dataset(field_path, data=dt_str)

        # initialize the units group
        unit_grp = self._h5.create_group('units')

        # if units were not given set them all to None
        if units is None:
            units = {}
            for field_path in self._field_feature_shapes.keys():
                units[field_path] = None

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


        # get the number of coordinates of positions. If there is a
        # main_reps then we have to set the number of atoms to that,
        # if not we count the number of atoms in the topology
        if self._main_rep_idxs is None:
            self._n_coords = json_top_atom_count(self.topology)
            self._main_rep_idxs = list(range(self._n_coords))
        else:
            self._n_coords = len(self._main_rep_idxs)

        # get the number of dimensions as a default
        if n_dims is None:
            self._n_dims = N_DIMS

        # feature shapes for positions and positions-like fields are
        # not known at the module level due to different number of
        # coordinates (number of atoms) and number of dimensions
        # (default 3 spatial). We set them now that we know this
        # information.
        # add the postitions shape
        field_feature_shapes[POSITIONS] = (self._n_coords, self._n_dims)
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
        """The topology for the full simulated system. May not be the main
        representation in the POSITIONS field; for that use the
        `topology` method.

        """
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

    def get_mdtraj_topology(self, alt_rep=POSITIONS):
        """Get an MDTraj `Topology` object for a subset of the atoms in the
        positions of a particular representation. By default gives the
        topology for the main 'positions' field (when alt_rep
        'positions'). To get the full topology the file was
        initialized with set `alt_rep` to `None`. Topologies for
        alternative representations (subfields of 'alt_reps') can be
        obtained by passing in the key for that alt_rep. For example,
        'all_atoms' for the field in alt_reps called 'all_atoms'.

        """

        full_mdj_top = json_to_mdtraj_topology(self.topology)
        if alt_rep is None:
            return full_mdj_top
        elif alt_rep == POSITIONS:
            # get the subset topology for the main rep idxs
            return full_mdj_top.subset(self.main_rep_idxs)
        elif alt_rep in self.alt_rep_idxs:
            # get the subset for the alt rep
            return full_mdj_top.subset(self.alt_rep_idxs[alt_rep])
        else:
            raise ValueError("alt_rep {} not found".format(alt_rep))

    def get_topology(self, alt_rep=POSITIONS):
        """Get a JSON topology for a subset of the atoms in the
        positions of a particular representation. By default gives the
        topology for the main 'positions' field (when alt_rep
        'positions'). To get the full topology the file was
        initialized with set `alt_rep` to `None`. Topologies for
        alternative representations (subfields of 'alt_reps') can be
        obtained by passing in the key for that alt_rep. For example,
        'all_atoms' for the field in alt_reps called 'all_atoms'.

        """

        mdj_top = self.get_mdtraj_topology(alt_rep=alt_rep)
        json_top = mdtraj_to_json_topology(mdj_top)

        return json_top

    @property
    def sparse_fields(self):
        return self.h5['_settings/sparse_fields'][:]

    @property
    def main_rep_idxs(self):
        if '/_settings/main_rep_idxs' in self.h5:
            return self.h5['/_settings/main_rep_idxs'][:]
        else:
            return None

    @property
    def alt_rep_idxs(self):
        idxs_grp = self.h5['/_settings/alt_reps_idxs']
        return {name : ds[:] for name, ds in idxs_grp.items()}

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

    def traj_n_frames(self, run_idx, traj_idx):
        return self.traj(run_idx, traj_idx)[POSITIONS].shape[0]

    def run_n_frames(self, run_idx):
        return self.traj_n_frames(run_idx, 0)

    def run_n_cycles(self, run_idx):
        return self.run_n_frames(run_idx)

    def run_trajs(self, run_idx):
        return self._h5['runs/{}/trajectories'.format(run_idx)]

    def n_run_trajs(self, run_idx):
        return len(self._h5['runs/{}/trajectories'.format(run_idx)])

    def run_traj_idxs(self, run_idx):
        return range(len(self._h5['runs/{}/trajectories'.format(run_idx)]))

    def run_traj_idx_tuples(self, runs=None):
        tups = []
        if runs is None:
            run_idxs = self.run_idxs
        else:
            run_idxs = runs
        for run_idx in run_idxs:
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

    # application level methods for setting the fields for run record
    # groups given the objects themselves
    def init_run_resampling(self, run_idx, resampler_type):

        fields = resampler_type.resampling_fields()

        grp = self.init_run_record_grp(run_idx, RESAMPLING, fields)

        return grp

    def init_run_resampler(self, run_idx, resampler_type):

        fields = resampler_type.resampler_fields()

        grp = self.init_run_record_grp(run_idx, RESAMPLER, fields)

        return grp

    def init_run_warping(self, run_idx, bc_type):

        fields = bc_type.warping_fields()
        grp = self.init_run_record_grp(run_idx, WARPING, fields)

        return grp

    def init_run_progress(self, run_idx, bc_type):

        fields = bc_type.progress_fields()

        grp = self.init_run_record_grp(run_idx, PROGRESS, fields)

        return grp

    def init_run_bc(self, run_idx, bc_type):

        fields = bc_type.bc_fields()

        grp = self.init_run_record_grp(run_idx, BC, fields)

        return grp

    # application level methods for initializing the run records
    # groups with just the fields and without the objects
    def init_run_fields_resampling(self, run_idx, fields):

        grp = self.init_run_record_grp(run_idx, RESAMPLING, fields)

        return grp

    def init_run_fields_resampler(self, run_idx, fields):

        grp = self.init_run_record_grp(run_idx, RESAMPLER, fields)

        return grp

    def init_run_fields_warping(self, run_idx, fields):

        grp = self.init_run_record_grp(run_idx, WARPING, fields)

        return grp

    def init_run_fields_progress(self, run_idx, fields):

        grp = self.init_run_record_grp(run_idx, PROGRESS, fields)

        return grp

    def init_run_fields_bc(self, run_idx, fields):

        grp = self.init_run_record_grp(run_idx, BC, fields)

        return grp


    def init_run_record_grp(self, run_idx, run_record_key, fields):

        # initialize the record group based on whether it is sporadic
        # or continual

        if self._is_sporadic_records(run_record_key):
            grp = self._init_run_sporadic_record_grp(run_idx, run_record_key,
                                                     fields)
        else:
            grp = self._init_run_continual_record_grp(run_idx, run_record_key,
                                                      fields)

    def _init_run_sporadic_record_grp(self, run_idx, run_record_key, fields):

        # create the group
        run_grp = self.run(run_idx)
        record_grp = run_grp.create_group(run_record_key)

        # initialize the cycles dataset that maps when the records
        # were recorded
        record_grp.create_dataset(CYCLE_IDXS, (0,), dtype=np.int,
                                  maxshape=(None,))

        # for each field simply create the dataset
        for field_name, field_shape, field_dtype in fields:

            # initialize this field
            self._init_run_records_field(run_idx, run_record_key,
                                         field_name, field_shape, field_dtype)

        return record_grp


    def _init_run_continual_record_grp(self, run_idx, run_record_key, fields):

        # create the group
        run_grp = self.run(run_idx)
        record_grp = run_grp.create_group(run_record_key)

        # for each field simply create the dataset
        for field_name, field_shape, field_dtype in fields:

            self._init_run_records_field(run_idx, run_record_key,
                                         field_name, field_shape, field_dtype)

        return record_grp

    def _init_run_records_field(self, run_idx, run_record_key,
                                field_name, field_shape, field_dtype):

        record_grp = self.run(run_idx)[run_record_key]

        # check if it is variable length
        if field_shape is Ellipsis:
            # make a special dtype that allows it to be
            # variable length
            vlen_dt = h5py.special_dtype(vlen=field_dtype)

            # this is only allowed to be a single dimension
            # since no real shape was given
            dset = record_grp.create_dataset(field_name, (0,), dtype=vlen_dt,
                                        maxshape=(None,))

            # add it to the listing of records fields with variable lengths
            self._run_records_fields_vlen.append('{}/{}'.format(run_record_key, field_name))

        # its not just make it normally
        else:
            # create the group
            dset = record_grp.create_dataset(field_name, (1, *field_shape), dtype=field_dtype,
                                      maxshape=(None, *field_shape))

        return dset

    def _is_sporadic_records(self, run_record_key):

        # assume it is continual and check if it is in the sporadic groups
        if run_record_key in SPORADIC_RECORDS:
            return True
        else:
            return False

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

    def traj_n_frames(self, run_idx, traj_idx):
        return self.traj(run_idx, traj_idx)[POSITIONS].shape[0]

    def add_traj(self, run_idx, data, weights=None, sparse_idxs=None, metadata=None):

        # convenient alias
        traj_data = data

        # initialize None kwargs
        if sparse_idxs is None:
            sparse_idxs = {}
        if metadata is None:
            metadata = {}

        # positions are mandatory
        assert POSITIONS in traj_data, "positions must be given to create a trajectory"
        assert isinstance(traj_data[POSITIONS], np.ndarray)

        n_frames = traj_data[POSITIONS].shape[0]

        # if weights are None then we assume they are 1.0
        if weights is None:
            weights = np.ones((n_frames, 1), dtype=float)
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
        assert traj_data[POSITIONS].shape[1] == self.n_atoms, \
            "positions given have different number of atoms: {}, should be {}".format(
                traj_data[POSITIONS].shape[1], self.n_atoms)
        assert traj_data[POSITIONS].shape[2] == self.n_dims, \
            "positions given have different number of dims: {}, should be {}".format(
                traj_data[POSITIONS].shape[2], self.n_dims)

        # add datasets to the traj group

        # weights
        traj_grp.create_dataset(WEIGHTS, data=weights, maxshape=(None, *WEIGHT_SHAPE))
        # positions

        positions_shape = traj_data[POSITIONS].shape

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

    def _extend_dataset(self, dset_path, new_data):
        dset = self.h5[dset_path]
        extend_dataset(dset, new_data)

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
                "input value features have shape {}, expected {}".format(
                    values.shape[1:], field_data.maxshape[1:])

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
        n_new_frames = traj_data[POSITIONS].shape[0]

        n_frames = self.traj_n_frames(run_idx, traj_idx)

        # calculate the new sparse idxs for sparse fields that may be
        # being added
        sparse_idxs = np.array(range(n_frames, n_frames + n_new_frames))

        # get the trajectory group
        traj_grp = self._h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]

        ## weights

        # if weights are None then we assume they are 1.0
        if weights is None:
            weights = np.ones((n_new_frames, 1), dtype=float)
        else:
            assert isinstance(weights, np.ndarray), "weights must be a numpy.ndarray"
            assert weights.shape[0] == n_new_frames,\
                "weights and the number of frames must be the same length"

        # add the weights
        weights_ds = traj_grp[WEIGHTS]

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
                        if group_name == OBSERVABLES:
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

        # a common mistake when writing a new resampler is to not
        # return the correct type for the cycle resampling
        # records. `cycle_resamplingrecords` is supposed to be a list
        # of lists where the elements of the outer list represent the
        # "steps" a resampler can take. This is because a single
        # resampling is allowed to clone a walker and then clone that
        # walker again, which will always take multiple "steps", the
        # order of the steps is the order in which the steps should be
        # performed. A step is the list of actual ResamplingRecord
        # objects for each walker. These are in the order that the
        # walkers were passed to the `resample` function. Thus even if
        # there is only one resampling "step" this should be a list of
        # a single list, i.e. the only step. It is not expected many
        # resamplers will use multiple steps we though it nonetheless
        # prudent to allow for this potential behavior because the
        # handling of such data in the HDF5 (and subsequent analysis
        # steps) changes considerably with this consideration. In the
        # future perhaps an automatic converter will detect this and
        # modify it if we find it necessary, however it is more likely
        # that this will just hide more important errors,
        # i.e. returning only one step when you meant to return
        # more. So for now we just provide a custom error to help in
        # debugging because this is a common mistake.
        if not isinstance(cycle_resampling_records[0], list):
            raise TypeError("'cycle_resampling_records' must be a list of lists "
                            "(i.e. list of steps), but a list of types {} were given.".format(
                                type(cycle_resampling_records[0])))

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
                self.add_resampling_record(run_idx, decision_key,
                                            cycle_idx, step_idx, walker_idx,
                                            instruct_record)

        # update the current cycle idx
        self._current_resampling_rec_cycle += 1

    def add_resampling_record(self, run_idx, decision_key,
                               cycle_idx, step_idx, walker_idx,
                               instruct_record):

        run_grp = self.run(run_idx)
        varlengths = self._instruction_varlength_flags(run_idx)

        # we need to treat variable length decision instructions
        # differently than fixed width

        # test whether the decision has a variable width instruction format
        if varlengths[decision_key][()]:
            self._add_varlength_resampling_record(run_idx, decision_key,
                                                   cycle_idx, step_idx, walker_idx,
                                                   instruct_record)
        else:
            self._add_fixed_width_instruction_record(run_idx, decision_key,
                                                     cycle_idx, step_idx, walker_idx,
                                                     instruct_record)

    def _add_varlength_resampling_record(self, run_idx, decision_key,
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
            dt = _make_numpy_varlength_resampling_dtype(instruct_dtype_tokens,
                                                                  instruct_width)
            # make the dataset with the given dtype
            dset = instruct_grp.create_dataset(str(len(instruct_record)), (0,),
                                               dtype=dt, maxshape=(None,))

            # record it in the initialized list
            widths_dset.resize( (widths_dset.shape[0] + 1, ) )
            widths_dset[-1] = instruct_width

        # if it exists get a reference to the dataset according to length
        else:
            dset = instruct_grp[str(len(instruct_record))]

        # make the complete record to add to the dataset, need to
        # convert the instruct record to a normal tuple instead of the
        # custom variable length record class
        record = (cycle_idx, step_idx, walker_idx, tuple(instruct_record))
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
                print(key)
                # if the dataset is of variable length handle it specially
                if self.resampling_aux_shapes[key] is Ellipsis:
                    # resize the array but it is only of rank because
                    # of variable length data
                    dset.resize( (dset.shape[0] + 1, ) )
                    # does not need to be wrapped in another dimension
                    # like for other aux data
                    dset[-1] = aux_data

                else:
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


    def _instruction_varlength_flags(self, run_idx):
        varlength_grp = self._h5['runs/{}/resampling/decision/variable_length'.format(run_idx)]
        varlength = {}
        for decision_name, dset in varlength_grp.items():
            varlength[decision_name] = dset

        return varlength

    ## application level append methods for run records groups

    def append_warping_records(self, run_idx, cycle_idx, warping_data):
        self.append_records_group(run_idx, WARPING, cycle_idx, warping_data)

    def append_bc_records(self, run_idx, cycle_idx, bc_data):
        self.append_records_group(run_idx, BC, cycle_idx, bc_data)

    def append_progress_records(self, run_idx, cycle_idx, progress_data):
        self.append_records_group(run_idx, PROGRESS, cycle_idx, progress_data)

    def append_resampling_records(self, run_idx, cycle_idx, resampling_data):
        self.append_records_group(run_idx, RESAMPLING, cycle_idx, resampling_data)

    def append_resampler_records(self, run_idx, cycle_idx, resampler_data):
        self.append_records_group(run_idx, RESAMPLER, cycle_idx, resampler_data)

    def append_records_group(self, run_idx, run_record_key, cycle_idx, fields_data):
        """Append data for a whole records group, that is every field
        dataset. This must have the cycle index for the data it is
        appending as this is done for sporadic and continual datasets.

        """

        record_grp = self.records_grp(run_idx, run_record_key)


        # if it is sporadic add the cycle idx
        if self._is_sporadic_records(run_record_key):

            # get the cycle idxs dataset
            record_cycle_idxs_ds = record_grp[CYCLE_IDXS]

            # then we add the cycle to the cycle_idxs
            record_cycle_idxs_ds.resize( (record_cycle_idxs_ds.shape[0] + 1,
                                       *record_cycle_idxs_ds.shape[1:]) )
            # add the new data
            record_cycle_idxs_ds[-1:, ...] = np.array([cycle_idx])

        # then add all the data for the field
        for record_dict in fields_data:
            for field_name, field_data in record_dict.items():
                self._extend_run_record_data_field(run_idx, run_record_key,
                                                   field_name, np.array([field_data]))

    def _extend_run_record_data_field(self, run_idx, run_record_key,
                                          field_name, field_data):
        """Adds data for a single field dataset in a run records group. This
        is done without paying attention to whether it is sporadic or
        continual and is supposed to be only the data write method.

        """

        records_grp = self.h5['runs/{}/{}'.format(run_idx, run_record_key)]
        field = records_grp[field_name]

        # make sure this is a feature vector
        assert len(field_data.shape) > 1, \
            "field_data must be a feature vector with the same number of dimensions as the number"

        # of datase new frames
        n_new_frames = field_data.shape[0]

        # check whether it is a variable length record
        field_path = '{}/{}'.format(run_record_key, field_name)
        if field_path in self._run_records_fields_vlen:

            if all([i == 0 for i in field.shape]):
                # initialize this array
                # if it is empty resize it to make an array the size of
                # the new field_data with the maxshape for the feature
                # dimensions
                field.resize( (n_new_frames,) )

                # set the new data to this
                for i, row in enumerate(field_data):
                    field[i] = row

            else:
                import ipdb; ipdb.set_trace()
                # resize the array but it is only of rank because
                # of variable length data
                field.resize( (field.shape[0] + n_new_frames, ) )

                # add each row to the newly made space
                for i, row in enumerate(field_data):
                    field[(field.shape[0] - 1) + i] = row

        else:
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
                # append to the dataset on the first dimension, keeping the
                # others the same, these must be feature vectors and therefore
                # must exist
                field.resize( (field.shape[0] + n_new_frames, *field.shape[1:]) )
                # add the new data
                field[-n_new_frames:, ...] = field_data


    def add_cycle_bc_aux_data(self, run_idx, bc_aux_data):

        data_grp = self._h5['runs/{}/boundary_conditions/aux_data'.format(run_idx)]

        # add the data for each aux data type
        for key, aux_data in bc_aux_data.items():

            # if the datasets were initialized just add the new data
            if key in self._bc_aux_init:

                # get the dataset
                dset = data_grp[key]

                # if the dataset is of variable length handle it specially
                if self.bc_aux_shapes[key] is Ellipsis:
                    # resize the array but it is only of rank because
                    # of variable length data
                    dset.resize( (dset.shape[0] + 1, ) )
                    # does not need to be wrapped in another dimension
                    # like for other aux data
                    dset[-1] = aux_data

                else:
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

    def _get_contiguous_traj_field(self, run_idx, traj_idx, field_path, frames=None):

        full_path = "/runs/{}/trajectories/{}/{}".format(run_idx, traj_idx, field_path)

        if frames is None:
            field = self._h5[full_path][:]
        else:
            field = self._h5[full_path][frames]

        return field

    def _get_sparse_traj_field(self, run_idx, traj_idx, field_path, frames=None):

        traj_path = "/runs/{}/trajectories/{}".format(run_idx, traj_idx)
        traj_grp = self.h5[traj_path]
        field = traj_grp[field_path]

        n_frames = traj_grp[POSITIONS].shape[0]

        if frames is None:
            data = field['data'][:]
            sparse_idxs = field['_sparse_idxs'][:]

            filled_data = np.full( (n_frames, *data.shape[1:]), np.nan)
            filled_data[sparse_idxs] = data

            mask = np.full( (n_frames, *data.shape[1:]), True)
            mask[sparse_idxs] = False

            masked_array = np.ma.masked_array(filled_data, mask=mask)

        else:
            # the empty arrays the size of the number of requested frames
            filled_data = np.full( (len(frames), *field['data'].shape[1:]), np.nan)
            mask = np.full( (len(frames), *field['data'].shape[1:]), True )

            sparse_idxs = field['_sparse_idxs'][:]

            # we get a boolean array of the rows of the data table
            # that we are to slice from
            sparse_frame_idxs = np.argwhere(np.isin(sparse_idxs, frames))

            # take the data which exists and is part of the frames
            # selection, and put it into the filled data where it is
            # supposed to be
            filled_data[np.isin(frames, sparse_idxs)] = field['data'][list(sparse_frame_idxs)]

            # unmask the present values
            mask[np.isin(frames, sparse_idxs)] = False

            masked_array = np.ma.masked_array(filled_data, mask=mask)

        return masked_array

    def get_traj_field(self, run_idx, traj_idx, field_path, frames=None):
        """Returns a numpy array for the given field."""

        traj_path = "/runs/{}/trajectories/{}".format(run_idx, traj_idx)

        # if the field doesn't exist return None
        if not field_path in self._h5[traj_path]:
            raise KeyError("key for field {} not found".format(field_path))
            # return None

        # get the field depending on whether it is sparse or not
        if field_path in self.sparse_fields:
            return self._get_sparse_traj_field(run_idx, traj_idx, field_path,
                                               frames=frames)
        else:
            return self._get_contiguous_traj_field(run_idx, traj_idx, field_path,
                                                   frames=frames)

    def get_trace_fields(self, frame_tups, fields):
        frame_fields = {field : [] for field in fields}
        for run_idx, traj_idx, cycle_idx in frame_tups:
            for field in fields:
                frame_field = self.get_traj_field(run_idx, traj_idx, field, frames=[cycle_idx])
                # the first dimension doesn't matter here since we
                # only get one frame at a time.
                frame_fields[field].append(frame_field[0])

        # combine all the parts of each field into single arrays
        for field in fields:
            frame_fields[field] = np.array(frame_fields[field])

        return frame_fields

    def get_run_trace_fields(self, run_idx, frame_tups, fields):
        frame_fields = {field : [] for field in fields}
        for cycle_idx, traj_idx in frame_tups:
            for field in fields:
                frame_field = self.get_traj_field(run_idx, traj_idx, field, frames=[cycle_idx])
                # the first dimension doesn't matter here since we
                # only get one frame at a time.
                frame_fields[field].append(frame_field[0])

        # combine all the parts of each field into single arrays
        for field in fields:
            frame_fields[field] = np.array(frame_fields[field])

        return frame_fields

    def _add_run_field(self, run_idx, field_path, data, sparse_idxs=None):
        """ Add a field to your trajectories runs"""

        # check that the data has the correct number of trajectories
        assert len(data) == self.n_run_trajs(run_idx),\
            "The number of trajectories in data, {}, is different than the number"\
            "of trajectories in the run, {}.".format(len(data), self.n_run_trajs(run_idx))

        # for each trajectory check that the data is compliant
        for traj_idx, traj_data in enumerate(data):
            # check that the number of frames is not larger than that for the run
            if traj_data.shape[0] > self.run_n_frames(run_idx):
                raise ValueError("The number of frames in data for traj {} , {},"
                                  "is larger than the number of frames"
                                  "for this run, {}.".format(
                                          traj_idx, data.shape[1], self.run_n_frames(run_idx)))


            # if the number of frames given is the same or less than
            # the number of frames in the run
            elif (traj_data.shape[0] <= self.run_n_frames(run_idx)):

                # if sparse idxs were given we check to see there is
                # the right number of them
                if sparse_idxs is not None:
                    #  and that they match the number of frames given
                    if data.shape[0] != len(sparse_idxs[traj_idx]):

                        raise ValueError("The number of frames provided for traj {}, {},"
                                          "was less than the total number of frames, {},"
                                          "but an incorrect number of sparse idxs were supplied, {}."\
                                         .format(traj_idx, traj_data.shape[0],
                                            self.run_n_frames(run_idx), len(sparse_idxs[traj_idx])))


                # if there were strictly fewer frames given and the
                # sparse idxs were not given we need to raise an error
                elif (traj_data.shape[0] < self.run_n_frames(run_idx)):
                    raise ValueError("The number of frames provided for traj {}, {},"
                                      "was less than the total number of frames, {},"
                                      "but sparse_idxs were not supplied.".format(
                                              traj_idx, traj_data.shape[0],
                                              self.run_n_frames(run_idx)))

        # add it to each traj
        for i, idx_tup in enumerate(self.run_traj_idx_tuples([run_idx])):
            if sparse_idxs is None:
                self._add_traj_field_data(*idx_tup, field_path, data[i])
            else:
                self._add_traj_field_data(*idx_tup, field_path, data[i],
                                          sparse_idxs=sparse_idxs[i])

    def _add_field(self, field_path, data, sparse_idxs=None):

        for i, run_idx in enumerate(self.run_idxs):
            if sparse_idxs is not None:
                self._add_run_field(run_idx, field_path, data[i], sparse_idxs=sparse_idxs[i])
            else:
                self._add_run_field(run_idx, field_path, data[i])

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

    def iter_run_trajs(self, run_idx, idxs=False):
        run_sel = self.run_traj_idx_tuples([run_idx])
        return self.iter_trajs(idxs=idxs, traj_sel=run_sel)

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
                           *args)

        if idxs:
            if traj_sel is None:
                traj_sel = self.run_traj_idx_tuples()
            return zip(traj_sel, results)
        else:
            return results

    def add_run_observable(self, run_idx, observable_name, data, sparse_idxs=None):
        obs_path = "{}/{}".format(OBSERVABLES, observable_name)

        self._add_run_field(run_idx, obs_path, data, sparse_idxs=sparse_idxs)


    def add_observable(self, observable_name, data, sparse_idxs=None):
        obs_path = "{}/{}".format(OBSERVABLES, observable_name)

        self._add_field(obs_path, data, sparse_idxs=sparse_idxs)

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
                "`save_to_hdf5` should be the field name to save the data in the `observables`"\
                " group in each trajectory"
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
                    obs_grp = self.traj(run_idx, traj_idx)[OBSERVABLES]
                except KeyError:

                    if debug_prints:
                        print("Group uninitialized. Initializing.")

                    obs_grp = self.traj(run_idx, traj_idx).create_group(OBSERVABLES)

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

    def records_grp(self, run_idx, run_record_key):
        path = "runs/{}/{}".format(run_idx, run_record_key)
        return self.h5[path]

    def resampling_grp(self, run_idx):
        return self.records_grp(run_idx, RESAMPLING)

    def resampler_grp(self, run_idx):
        return self.records_grp(run_idx, RESAMPLER)

    def warping_grp(self, run_idx):
        return self.records_grp(run_idx, WARPING)

    def bc_grp(self, run_idx):
        return self.records_grp(run_idx, BC)

    def progress_grp(self, run_idx):
        return self.records_grp(run_idx, PROGRESS)

    def resampling_records(self, run_idx, sort=True):

        res_grp = self.resampling_records_grp(run_idx)
        decision_enum = self.decision_enum(run_idx)

        res_recs = []
        for dec_name, dec_id in self.decision_enum(run_idx).items():

            # if this is a decision with variable length instructions
            if self._instruction_varlength_flags(run_idx)[dec_name][()]:
                dec_grp = res_grp[dec_name]
                recs = []
                # go through each dataset of different records
                for init_length in dec_grp['_initialized'][:]:
                    rec_ds = dec_grp['{}'.format(init_length)]

                    # reorder for to match the field order
                    recs = [rec_ds[field] for field in [CYCLE_IDXS, STEP, WALKER]]
                    # fill up a column for the decision id
                    recs.append(np.full((rec_ds.shape[0],), dec_id))
                    # put the instructions last
                    recs.append(rec_ds[INSTRUCTION])

                    # make an array
                    recs = np.array(recs)
                    # swap the axes so it is row-oriented
                    recs = np.swapaxes(recs, 0, 1)

                    # return to a list of lists and save into the full record
                    res_recs.append([tuple(row) for row in recs])

            else:
                rec_ds = res_grp[dec_name]

                # reorder for to match the field order
                recs = [rec_ds[field] for field in [CYCLE_IDXS, STEP, WALKER]]
                # fill up a column for the decision id
                recs.append(np.full((rec_ds.shape[0],), dec_id))
                # put the instructions last
                recs.append(rec_ds[INSTRUCTION])

                # make an array
                recs = np.array(recs)
                # swap the axes so it is row-oriented
                recs = np.swapaxes(recs, 0, 1)

                # return to a list of lists and save into the full record
                res_recs.append([tuple(row) for row in recs])

        # combine them all into one list
        res_recs = list(it.chain(*res_recs))

        if sort:
            res_recs.sort()

        return res_recs

    def resampling_records_dataframe(self, run_idx):

        return pd.DataFrame(data=self.resampling_records(run_idx), columns=RESAMPLING_RECORD_FIELDS)

    @staticmethod
    def resampling_panel(resampling_records, is_sorted=False):

        resampling_panel = []

        # if the records are not sorted this must be done:
        if not is_sorted:
            resampling_records.sort()

        # iterate through the resampling records
        rec_it = iter(resampling_records)
        cycle_idx = 0
        cycle_recs = []
        stop = False
        while not stop:

            # iterate through records until either there is none left or
            # until you get to the next cycle
            cycle_stop = False
            while not cycle_stop:
                try:
                    rec = next(rec_it)
                except StopIteration:
                    # this is the last record of all the records
                    stop = True
                    # this is the last record for the last cycle as well
                    cycle_stop = True
                    # alias for the current cycle
                    curr_cycle_recs = cycle_recs
                else:
                    # if the resampling record retrieved is from the next
                    # cycle we finish the last cycle
                    if rec[RESAMPLING_RECORD_FIELDS.index(CYCLE_IDXS)] > cycle_idx:
                        cycle_stop = True
                        # save the current cycle as a special
                        # list which we will iterate through
                        # to reduce down to the bare
                        # resampling record
                        curr_cycle_recs = cycle_recs

                        # start a new cycle_recs for the record
                        # we just got
                        cycle_recs = [rec]
                        cycle_idx += 1

                if not cycle_stop:
                    cycle_recs.append(rec)

                else:

                    # we need to break up the records in the cycle into steps
                    cycle_table = []

                    # temporary container for the step we are working on
                    step_recs = []
                    step_idx = 0
                    step_stop = False
                    cycle_it = iter(curr_cycle_recs)
                    while not step_stop:
                        try:
                            cycle_rec = next(cycle_it)
                        # stop the step if this is the last record for the cycle
                        except StopIteration:
                            step_stop = True
                            # alias for the current step
                            curr_step_recs = step_recs

                        # or if the next stop index has been obtained
                        else:
                            if cycle_rec[RESAMPLING_RECORD_FIELDS.index(STEP)] > step_idx:
                                step_stop = True
                                # save the current step as a special
                                # list which we will iterate through
                                # to reduce down to the bare
                                # resampling record
                                curr_step_recs = step_recs

                                # start a new step_recs for the record
                                # we just got
                                step_recs = [cycle_rec]
                                step_idx += 1


                        if not step_stop:
                            step_recs.append(cycle_rec)
                        else:
                            # go through the walkers for this step since it is completed
                            step_row = [None for i in range(len(curr_step_recs))]
                            for walker_rec in curr_step_recs:

                                # collect data from the record
                                walker_idx = walker_rec[
                                    RESAMPLING_RECORD_FIELDS.index(WALKER)]
                                decision_id = walker_rec[
                                    RESAMPLING_RECORD_FIELDS.index('decision_id')]
                                instruction = walker_rec[
                                    RESAMPLING_RECORD_FIELDS.index(INSTRUCTION)]

                                # set the resampling record for the walker in the step records
                                step_row[walker_idx] = (decision_id, instruction)

                            # add the records for this step to the cycle table
                            cycle_table.append(step_row)

            # add the table for this cycles records to the parent panel
            resampling_panel.append(cycle_table)

        return resampling_panel


    def run_resampling_panel(self, run_idx):
        return self.resampling_panel(self.resampling_records(run_idx))

    def resampling_records_grp(self, run_idx):
        return self._h5['runs/{}/resampling/records'.format(run_idx)]

    def resampling_aux_data_fields(self, run_idx):

        return list(self.resampling_aux_data_grp(run_idx))

    def resampling_aux_data_field(self, run_idx, field):

        aux_grp = self.resampling_aux_data_grp(run_idx)

        return aux_grp[field][:]

    def resampling_aux_data(self, run_idx):

        fields_data = {}
        for field in self.resampling_aux_data_fields(run_idx):
            fields_data[field] = self.resampling_aux_data_grp(run_idx)[field][:]

        return fields_data


    def warp_records(self, run_idx):

        warp_grp = self.warp_grp(run_idx)

        return warp_grp['records'][:]

    def warp_records_dataframe(self, run_idx):

        return pd.DataFrame(data=self.warp_records(run_idx), columns=WARP_RECORD_FIELDS)

    def warp_aux_data_grp(self, run_idx):
        return self._h5['runs/{}/warping/aux_data'.format(run_idx)]

    def warp_aux_data_fields(self, run_idx):
        return list(self.warp_aux_data_grp(run_idx))

    def warp_aux_data_field(self, run_idx, field):

        aux_grp = self.warp_aux_data_grp(run_idx)

        return aux_grp[field][:]

    def warp_aux_data(self, run_idx):

        fields_data = {}
        for field in self.warp_aux_data_fields(run_idx):
            fields_data[field] = self.warp_aux_data_grp(run_idx)[field][:]

        return fields_data

    def bc_records(self, run_idx):

        bc_grp = self.bc_grp(run_idx)

        return bc_grp['records'][:]

    def bc_records_dataframe(self, run_idx):

        return pd.DataFrame(data=self.bc_records(run_idx), columns=BC_RECORD_FIELDS)

    def bc_aux_data_grp(self, run_idx):
        return self._h5['runs/{}/boundary_conditions/aux_data'.format(run_idx)]

    def bc_aux_data_fields(self, run_idx):
        return list(self.bc_aux_data_grp(run_idx))

    def bc_aux_data_field(self, run_idx, field):

        aux_grp = self.bc_aux_data_grp(run_idx)

        return aux_grp[field][:]

    def bc_aux_data(self, run_idx):

        fields_data = {}
        for field in self.bc_aux_data_fields(run_idx):
            fields_data[field] = self.bc_aux_data_grp(run_idx)[field][:]

        return fields_data


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


    def to_mdtraj(self, run_idx, traj_idx, frames=None, alt_rep=None):

        # the default for alt_rep is the main rep
        if alt_rep is None:
            rep_key = POSITIONS
            rep_path = rep_key
        else:
            rep_key = alt_rep
            rep_path = 'alt_reps/{}'.format(alt_rep)

        topology = self.get_mdtraj_topology(alt_rep=rep_key)

        traj_grp = self.traj(run_idx, traj_idx)

        pos_dset = self.get_traj_field(run_idx, traj_idx, rep_path)

        # get the data for all or for the frames specified
        time = None
        box_vectors = None
        if frames is None:
            positions = pos_dset[:]
            try:
                time = traj_grp[TIME][:, 0]
            except KeyError:
                warn("time not in this trajectory, ignoring")
            try:
                box_vectors = traj_grp[BOX_VECTORS][:]
            except KeyError:
                warn("box_vectors not in this trajectory, ignoring")
        else:
            positions = pos_dset[frames]
            try:
                time = traj_grp[TIME][frames][:, 0]
            except KeyError:
                warn("time not in this trajectory, ignoring")
            try:
                box_vectors = traj_grp[BOX_VECTORS][frames]
            except KeyError:
                warn("box_vectors not in this trajectory, ignoring")


        unitcell_lengths, unitcell_angles = traj_box_vectors_to_lengths_angles(box_vectors)

        traj = mdj.Trajectory(positions, topology,
                       time=time,
                       unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)

        return traj

    def trace_to_mdtraj(self, trace, alt_rep=None):

        # the default for alt_rep is the main rep
        if alt_rep is None:
            rep_key = POSITIONS
            rep_path = rep_key
        else:
            rep_key = alt_rep
            rep_path = 'alt_reps/{}'.format(alt_rep)

        topology = self.get_mdtraj_topology(alt_rep=rep_key)

        trace_fields = self.get_trace_fields(trace, [rep_path, BOX_VECTORS])

        unitcell_lengths, unitcell_angles = traj_box_vectors_to_lengths_angles(
                                               trace_fields[BOX_VECTORS])

        cycles = [cycle for run, cycle, walker in trace]
        traj = mdj.Trajectory(trace_fields[rep_key], topology,
                       time=cycles,
                       unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)

        return traj

    def run_trace_to_mdtraj(self, run_idx, trace, alt_rep=None):

        # the default for alt_rep is the main rep
        if alt_rep is None:
            rep_key = POSITIONS
            rep_path = rep_key
        else:
            rep_key = alt_rep
            rep_path = 'alt_reps/{}'.format(alt_rep)

        topology = self.get_mdtraj_topology(alt_rep=rep_key)

        trace_fields = self.get_run_trace_fields(run_idx, trace, [rep_path, BOX_VECTORS])

        unitcell_lengths, unitcell_angles = traj_box_vectors_to_lengths_angles(
                                               trace_fields[BOX_VECTORS])

        cycles = [cycle for cycle, walker in trace]
        traj = mdj.Trajectory(trace_fields[rep_key], topology,
                       time=cycles,
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


## DATA COMPLIANCE STUFF
def _check_data_compliance(traj_data, compliance_requirements=COMPLIANCE_REQUIREMENTS):
    """Given a dictionary of trajectory data it returns the
       COMPLIANCE_TAGS that the data satisfies. """

    # cast the nested tuples to a dictionary if necessary
    compliance_dict = dict(compliance_requirements)

    fields = set()
    for field, value in traj_data.items():

        # don't check observables
        if field in [OBSERVABLES]:
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

class RunCycleSlice(object):

    def __init__(self, run_idx, cycles, wepy_hdf5_file):

        self._h5 = wepy_hdf5_file._h5
        self.run_idx = run_idx
        self.cycles = cycles
        self.mode = wepy_hdf5_file.mode

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
        for traj_idx in self.run_traj_idxs(self.run_idx):
            tups.append((self.run_idx, traj_idx))

        return tups
    def run_cycle_idx_tuples(self):
        tups = []
        for cycle_idx in self.cycles:
             tups.append((self.run_idx, cycle_idx))
        return tups


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

    def iter_cycle_fields(self, fields, cycle_idx, idxs=False, traj_sel=None):
        """Generator for all of the specified non-compound fields
        h5py.Datasets for all trajectories in the dataset across all
        runs. Fields is a list of valid relative paths to datasets in
        the trajectory groups.

        """

        for field in fields:
            dsets = {}
            fields_data = ()
            for idx_tup, traj in self.iter_trajs(idxs=True, traj_sel=traj_sel):
                run_idx, traj_idx = idx_tup
                try:
                    dset = traj[field][cycle_idx]
                    if not isinstance(dset, np.ndarray):
                        dset = np.array([dset])
                    if len(dset.shape)==1:
                        fields_data += (dset,)
                    else:
                        fields_data += ([dset],)
                except KeyError:
                    warn("field \"{}\" not found in \"{}\"".format(field, traj.name), RuntimeWarning)
                    dset = None

            dsets =  np.concatenate(fields_data, axis=0)

            if idxs:
                yield (run_idx, traj_idx, field), dsets
            else:
                yield field, dsets


    def iter_cycles_fields(self, fields, idxs=False, traj_sel=None, debug_prints=False):

        for cycle_idx in self.cycles:
            dsets = {}
            for field, dset in self.iter_cycle_fields(fields, cycle_idx, traj_sel=traj_sel):
                dsets[field] = dset
            if idxs:
                yield (cycle_idx,  dsets)
            else:
                yield dsets


    def traj_cycles_map(self, func, fields, *args, map_func=map, idxs=False, traj_sel=None,
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
            if isinstance(arg, list) and not isinstance(arg, str):
                assert len(arg) == len(self.cycles), "Sequence has fewer"
                mapped_args.append(arg)
            # if it is not a sequence or generator we make a generator out
            # of it to map as inputs
            else:
                mapped_arg = (arg for i in range(len(self.cycles)))
                mapped_args.append(mapped_arg)



        results = map_func(func, self.iter_cycles_fields(fields, traj_sel=traj_sel, idxs=False,
                                                        debug_prints=debug_prints),
                           *mapped_args)

        if idxs:
            if traj_sel is None:
                traj_sel = self.run_cycle_idx_tuples()
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

            #DEBUG enforce this until sparse trajectories are implemented
            assert traj_sel is None, "no selections until sparse trajectory data is implemented"

        idx =0

        for result in self.traj_cycles_map(func, fields, *args,
                                       map_func=map_func, traj_sel=traj_sel, idxs=True,
                                       debug_prints=debug_prints):
            idx_tup, obs_value = result
            run_idx, traj_idx = idx_tup

            # if we are saving this to the trajectories observables add it as a dataset
            if save_to_hdf5:

                if debug_prints:
                    print("Saving run {} traj {} observables/{}".format(
                        run_idx, traj_idx, field_name))

                # try to get the observables group or make it if it doesn't exist
                try:
                    obs_grp = self.traj(run_idx, traj_idx)[OBSERVABLES]
                except KeyError:

                    if debug_prints:
                        print("Group uninitialized. Initializing.")

                    obs_grp = self.traj(run_idx, traj_idx).create_group(OBSERVABLES)

                # try to create the dataset
                try:
                    obs_grp.create_dataset(field_name, data=obs_value)
                # if it fails we either overwrite or raise an error
                except RuntimeError:
                    # if we are in a permissive write mode we delete the
                    # old dataset and add the new one, overwriting old data
                    if self.mode in ['w', 'w-', 'x', 'r+']:

                        if debug_prints:
                            print("Dataset already present. Overwriting.")

                        del obs_grp[field_name]
                        obs_grp.create_dataset(field_name, data=obs_value)
                    # this will happen in 'c' and 'c-' modes
                    else:
                        raise RuntimeError(
                            "Dataset already exists and file is in concatenate mode ('c' or 'c-')")

            # also return it if requested
            if return_results:
                if idxs:
                    yield idx_tup, obs_value
                else:
                    yield obs_value
