import os.path as osp
from collections import Sequence, namedtuple
import itertools as it
import json
from warnings import warn
from copy import copy
import logging

import numpy as np
import h5py
import networkx as nx

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
SPORADIC_RECORDS = (RESAMPLER, WARPING, RESAMPLING, BC)

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
                 alt_reps=None, main_rep_idxs=None,
                 expert_mode=False
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

        """

        self._filename = filename

        if expert_mode is True:
            self._h5 = None
            self._wepy_mode = None
            self._h5py_mode = None
            self.closed = None

            # terminate the constructor here
            return None

        assert mode in ['r', 'r+', 'w', 'w-', 'x', 'a', 'c', 'c-'], \
          "mode must be either 'r', 'r+', 'w', 'x', 'w-', 'a', 'c', or 'c-'"



        # the top level mode enforced by wepy.hdf5
        self._wepy_mode = mode

        # the lower level h5py mode. THis was originally different to
        # accomodate different modes at teh wepy level for
        # concatenation. I will leave these separate because this is
        # used elsewhere and could be a feature in the future.
        self._h5py_mode = mode

        # Temporary metadata: used to initialize the object but not
        # used after that

        self._topology = topology
        self._units = units
        self._n_dims = n_dims
        self._n_coords = None

        # set hidden feature shapes and dtype, which are only
        # referenced if needed when trajectories are created. These
        # will be saved in the settings section in the actual HDF5
        # file
        self._field_feature_shapes_kwarg = feature_shapes
        self._field_feature_dtypes_kwarg = feature_dtypes
        self._field_feature_dtypes = None
        self._field_feature_shapes = None

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


        # open the file and then run the different constructors based
        # on the mode
        with h5py.File(filename, mode=self._h5py_mode) as h5:
            self._h5 = h5

            # create file mode: 'w' will create a new file or overwrite,
            # 'w-' and 'x' will not overwrite but will create a new file
            if self._wepy_mode in ['w', 'w-', 'x']:
                self._create_init()

            # read/write mode: in this mode we do not completely overwrite
            # the old file and start again but rather write over top of
            # values if requested
            elif self._wepy_mode in ['r+']:
                self._read_write_init()

            # add mode: read/write create if doesn't exist
            elif self._wepy_mode in ['a']:
                if osp.exists(self._filename):
                    self._read_write_init()
                else:
                    self._create_init()

            # read only mode
            elif self._wepy_mode == 'r':

                # if any data was given, warn the user
                if any([kwarg is not None for kwarg in
                        [topology, units, sparse_fields,
                         feature_shapes, feature_dtypes,
                         n_dims, alt_reps, main_rep_idxs]]):
                   warn("Data was given but opening in read-only mode", RuntimeWarning)

                # then run the initialization process
                self._read_init()

            # flush the buffers
            self._h5.flush()

            # set the h5py mode to the value in the actual h5py.File
            # object after creation
            self._h5py_mode = self._h5.mode

        # get rid of the temporary variables
        del self._topology
        del self._units
        del self._n_dims
        del self._n_coords
        del self._field_feature_shapes_kwarg
        del self._field_feature_dtypes_kwarg
        del self._field_feature_shapes
        del self._field_feature_dtypes
        del self._sparse_fields
        del self._main_rep_idxs
        del self._alt_reps

        # variable to reflect if it is closed or not, should be closed
        # after initialization
        self.closed = True

        # end of the constructor
        return None


    # context manager methods

    def __enter__(self):
        self._h5 = h5py.File(self._filename)
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._h5.flush()
        self.close()


    # custom deepcopy to avoid copying the actual HDF5 object


    # constructors
    def _create_init(self):
        """Completely overwrite the data in the file. Reinitialize the values
        and set with the new ones if given."""

        assert self._topology is not None, \
            "Topology must be given for a creation constructor"

        # initialize the runs group
        runs_grp = self._h5.create_group('runs')

        # initialize the settings group
        settings_grp = self._h5.create_group('_settings')

        # create the topology dataset
        self._h5.create_dataset('topology', data=self._topology)

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

        # initialize to the defaults, this gives values to
        # self._n_coords, and self.field_feature_dtypes, and
        # self.field_feature_shapes
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
        if self._units is None:
            self._units = {}
            for field_path in self._field_feature_shapes.keys():
                self._units[field_path] = None

        # set the units
        for field_path, unit_value in self._units.items():

            # ignore the field if not given
            if unit_value is None:
                continue

            unit_path = '/units/{}'.format(field_path)

            unit_grp.create_dataset(unit_path, data=unit_value)


        # create the group for the run data records
        records_grp = settings_grp.create_group('record_fields')

        # create a dataset for the continuation run tuples
        # (continuation_run, base_run), where the first element
        # of the new run that is continuing the run in the second
        # position
        self._init_continuations()

    def _read_write_init(self):
        """Write over values if given but do not reinitialize any old ones. """

        self._read_init()

    def _add_init(self):
        """Create the dataset if it doesn't exist and put it in r+ mode,
        otherwise, just open in r+ mode."""

        if not any(self._exist_flags):
            self._create_init()
        else:
            self._read_write_init()

    def _read_init(self):
        """Read only initialization currently has nothing to do."""

        pass

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

    def clone(self, path, mode='x'):
        """Clones this WepyHDF5 file without any of the actual runs and run
        data. This includes the topology, units, sparse_fields,
        feature shapes and dtypes, alt_reps, and main representation
        information.

        This method will flush the buffers for this file.

        Does not preserve metadata pertaining to inter-run
        relationships like continuations.

        """

        assert mode in ['w', 'w-', 'x'], "must be opened in a file creation mode"

        # we manually construct an HDF5 and copy the groups over
        new_h5 = h5py.File(path, mode=mode)

        new_h5.create_group('runs')

        # flush the datasets buffers
        self.h5.flush()
        new_h5.flush()

        # copy the existing datasets to the new one
        h5py.h5o.copy(self._h5.id, b'topology', new_h5.id, b'topology')
        h5py.h5o.copy(self._h5.id, b'units', new_h5.id, b'units')
        h5py.h5o.copy(self._h5.id, b'_settings', new_h5.id, b'_settings')

        # for the settings we need to get rid of the data for interun
        # relationships like the continuations, so we reinitialize the
        # continuations
        self._init_continuations()

        # now make a WepyHDF5 object in "expert_mode" which means it
        # is just empy and we construct it manually, "surgically" as I
        # like to call it
        new_wepy_h5 = WepyHDF5(path, expert_mode=True)

        # perform the surgery:

        # attach the h5py.File
        new_wepy_h5._h5 = new_h5
        # set the wepy mode to read-write since the creation flags
        # were already used in construction of the h5py.File object
        new_wepy_h5._wepy_mode = 'r+'
        new_wepy_h5._h5py_mode = 'r+'

        # close the h5py.File and set the attribute to closed
        new_wepy_h5._h5.close()
        new_wepy_h5.closed = True


        # return the runless WepyHDF5 object
        return new_wepy_h5

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
    def settings_grp(self):
        settings_grp = self.h5['_settings']
        return settings_grp

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
    def continuations(self):
        return self.settings_grp['continuations'][:]

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
        return list(range(len(self._h5['runs'])))

    def next_run_idx(self):
        return self.n_runs

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

    def contig_n_cycles(self, run_idxs):

        # check the contig to make sure it is a valid contig
        if not self.is_contig(run_idxs):
            raise ValueError("The run_idxs provided are not a valid contig, {}.".format(
                run_idxs))

        n_cycles = 0
        for run_idx in run_idxs:
            n_cycles += self.run_n_cycles(run_idx)

        return n_cycles

    def run_trajs(self, run_idx):
        return self._h5['runs/{}/trajectories'.format(run_idx)]

    def n_run_trajs(self, run_idx):
        return len(self._h5['runs/{}/trajectories'.format(run_idx)])

    def next_run_traj_idx(self, run_idx):
        return self.n_run_trajs(run_idx)

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

    def _init_continuations(self):
        """This will either create a dataset in the settings for the
        continuations or if continuations already exist it will reinitialize
        them and delete the data that exists there.

        """

        # if the continuations dset already exists we reinitialize the
        # data
        if 'continuations' in self.settings_grp:
            cont_dset = self.settings_grp['continuations']
            cont_dset.resize( (0,2) )

        # otherwise we just create the data
        else:
            cont_dset = self.settings_grp.create_dataset('continuations', shape=(0,2), dtype=np.int,
                                    maxshape=(None, 2))

        return cont_dset


    def init_record_fields(self, run_record_key, record_fields):
        """Save which records are to be considered from a run record group's
        datasets to be in the table like representation. This exists
        to allow there to large and small datasets for records to be
        stored together but allow for a more compact single table like
        representation to be produced for serialization.

        """

        record_fields_grp = self.settings_grp['record_fields']

        # make a dataset for the sparse fields allowed.  this requires
        # a 'special' datatype for variable length strings. This is
        # supported by HDF5 but not numpy.
        vlen_str_dt = h5py.special_dtype(vlen=str)

        # create the dataset with the strings of the fields which are records
        record_group_fields_ds = record_fields_grp.create_dataset(run_record_key,
                                                             (len(record_fields),),
                                                                  dtype=vlen_str_dt,
                                                                  maxshape=(None,))

        # set the flags
        for i, record_field in enumerate(record_fields):
            record_group_fields_ds[i] = record_field

    def init_resampling_record_fields(self, resampler):
        self.init_record_fields(RESAMPLING, resampler.resampling_record_field_names())

    def init_resampler_record_fields(self, resampler):
        self.init_record_fields(RESAMPLER, resampler.resampler_record_field_names())

    def init_bc_record_fields(self, bc):
        self.init_record_fields(BC, bc.bc_record_field_names())

    def init_warping_record_fields(self, bc):
        self.init_record_fields(WARPING, bc.warping_record_field_names())

    def init_progress_record_fields(self, bc):
        self.init_record_fields(PROGRESS, bc.progress_record_field_names())

    @property
    def record_fields(self):

        record_fields_grp = self.settings_grp['record_fields']

        record_fields_dict = {}
        for group_name, dset in record_fields_grp.items():
            record_fields_dict[group_name] = list(dset)

        return record_fields_dict

    def _add_run_init(self, run_idx, continue_run=None):
        """Routines for creating a run includes updating and setting object
        global variables, increasing the counter for the number of runs."""

        # add the run idx as metadata in the run group
        self._h5['runs/{}'.format(run_idx)].attrs['run_idx'] = run_idx

        # if this is continuing another run add the tuple (this_run,
        # continues_run) to the contig settings
        if continue_run is not None:

            self.add_continuation(run_idx, continue_run)


    def add_continuation(self, continuation_run, base_run):
        """Add a continuation between runs.

        continuation_run :: the run index of the run that continues base_run

        base_run :: the run that is being continued

        """

        continuations_dset = self.settings_grp['continuations']
        continuations_dset.resize((continuations_dset.shape[0] + 1, continuations_dset.shape[1],))
        continuations_dset[continuations_dset.shape[0] - 1] = np.array([continuation_run, base_run])

    def link_run(self, filepath, run_idx, continue_run=None, **kwargs):
        """Add a run from another file to this one as an HDF5 external
        link. Intuitively this is like mounting a drive in a filesystem."""

        # link to the external run
        ext_run_link = h5py.ExternalLink(filepath, 'runs/{}'.format(run_idx))

        # the run index in this file, as determined by the counter
        here_run_idx = self.next_run_idx()

        # set the local run as the external link to the other run
        self._h5['runs/{}'.format(here_run_idx)] = ext_run_link

        # run the initialization routines for adding a run
        self._add_run_init(here_run_idx, continue_run=continue_run)

        run_grp = self._h5['runs/{}'.format(here_run_idx)]

        # add metadata if given
        for key, val in kwargs.items():
            if key != 'run_idx':
                run_grp.attrs[key] = val
            else:
                warn('run_idx metadata is set by wepy and cannot be used', RuntimeWarning)

        return here_run_idx

    def link_file_runs(self, wepy_h5_path):
        """Link all runs from another WepyHDF5 file. This preserves
        continuations within that file. This will open the file if not
        already opened.

        returns the indices of the new runs in this file.

        """

        wepy_h5 = WepyHDF5(wepy_h5_path, mode='r')
        with wepy_h5:
            ext_run_idxs = wepy_h5.run_idxs
            continuations = wepy_h5.continuations

        # add the runs
        new_run_idxs = []
        for ext_run_idx in ext_run_idxs:

            # link the next run, and get its new run index
            new_run_idx = self.link_run(wepy_h5_path, ext_run_idx)

            # save that run idx
            new_run_idxs.append(new_run_idx)

        # copy the continuations over translating the run idxs,
        # for each continuation in the other files continuations
        for continuation in continuations:

            # translate each run index from the external file
            # continuations to the run idxs they were just assigned in
            # this file
            self.add_continuation(new_run_idxs[continuation[0]],
                                  new_run_idxs[continuation[1]])

        return new_run_idxs


    def new_run(self, continue_run=None, **kwargs):

        # check to see if the continue_run is actually in this file
        if continue_run is not None:
            if continue_run not in self.run_idxs:
                raise ValueError("The continue_run idx given, {}, is not present in this file".format(
                    continue_run))

        # get the index for this run
        new_run_idx = self.next_run_idx()

        # create a new group named the next integer in the counter
        run_grp = self._h5.create_group('runs/{}'.format(new_run_idx))

        # initialize the walkers group
        traj_grp = run_grp.create_group('trajectories')


        # run the initialization routines for adding a run
        self._add_run_init(new_run_idx, continue_run=continue_run)


        # TODO get rid of this?
        # add metadata if given
        for key, val in kwargs.items():
            if key != 'run_idx':
                run_grp.attrs[key] = val
            else:
                warn('run_idx metadata is set by wepy and cannot be used', RuntimeWarning)

        return run_grp

    # application level methods for setting the fields for run record
    # groups given the objects themselves
    def init_run_resampling(self, run_idx, resampler):

        # set the enumeration of the decisions
        self.init_run_resampling_decision(0, resampler)

        # set the data fields that can be used for table like records
        resampler.resampler_record_field_names()
        resampler.resampling_record_field_names()

        # then make the records group
        fields = resampler.resampling_fields()
        grp = self.init_run_record_grp(run_idx, RESAMPLING, fields)

        return grp

    def init_run_resampling_decision(self, run_idx, resampler):

        self.init_run_fields_resampling_decision(run_idx, resampler.DECISION.enum_dict_by_name())

    def init_run_resampler(self, run_idx, resampler):

        fields = resampler.resampler_fields()

        grp = self.init_run_record_grp(run_idx, RESAMPLER, fields)

        return grp

    def init_run_warping(self, run_idx, bc):

        fields = bc.warping_fields()
        grp = self.init_run_record_grp(run_idx, WARPING, fields)

        return grp

    def init_run_progress(self, run_idx, bc):

        fields = bc.progress_fields()

        grp = self.init_run_record_grp(run_idx, PROGRESS, fields)

        return grp

    def init_run_bc(self, run_idx, bc):

        fields = bc.bc_fields()

        grp = self.init_run_record_grp(run_idx, BC, fields)

        return grp

    # application level methods for initializing the run records
    # groups with just the fields and without the objects
    def init_run_fields_resampling(self, run_idx, fields):

        grp = self.init_run_record_grp(run_idx, RESAMPLING, fields)

        return grp

    def init_run_fields_resampling_decision(self, run_idx, decision_enum_dict):

        decision_grp = self.run(run_idx).create_group('decision')
        for name, value in decision_enum_dict.items():
            decision_grp.create_dataset(name, data=value)


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

        # its not just make it normally
        else:
            # create the group
            dset = record_grp.create_dataset(field_name, (0, *field_shape), dtype=field_dtype,
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
        traj_idx = self.next_run_traj_idx(run_idx)
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

    def decision_grp(self, run_idx):
        return self.run(run_idx)['decision']

    def decision_enum(self, run_idx):

        enum_grp = self.decision_grp(run_idx)
        enum = {}
        for decision_name, dset in enum_grp.items():
            enum[decision_name] = dset[()]

        return enum

    def decision_value_names(self, run_idx):
        enum_grp = self.decision_grp(run_idx)
        rev_enum = {}
        for decision_name, dset in enum_grp.items():
            value = dset[()]
            rev_enum[value] = decision_name

        return rev_enum

    ## application level append methods for run records groups

    def extend_cycle_warping_records(self, run_idx, cycle_idx, warping_data):
        self.extend_cycle_run_group_records(run_idx, WARPING, cycle_idx, warping_data)

    def extend_cycle_bc_records(self, run_idx, cycle_idx, bc_data):
        self.extend_cycle_run_group_records(run_idx, BC, cycle_idx, bc_data)

    def extend_cycle_progress_records(self, run_idx, cycle_idx, progress_data):
        self.extend_cycle_run_group_records(run_idx, PROGRESS, cycle_idx, progress_data)

    def extend_cycle_resampling_records(self, run_idx, cycle_idx, resampling_data):
        self.extend_cycle_run_group_records(run_idx, RESAMPLING, cycle_idx, resampling_data)

    def extend_cycle_resampler_records(self, run_idx, cycle_idx, resampler_data):
        self.extend_cycle_run_group_records(run_idx, RESAMPLER, cycle_idx, resampler_data)

    def extend_cycle_run_group_records(self, run_idx, run_record_key, cycle_idx, fields_data):
        """Append data for a whole records group, that is every field
        dataset. This must have the cycle index for the data it is
        appending as this is done for sporadic and continual datasets.

        """

        record_grp = self.records_grp(run_idx, run_record_key)

        # if it is sporadic add the cycle idx
        if self._is_sporadic_records(run_record_key):

            # get the cycle idxs dataset
            record_cycle_idxs_ds = record_grp[CYCLE_IDXS]

            # number of old and new records
            n_new_records = len(fields_data)
            n_existing_records = record_cycle_idxs_ds.shape[0]

            # make a new chunk for the new records
            record_cycle_idxs_ds.resize( (n_existing_records + n_new_records,) )

            # add an array of the cycle idx for each record
            record_cycle_idxs_ds[n_existing_records:] = np.full((n_new_records,), cycle_idx)

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

        # check whether it is a variable length record, by getting the
        # record dataset dtype and using the checker to see if it is
        # the vlen special type in h5py
        if h5py.check_dtype(vlen=field.dtype) is not None:

            # if it is we have to treat it differently, since it
            # cannot be multidimensional

            # if the dataset has no data in it we need to reshape it
            if all([i == 0 for i in field.shape]):
                # initialize this array
                # if it is empty resize it to make an array the size of
                # the new field_data with the maxshape for the feature
                # dimensions
                field.resize( (n_new_frames,) )

                # set the new data to this
                for i, row in enumerate(field_data):
                    field[i] = row

            # otherwise just add the data
            else:

                # resize the array but it is only of rank because
                # of variable length data
                field.resize( (field.shape[0] + n_new_frames, ) )

                # add each row to the newly made space
                for i, row in enumerate(field_data):
                    field[(field.shape[0] - 1) + i] = row

        # if it is not variable length we don't have to treat it
        # differently
        else:

            # if this is empty we need to reshape the dataset to accomodate data
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

            # otherwise just add the data
            else:
                # append to the dataset on the first dimension, keeping the
                # others the same, these must be feature vectors and therefore
                # must exist
                field.resize( (field.shape[0] + n_new_frames, *field.shape[1:]) )
                # add the new data
                field[-n_new_frames:, ...] = field_data

    def get_traj_field_cycle_idxs(self, run_idx, traj_idx, field_path):
        """ Returns the sparse indices for a field"""

        traj_path = "/runs/{}/trajectories/{}".format(run_idx, traj_idx)

        # if the field doesn't exist return None
        if not field_path in self._h5[traj_path]:
            raise KeyError("key for field {} not found".format(field_path))
            # return None

        # if the field is not sparse just return the cycle indices for
        # that run
        if field_path not in self.sparse_fields:
            cycle_idxs = np.array(range(self.run_n_cycles(run_idx)))
        else:
            cycle_idxs = self._h5[traj_path][field_path]['_sparse_idxs'][:]

        return cycle_idxs

    def get_traj_field(self, run_idx, traj_idx, field_path, frames=None, masked=True):
        """Returns a numpy array for the given field.

        You can control how sparse fields are returned using the
        `masked` option. When True (default) a masked numpy array will
        be returned such that you can get which cycles it is from,
        when False an unmasked array of the data will be returned
        which has no cycle information.

        """

        traj_path = "/runs/{}/trajectories/{}".format(run_idx, traj_idx)

        # if the field doesn't exist return None
        if not field_path in self._h5[traj_path]:
            raise KeyError("key for field {} not found".format(field_path))
            # return None

        # get the field depending on whether it is sparse or not
        if field_path in self.sparse_fields:
            return self._get_sparse_traj_field(run_idx, traj_idx, field_path,
                                               frames=frames, masked=masked)
        else:
            return self._get_contiguous_traj_field(run_idx, traj_idx, field_path,
                                                   frames=frames)

    def _get_contiguous_traj_field(self, run_idx, traj_idx, field_path, frames=None):

        full_path = "/runs/{}/trajectories/{}/{}".format(run_idx, traj_idx, field_path)

        if frames is None:
            field = self._h5[full_path][:]
        else:
            field = self._h5[full_path][list(frames)]

        return field

    def _get_sparse_traj_field(self, run_idx, traj_idx, field_path, frames=None, masked=True):

        traj_path = "/runs/{}/trajectories/{}".format(run_idx, traj_idx)
        traj_grp = self.h5[traj_path]
        field = traj_grp[field_path]

        n_frames = traj_grp[POSITIONS].shape[0]

        if frames is None:
            data = field['data'][:]

            # if it is to be masked make the masked array
            if masked:
                sparse_idxs = field['_sparse_idxs'][:]

                filled_data = np.full( (n_frames, *data.shape[1:]), np.nan)
                filled_data[sparse_idxs] = data

                mask = np.full( (n_frames, *data.shape[1:]), True)
                mask[sparse_idxs] = False

                data = np.ma.masked_array(filled_data, mask=mask)

        else:

            # get the sparse idxs and the frames to slice from the
            # data
            sparse_idxs = field['_sparse_idxs'][:]

            # we get a boolean array of the rows of the data table
            # that we are to slice from
            sparse_frame_idxs = np.argwhere(np.isin(sparse_idxs, frames))

            data = field['data'][list(sparse_frame_idxs)]

            # if it is to be masked make the masked array
            if masked:
                # the empty arrays the size of the number of requested frames
                filled_data = np.full( (len(frames), *field['data'].shape[1:]), np.nan)
                mask = np.full( (len(frames), *field['data'].shape[1:]), True )

                # take the data which exists and is part of the frames
                # selection, and put it into the filled data where it is
                # supposed to be
                filled_data[np.isin(frames, sparse_idxs)] = data

                # unmask the present values
                mask[np.isin(frames, sparse_idxs)] = False

                data = np.ma.masked_array(filled_data, mask=mask)

        return data


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
        for traj_idx, cycle_idx in frame_tups:
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

    def iter_trajs_fields(self, fields, idxs=False, traj_sel=None):
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

    def traj_fields_map(self, func, fields, *args, map_func=map, idxs=False, traj_sel=None):
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
        #first go through each run and get the number of cycles
        n_cycles = 0
        for run_idx in self.run_idxs:
            n_cycles += self.run_n_cycles(run_idx)

        mapped_args = []
        for arg in args:
            # if it is a sequence or generator we keep just pass it to the mapper
            if isinstance(arg, list) and not isinstance(arg, str):
                assert len(arg) == len(n_cycles), "Sequence has fewer"
                mapped_args.append(arg)
            # if it is not a sequence or generator we make a generator out
            # of it to map as inputs
            else:
                mapped_arg = (arg for i in range(n_cycles))
                mapped_args.append(mapped_arg)


        results = map_func(func, self.iter_trajs_fields(fields, traj_sel=traj_sel, idxs=False),
                           *mapped_args)

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
                           save_to_hdf5=None, idxs=False, return_results=True):
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
                                           map_func=map_func, traj_sel=traj_sel, idxs=True):

            idx_tup, obs_features = result
            run_idx, traj_idx = idx_tup

            # if we are saving this to the trajectories observables add it as a dataset
            if save_to_hdf5:

                logging.info("Saving run {} traj {} observables/{}".format(
                    run_idx, traj_idx, field_name))

                # try to get the observables group or make it if it doesn't exist
                try:
                    obs_grp = self.traj(run_idx, traj_idx)[OBSERVABLES]
                except KeyError:

                    logging.info("Group uninitialized. Initializing.")

                    obs_grp = self.traj(run_idx, traj_idx).create_group(OBSERVABLES)

                # try to create the dataset
                try:
                    obs_grp.create_dataset(field_name, data=obs_features)
                # if it fails we either overwrite or raise an error
                except RuntimeError:
                    # if we are in a permissive write mode we delete the
                    # old dataset and add the new one, overwriting old data
                    if self.mode in ['w', 'w-', 'x', 'r+']:

                        logging.info("Dataset already present. Overwriting.")

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

    @classmethod
    def _spanning_paths(cls, edges, root):

        # nodes targetting this root
        root_sources = []

        # go through all the edges and find those with this
        # node as their target
        for edge_source, edge_target in edges:

            # check if the target_node we are looking for matches
            # the edge target node
            if root == edge_target:

                # if this root is a target of the source add it to the
                # list of edges targetting this root
                root_sources.append(edge_source)

        # from the list of source nodes targetting this root we choose
        # the lowest index one, so we sort them and iterate through
        # finding the paths starting from it recursively
        root_paths = []
        root_sources.sort()
        for new_root in root_sources:

            # add these paths for this new root to the paths for the
            # current root
            root_paths.extend(cls._spanning_paths(edges, new_root))

        # if there are no more sources to this root it is a leaf node and
        # we terminate recursion, by not entering the loop above, however
        # we manually generate an empty list for a path so that we return
        # this "root" node as a leaf, for default.
        if len(root_paths) < 1:
            root_paths = [[]]

        final_root_paths = []
        for root_path in root_paths:
            final_root_paths.append([root] + root_path)

        return final_root_paths

    def spanning_contigs(self):
        """Returns a list of all possible spanning contigs given the
        continuations present in this file. Contigs are a list of runs
        in the order that makes a continuous set of data. Spanning
        contigs are always as long as possible, thus all must start
        from a root and end at a leaf node.

        This algorithm always returns them in a canonical order (as
        long as the runs are not rearranged after being added). This
        means that the indices here are the indices of the contigs.

        Contigs can in general are any such path drawn from what we
        call the "contig tree" which is the tree (or forest of trees)
        generated by the directed edges of the 'continuations'. They
        needn't be spanning from root to leaf.

        """

        # a list of all the unique nodes (runs)
        nodes = list(self.run_idxs)

        # first find the roots of the forest, start with all of them
        # and eliminate them if any directed edge points out of them
        # (in the first position of a continuation; read a
        # continuation (say (B,A)) as B continues A, thus the edge is
        # A <- B)
        roots = nodes
        for edge_source, edge_target in self.continuations:

            # if the edge source node is still in roots pop it out,
            # because a root can never be a source in an edge
            if edge_source in roots:
                _ = roots.pop(roots.index(edge_source))

        # collect all the spanning contigs
        spanning_contigs = []

        # roots should be sorted already, so we just iterate over them
        for root in roots:

            # get the spanning paths by passing the continuation edges
            # and this root to this recursive static method
            root_spanning_contigs = self._spanning_paths(self.continuations, root)

            spanning_contigs.extend(root_spanning_contigs)

        return spanning_contigs

    def contig_tree(self):
        """Returns a networkx directed graph where each node is a run index
        and the connections between them are the continuations.

        """

        contig_tree = nx.DiGraph()
        # first add all the runs as nodes
        contig_tree.add_nodes_from(self.run_idxs)

        # then add the continuations as edges
        contig_tree.add_edges_from(self.continuations)

        return contig_tree

    def is_contig(self, run_idxs):
        """This method checks that if a given list of run indices is a valid
        contig or not.
        """
        run_idx_continuations = [np.array([run_idxs[idx+1], run_idxs[idx]])
                            for idx in range(len(run_idxs)-1)]
        #gets the contigs array
        continuations = self.settings_grp['continuations'][:]

        # checks if sub contigs are in contigs list or not.
        for run_continuous in run_idx_continuations:
            contig = False
            for continuous in continuations:
                if np.array_equal(run_continuous, continuous):
                    contig = True
            if not contig:
                return False

        return True


    def contig_trace_to_trace(self, run_idxs, contig_trace):
        """This method takes a trace of frames over the given contig
        (run_idxs), itself a list of tuples (traj_idx, cycle_idx) and
        converts it to a trace valid for actually accessing values
        from a WepyHDF5 object i.e. a list of tuples
        (run_idx, traj_idx, cycle_idx).

        """

        # check the contig to make sure it is a valid contig
        if not self.is_contig(run_idxs):
            raise ValueError("The run_idxs provided are not a valid contig, {}.".format(
                run_idxs))

        # get the number of cycles in each of the runs in the contig
        runs_n_cycles = []
        for run_idx in run_idxs:
            runs_n_cycles.append(self.run_n_cycles(run_idx))

        # cumulative number of cycles
        run_cum_cycles = np.cumsum(runs_n_cycles)

        # add a zero to the beginning for the starting point of no
        # frames
        run_cum_cycles = np.hstack( ([0], run_cum_cycles) )

        # go through the frames of the contig trace and convert them
        new_trace = []
        for traj_idx, contig_cycle_idx in contig_trace:

            # get the index of the run that this cycle idx of the
            # contig is in
            frame_run_idx = np.searchsorted(run_cum_cycles, contig_cycle_idx, side='right') - 1

            # use that to get the total number of frames up until the
            # point of that and subtract that from the contig cycle
            # idx to get the cycle index in the run
            run_frame_idx = contig_cycle_idx - run_cum_cycles[frame_run_idx]

            # the tuple for this frame for the whole file
            trace_frame_idx = (frame_run_idx, traj_idx, run_frame_idx)

            # add it to the trace
            new_trace.append(trace_frame_idx)

        return new_trace


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

    def run_records(self, run_idx, run_record_key):

        # wrap this in a list since the underlying functions accept a
        # list of records
        run_idxs = [run_idx]

        return self.contig_records(run_idxs, run_record_key)

    def contig_records(self, run_idxs, run_record_key):

        # if there are no fields return an empty list
        record_fields = self.record_fields[run_record_key]
        if len(record_fields) == 0:
            return []

        # get the iterator for the record idxs, if the group is
        # sporadic then we just use the cycle idxs
        if self._is_sporadic_records(run_record_key):
            records = self._run_records_sporadic(run_idxs, run_record_key)
        else:
            records = self._run_records_continual(run_idxs, run_record_key)

        return records


    # TODO remove
    def cycle_tree(self):
        # make a network which has nodes (run_idx, cycle_idx) and the
        # main attibute is the resampling steps for that cycle
        cycle_tree = nx.DiGraph()

        # first go through each run without continuations
        for run_idx in self.run_idxs:
            n_cycles = self.run_n_cycles(run_idx)

            # make all the nodes for this run
            nodes = [(run_idx, step_idx) for step_idx in range(n_cycles)]
            cycle_tree.add_nodes_from(nodes)

            # the same for the edges
            edge_node_idxs = list(zip(range(1, n_cycles), range(n_cycles - 1)))

            edges = [(nodes[a], nodes[b]) for a, b in edge_node_idxs]
            cycle_tree.add_edges_from(edges)

        # after we have added all the nodes and edges for the run
        # subgraphs we need to connect them together with the
        # information in the contig tree.
        for edge_source, edge_target in self.continuations:

            # for the source node (the restart run) we use the run_idx
            # from the edge source node and the index of the first
            # cycle
            source_node = (edge_source, 0)

            # for the target node (the run being continued) we use the
            # run_idx from the edge_target and the last cycle index in
            # the run
            target_node = (edge_target, self.run_n_cycles(edge_target)-1)

            # make the edge
            edge = (source_node, target_node)

            # add this connector edge to the network
            cycle_tree.add_edge(*edge)

        return cycle_tree

    def contig_tree_records(self, run_record_keys):
        """Get records in the form of a tree for the whole contig tree. Each
        collection of records for a run will be in the node of the
        contig tree corresponding to the run.

        """

        contig_tree = self.contig_tree()

        # just loop through each run and get the records for it then
        # assign it to the node in the contig tree
        for run_idx in self.run_idxs:
            for run_record_key in run_record_keys:
                contig_tree.nodes[run_idx][run_record_key] = self.run_records(run_idx, run_record_key)

        return contig_tree


    def _run_record_namedtuple(self, run_record_key):

        Record = namedtuple('{}_Record'.format(run_record_key),
                            ['cycle_idx'] + self.record_fields[run_record_key])

        return Record

    def _convert_record_field_to_table_column(self, run_idx, run_record_key, record_field):

        # get the field dataset
        rec_grp = self.records_grp(run_idx, run_record_key)
        dset = rec_grp[record_field]

        # if it is variable length or if it has more than one element
        # cast all elements to tuples
        if h5py.check_dtype(vlen=dset.dtype) is not None:
            rec_dset = [tuple(value) for value in dset[:]]

        # if it is not variable length make sure it is not more than a
        # 1D feature vector
        elif len(dset.shape) > 2:
            raise TypeError(
                "cannot convert fields with feature vectors more than 1 dimension,"
                " was given {} for {}/{}".format(
                    dset.shape[1:], run_record_key, record_field))

        # if it is only a rank 1 feature vector and it has more than
        # one element make a tuple out of it
        elif dset.shape[1] > 1:
            rec_dset = [tuple(value) for value in dset[:]]

        # otherwise just get the single value instead of keeping it as
        # a single valued feature vector
        else:
            rec_dset = [value[0] for value in dset[:]]

        return rec_dset

    def _convert_record_fields_to_table_columns(self, run_idx, run_record_key):
        fields = {}
        for record_field in self.record_fields[run_record_key]:
            fields[record_field] = self._convert_record_field_to_table_column(
                                           run_idx, run_record_key, record_field)

        return fields

    def _make_records(self, run_record_key, cycle_idxs, fields):
        Record = self._run_record_namedtuple(run_record_key)

        # for each record we make a tuple and yield it
        records = []
        for record_idx in range(len(cycle_idxs)):

            # make a record for this cycle
            record_d = {'cycle_idx' : cycle_idxs[record_idx]}
            for record_field, column in fields.items():
                datum = column[record_idx]
                record_d[record_field] = datum

            record = Record(**record_d)

            records.append(record)

        return records

    def _run_records_sporadic(self, run_idxs, run_record_key):

        # we loop over the run_idxs in the contig and get the fields
        # and cycle idxs for the whole contig
        fields = None
        cycle_idxs = np.array([], dtype=int)
        # keep a cumulative total of the runs cycle idxs
        prev_run_cycle_total = 0
        for run_idx in run_idxs:

            # get all the value columns from the datasets, and convert
            # them to something amenable to a table
            run_fields = self._convert_record_fields_to_table_columns(run_idx, run_record_key)

            # we need to concatenate each field to the end of the
            # field in the master dictionary, first we need to
            # initialize it if it isn't already made
            if fields is None:
                # if it isn't initialized we just set it as this first
                # run fields dictionary
                fields = run_fields
            else:
                # if it is already initialized we need to go through
                # each field and concatenate
                for field_name, field_data in run_fields.items():
                    # just add it to the list of fields that will be concatenated later
                    fields[field_name].extend(field_data)

            # get the cycle idxs for this run
            rec_grp = self.records_grp(run_idx, run_record_key)
            run_cycle_idxs = rec_grp[CYCLE_IDXS][:]

            # add the total number of cycles that came before this run
            # to each of the cycle idxs to get the cycle_idxs in terms
            # of the full contig
            run_contig_cycle_idxs = run_cycle_idxs + prev_run_cycle_total

            # add these cycle indices to the records for the whole contig
            cycle_idxs = np.hstack( (cycle_idxs, run_contig_cycle_idxs) )

            # add the total number of cycle_idxs from this run to the
            # running total
            prev_run_cycle_total += self.run_n_cycles(run_idx)

        # then make the records from the fields
        records = self._make_records(run_record_key, cycle_idxs, fields)

        return records

    def _run_records_continual(self, run_idxs, run_record_key):

        cycle_idxs = np.array([], dtype=int)
        fields = None
        prev_run_cycle_total = 0
        for run_idx in run_idxs:
            # get all the value columns from the datasets, and convert
            # them to something amenable to a table
            run_fields = self._convert_record_fields_to_table_columns(run_idx, run_record_key)

            # we need to concatenate each field to the end of the
            # field in the master dictionary, first we need to
            # initialize it if it isn't already made
            if fields is None:
                # if it isn't initialized we just set it as this first
                # run fields dictionary
                fields = run_fields
            else:
                # if it is already initialized we need to go through
                # each field and concatenate
                for field_name, field_data in run_fields.items():
                    # just add it to the list of fields that will be concatenated later
                    fields[field_name].extend(field_data)

            # get one of the fields (if any to iterate over)
            record_fields = self.record_fields[run_record_key]
            main_record_field = record_fields[0]

            # make the cycle idxs from that
            run_rec_grp = self.records_grp(run_idx, run_record_key)
            run_cycle_idxs = list(range(run_rec_grp[main_record_field].shape[0]))

            # add the total number of cycles that came before this run
            # to each of the cycle idxs to get the cycle_idxs in terms
            # of the full contig
            run_contig_cycle_indices = run_cycle_idxs + prev_run_cycle_total

            # add these cycle indices to the records for the whole contig
            cycle_idxs = np.hstack( (cycle_idxs, run_contig_cycle_idxs) )

            # add the total number of cycle_idxs from this run to the
            # running total
            prev_run_cycle_total += self.run_n_cycles(run_idx)


        # then make the records from the fields
        records = self._make_records(run_record_key, cycle_idxs, fields)

        return records

    def run_records_dataframe(self, run_idx, run_record_key):
        records = self.run_records(run_idx, run_record_key)
        return pd.DataFrame(records)

    def contig_records_dataframe(self, run_idxs, run_record_key):
        records = self.contig_records(run_idxs, run_record_key)
        return pd.DataFrame(records)

    # application level specific methods for each main group

    # resampling
    def resampling_records(self, run_idxs):

        return self.contig_records(run_idxs, RESAMPLING)

    def resampling_records_dataframe(self, run_idxs):

        return pd.DataFrame(self.resampling_records(run_idxs))

    # resampler records
    def resampler_records(self, run_idxs):

        return self.contig_records(run_idxs, RESAMPLER)

    def resampler_records_dataframe(self, run_idxs):

        return pd.DataFrame(self.resampler_records(run_idxs))

    # warping
    def warping_records(self, run_idxs):

        return self.contig_records(run_idxs, WARPING)

    def warping_records_dataframe(self, run_idxs):

        return pd.DataFrame(self.warping_records(run_idxs))

    # boundary conditions
    def bc_records(self, run_idxs):

        return self.contig_records(run_idxs, BC)

    def bc_records_dataframe(self, run_idxs):

        return pd.DataFrame(self.bc_records(run_idxs))

    # progress
    def progress_records(self, run_idxs):

        return self.contig_records(run_idxs, PROGRESS)

    def progress_records_dataframe(self, run_idxs):

        return pd.DataFrame(self.progress_records(run_idxs))

    @staticmethod
    def resampling_panel(resampling_records, is_sorted=False):
        """Converts a simple collection of resampling records into a list of
        elements corresponding to cycles. It is like doing a pivot on
        the step indices into an extra dimension. Hence it can be
        thought of as a list of tables indexed by the cycle, hence the
        name panel.

        """

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
                    if rec.cycle_idx > cycle_idx:
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
                            #if cycle_rec[RESAMPLING_RECORD_FIELDS.index(STEP)] > step_idx:
                            if cycle_rec.step_idx > step_idx:
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
                                walker_idx = walker_rec.walker_idx
                                decision_id = walker_rec.decision_id
                                instruction = walker_rec.target_idxs

                                # set the resampling record for the walker in the step records
                                step_row[walker_idx] = (decision_id, instruction)

                            # add the records for this step to the cycle table
                            cycle_table.append(step_row)

            # add the table for this cycles records to the parent panel
            resampling_panel.append(cycle_table)

        return resampling_panel


    def run_resampling_panel(self, run_idx):
        return self.contig_resampling_panel([run_idx])


    def contig_resampling_panel(self, run_idxs):
        # check the contig to make sure it is a valid contig
        if not self.is_contig(run_idxs):
            raise ValueError("The run_idxs provided are not a valid contig, {}.".format(
                run_idxs))

        # make the resampling panel from the resampling records for the contig
        contig_resampling_panel = self.resampling_panel(self.resampling_records(run_idxs))

        return contig_resampling_panel


    # TODO remove
    def cycle_tree_resampling_panel(self):
        """Instead of a resampling panel this is the same thing except each
        cycle is a node rather than a table in the panel.

        """

        # get the cycle tree
        cycle_tree = self.cycle_tree()

        # then get the resampling tables for each cycle and put them
        # as attributes to the appropriate nodes
        for run_idx in self.run_idxs:

            run_resampling_panel = self.run_resampling_panel(run_idx)

            # add each cycle of this panel to the network by adding
            # them in as nodes with the resampling steps first
            for step_idx, step in enumerate(run_resampling_panel):
                node = (run_idx, step_idx)
                cycle_tree.nodes[node]["resampling_steps"] = step

        return cycle_tree


    def join(self, other_h5):
        """Given another WepyHDF5 file object does a left join on this
        file. Renumbering the runs starting from this file.
        """

        with other_h5 as h5:
            for run_idx in h5.run_idxs:
                # the other run group handle
                other_run = h5.run(run_idx)
                # copy this run to this file in the next run_idx group
                self.h5.copy(other_run, 'runs/{}'.format(self.next_run_idx()))

    def to_mdtraj(self, run_idx, traj_idx, frames=None, alt_rep=None):

        traj_grp = self.traj(run_idx, traj_idx)

        # the default for alt_rep is the main rep
        if alt_rep is None:
            rep_key = POSITIONS
            rep_path = rep_key
        else:
            rep_key = alt_rep
            rep_path = 'alt_reps/{}'.format(alt_rep)

        topology = self.get_mdtraj_topology(alt_rep=rep_key)

        frames = self.get_traj_field_cycle_idxs(run_idx, traj_idx, rep_path)

        # get the data for all or for the frames specified
        positions = self.get_traj_field(run_idx, traj_idx, rep_path,
                                        frames=frames, masked=False)
        try:
            time = self.get_traj_field(run_idx, traj_idx, TIME,
                                       frames=frames, masked=False)[:, 0]
        except KeyError:
            warn("time not in this trajectory, ignoring")
            time = None

        try:
            box_vectors = self.get_traj_field(run_idx, traj_idx, BOX_VECTORS,
                                              frames=frames, masked=False)
        except KeyError:
            warn("box_vectors not in this trajectory, ignoring")
            box_vectors = None


        if box_vectors is not None:
            unitcell_lengths, unitcell_angles = traj_box_vectors_to_lengths_angles(box_vectors)

        if (box_vectors is not None) and (time is not None):
            traj = mdj.Trajectory(positions, topology,
                           time=time,
                           unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)
        elif box_vectors is not None:
            traj = mdj.Trajectory(positions, topology,
                           unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)
        elif time is not None:
            traj = mdj.Trajectory(positions, topology,
                           time=time)
        else:
            traj = mdj.Trajectory(positions, topology)

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


    def iter_cycles_fields(self, fields, idxs=False, traj_sel=None):

        for cycle_idx in self.cycles:
            dsets = {}
            for field, dset in self.iter_cycle_fields(fields, cycle_idx, traj_sel=traj_sel):
                dsets[field] = dset
            if idxs:
                yield (cycle_idx,  dsets)
            else:
                yield dsets


    def traj_cycles_map(self, func, fields, *args, map_func=map, idxs=False, traj_sel=None):
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



        results = map_func(func, self.iter_cycles_fields(fields, traj_sel=traj_sel, idxs=False),
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
                           save_to_hdf5=None, idxs=False, return_results=True):
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
                                       map_func=map_func, traj_sel=traj_sel, idxs=True):
            idx_tup, obs_value = result
            run_idx, traj_idx = idx_tup

            # if we are saving this to the trajectories observables add it as a dataset
            if save_to_hdf5:

                logging.info("Saving run {} traj {} observables/{}".format(
                    run_idx, traj_idx, field_name))

                # try to get the observables group or make it if it doesn't exist
                try:
                    obs_grp = self.traj(run_idx, traj_idx)[OBSERVABLES]
                except KeyError:

                    logging.info("Group uninitialized. Initializing.")

                    obs_grp = self.traj(run_idx, traj_idx).create_group(OBSERVABLES)

                # try to create the dataset
                try:
                    obs_grp.create_dataset(field_name, data=obs_value)
                # if it fails we either overwrite or raise an error
                except RuntimeError:
                    # if we are in a permissive write mode we delete the
                    # old dataset and add the new one, overwriting old data
                    if self.mode in ['w', 'w-', 'x', 'r+']:

                        logging.info("Dataset already present. Overwriting.")

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
