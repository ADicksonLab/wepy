# -*- coding: utf-8 -*-

"""Primary wepy simulation database driver and access API using the
HDF5 format.

The HDF5 Format Specification
=============================

As part of the wepy framework this module provides a fully-featured
API for creating and accessing data generated in weighted ensemble
simulations run with wepy.

The need for a special purpose format is many-fold but primarily it is
the nonlinear branching structure of walker trajectories coupled with
weights.

That is for standard simulations data is organized as independent
linear trajectories of frames each related linearly to the one before
it and after it.

In weighted ensemble due to the resampling (i.e. cloning and merging)
of walkers, a single frame may have multiple 'child' frames.

This is the primary motivation for this format.

However, in practice it solves several other issues and itself is a
more general and flexible format than for just weighted ensemble
simulations.

Concretely the WepyHDF5 format is simply an informally described
schema that is commensurable with the HDF5 constructs of hierarchical
groups (similar to unix filesystem directories) arranged as a tree
with datasets as the leaves.

The hierarchy is fairly deep and so we will progress downwards from
the top and describe each broad section in turn breaking it down when
necessary.

Header
------

The items right under the root of the tree are:

- runs
- topology
- _settings

The first item 'runs' is itself a group that contains all of the
primary data from simulations. In WepyHDF5 the run is the unit
dataset. All data internal to a run is self contained. That is for
multiple dependent trajectories (e.g. from cloning and merging) all
exist within a single run.

This excludes metadata-like things that may be needed for interpreting
this data, such as the molecular topology that imposes structure over
a frame of atom positions. This information is placed in the
'topology' item.

The topology field has no specified internal structure at this
time. However, with the current implementation of the WepyHDF5Reporter
(which is the principal implementation of generating a WepyHDF5
object/file from simulations) this is simply a string dataset. This
string dataset should be a JSON compliant string. The format of which
is specified elsewhere and was borrowed from the mdtraj library.

Warning! this format and specification for the topology is subject to
change in the future and will likely be kept unspecified indefinitely.

For most intents and purposes (which we assume to be for molecular or
molecular-like simulations) the 'topology' item (and perhaps any other
item at the top level other than those proceeded by and underscore,
such as in the '_settings' item) is merely useful metadata that
applies to ALL runs and is not dynamical.

In the language of the orchestration module all data in 'runs' uses
the same 'apparatus' which is the function that takes in the initial
conditions for walkers and produces new walkers. The apparatus may
differ in the specific values of parameters but not in kind. This is
to facilitate runs that are continuations of other runs. For these
kinds of simulations the state of the resampler, boundary conditions,
etc. will not be as they were initially but are the same in kind or
type.

All of the necessary type information of data in runs is kept in the
'_settings' group. This is used to serialize information about the
data types, shapes, run to run continuations etc. This allows for the
initialization of an empty (no runs) WepyHDF5 database at one time and
filling of data at another time. Otherwise types of datasets would
have to be inferred from the data itself, which may not exist yet.

As a convention items which are preceeded by an underscore (following
the python convention) are to be considered hidden and mechanical to
the proper functioning of various WepyHDF5 API features, such as
sparse trajectory fields.

The '_settings' is specified as a simple key-value structure, however
values may be arbitrarily complex.

Runs
----

The meat of the format is contained within the runs group:

- runs

  - 0
  - 1
  - 2
  - ...

Under the runs group are a series of groups for each run. Runs are
named according to the order in which they were added to the database.

Within a run (say '0' from above) we have a number of items:

- 0

  - init_walkers
  - trajectories
  - decision
  - resampling
  - resampler
  - warping
  - progress
  - boundary_conditions

Trajectories
^^^^^^^^^^^^

The 'trajectories' group is where the data for the frames of the
walker trajectories is stored.

Even though the tree-like trajectories of weighted ensemble data may
be well suited to having a tree-like storage topology we have opted to
use something more familiar to the field, and have used a collection
of linear "trajectories".

This way of breaking up the trajectory data coupled with proper
records of resampling (see below) allows for the imposition of a tree
structure without committing to that as the data storage topology.

This allows the WepyHDF5 format to be easily used as a container
format for collections of linear trajectories. While this is not
supported in any real capacity it is one small step to convergence. We
feel that a format that contains multiple trajectories is important
for situations like weighted ensemble where trajectories are
interdependent. The transition to a storage format like HDF5 however
opens up many possibilities for new features for trajectories that
have not occurred despite several attempts to forge new formats based
on HDF5 (TODO: get references right; see work in mdtraj and MDHDF5).

Perhaps these formats have not caught on because the existing formats
(e.g. XTC, DCD) for simple linear trajectories are good enough and
there is little motivation to migrate.

However, by making the WepyHDF5 format (and related sub-formats to be
described e.g. record groups and the trajectory format) both cover a
new use case which can't be achieved with old formats and old ones
with ease.

Once users see the power of using a format like HDF5 from using wepy
they may continue to use it for simpler simulations.


In any case the 'trajectories' in the group for weighted ensemble
simulations should be thought of only as containers and not literally
as trajectories. That is frame 4 does not necessarily follow from
frame 3. So one may think of them more as "lanes" or "slots" for
trajectory data that needs to be stitched together with the
appropriate resampling records.

The routines and methods for generating contiguous trajectories from
the data in WepyHDF5 are given through the 'analysis' module, which
generates "traces" through the dataset.

With this in mind we will describe the sub-format of a trajectory now.

The 'trajectories' group is similar to the 'runs' group in that it has
sub-groups whose names are numbers. These numbers however are not the
order in which they are created but an index of that trajectory which
are typically laid out all at once.

For a wepy simulation with a constant number of walkers you will only
ever need as many trajectories/slots as there are walkers. So if you
have 8 walkers then you will have trajectories 0 through 7. Concretely:

- runs

  - 0

    - trajectories

      - 0
      - 1
      - 2
      - 3
      - 4
      - 5
      - 6
      - 7

If we look at trajectory 0 we might see the following groups within:

- positions
- box_vectors
- velocities
- weights

Which is what you would expect for a constant pressure molecular
dynamics simulation where you have the positions of the atoms, the box
size, and velocities of the atoms.

The particulars for what "fields" a trajectory in general has are not
important but this important use-case is directly supported in the
WepyHDF5 format.

In any such simulation, however, the 'weights' field will appear since
this is the weight of the walker of this frame and is a value
important to weighted ensemble and not the underlying dynamics.

The naive approach to these fields is that each is a dataset of
dimension (n_frames, feature_vector_shape[0], ...) where the first dimension
is the cycle_idx and the rest of the dimensions are determined by the
atomic feature vector for each field for a single frame.

For example, the positions for a molecular simulation with 100 atoms
with x, y, and z coordinates that ran for 1000 cycles would be a
dataset of the shape (1000, 100, 3). Similarly the box vectors would
be (1000, 3, 3) and the weights would be (1000, 1).

This uniformity vastly simplifies accessing and adding new variables
and requires that individual state values in walkers always be arrays
with shapes, even when they are single values (e.g. energy). The
exception being the weight which is handled separately.

However, this situation is actually more complex to allow for special
features.

First of all is the presence of compound fields which allow nesting of
multiple groups.

The above "trajectory fields" would have identifiers such as the
literal strings 'positions' and 'box_vectors', while a compound field
would have an identifier 'observables/rmsd' or 'alt_reps/binding_site'.

Use of trajectory field names using the '/' path separator will
automatically make a field a group and the last element of the field
name the dataset. So for the observables example we might have:

- 0

  - observables

    - rmsd
    - sasa

Where the rmsd would be accessed as a trajectory field of trajectory 0
as 'observables/rmsd' and the solvent accessible surface area as
'observables/sasa'.

This example introduces how the WepyHDF5 format is not only useful for
storing data produced by simulation but also in the analysis of that
data and computation of by-frame quantities.

The 'observables' compound group key prefix is special and will be
used in the 'compute_observables' method.

The other special compound group key prefix is 'alt_reps' which is
used for particle simulations to store "alternate representation" of
the positions. This is useful in cooperation with the next feature of
wepy trajectory fields to allow for more economical storage of data.

The next feature (and complication of the format) is the allowance for
sparse fields. As the fields were introduced we said that they should
have as many feature vectors as there are frames for the
simulation. In the example however, you will notice that storing both
the full atomic positions and velocities for a long simulation
requires a heavy storage burden.

So perhaps you only want to store the velocities (or forces) every 100
frames so that you can be able to restart a simulation form midway
through the simulation. This is achieved through sparse fields.

A sparse field is no longer a dataset but a group with two items:

- _sparse_idxs
- data

The '_sparse_idxs' are simply a dataset of integers that assign each
element of the 'data' dataset to a frame index. Using the above
example we run a simulation for 1000 frames with 100 atoms and we save
the velocities every 100 frames we would have a 'velocities/data'
dataset of shape (100, 100, 3) which is 10 times less data than if it
were saved every frame.

While this complicates the storage format use of the proper API
methods should be transparent whether you are returning a sparse field
or not.

As alluded to above the use of sparse fields can be used for more than
just accessory fields. In many simulations, such as those with full
atomistic simulations of proteins in solvent we often don't care about
the dynamics of most of the atoms in the simulation and so would like
to not have to save them.

The 'alt_reps' compound field is meant to solve this. For example, the
WepyHDF5Reporter supports a special option to save only a subset of
the atoms in the main 'positions' field but also to save the full
atomic system as an alternate representation, which is the field name
'alt_reps/all_atoms'. So that you can still save the full system every
once in a while but be economical in what positions you save every
single frame.

Note that there really isn't a way to achieve this with other
formats. You either make a completely new trajectory with only the
atoms of interest and now you are duplicating those in two places, or
you duplicate and then filter your full systems trajectory file and
rely on some sort of index to always live with it in the filesystem,
which is a very precarious scenario. The situation is particularly
hopeless for weighted ensemble trajectories.

Init Walkers
^^^^^^^^^^^^

The data stored in the 'trajectories' section is the data that is
returned after running dynamics in a cycle. Since we view the WepyHDF5
as a completely self-contained format for simulations it seems
negligent to rely on outside sources (such as the filesystem) for the
initial structures that seeded the simulations. These states (and
weights) can be stored in this group.

The format of this group is identical to the one for trajectories
except that there is only one frame for each slot and so the shape of
the datasets for each field is just the shape of the feature vector.

Record Groups
^^^^^^^^^^^^^

TODO: add reference to reference groups

The last five items are what are called 'record groups' and all follow
the same format.

Each record group contains itself a number of datasets, where the
names of the datasets correspond to the 'field names' from the record
group specification. So each record groups is simply a key-value store
where the values must be datasets.

For instance the fields in the 'resampling' (which is particularly
important as it encodes the branching structure) record group for a
WExplore resampler simulation are:

- step_idx
- walker_idx
- decision_id
- target_idxs
- region_assignment

Where the 'step_idx' is an integer specifying which step of resampling
within the cycle the resampling action took place (the cycle index is
metadata for the group). The 'walker_idx' is the index of the walker
that this action was assigned to. The 'decision_id' is an integer that
is related to an enumeration of decision types that encodes which
discrete action is to be taken for this resampling event (the
enumeration is in the 'decision' item of the run groups). The
'target_idxs' is a variable length 1-D array of integers which assigns
the results of the action to specific target 'slots' (which was
discussed for the 'trajectories' run group). And the
'region_assignment' is specific to WExplore which reports on which
region the walker was in at that time, and is a variable length 1-D
array of integers.

Additionally, record groups are broken into two types:

- continual
- sporadic

Continual records occur once per cycle and so there is no extra
indexing necessary.

Sporadic records can happen multiple or zero times per cycle and so
require a special index for them which is contained in the extra
dataset '_cycle_idxs'.

It is worth noting that the underlying methods for each record group
are general. So while these are the official wepy record groups that
are supported if there is a use-case that demands a new record group
it is a fairly straightforward task from a developers perspective.

"""

import os.path as osp
from collections import Sequence, namedtuple, defaultdict, Counter
import itertools as it
import json
from warnings import warn
from copy import copy
import logging
import gc

import numpy as np
import h5py
import networkx as nx

from wepy.analysis.parents import resampling_panel
from wepy.util.mdtraj import mdtraj_to_json_topology, json_to_mdtraj_topology, \
                             traj_fields_to_mdtraj
from wepy.util.util import traj_box_vectors_to_lengths_angles
from wepy.util.json_top import json_top_subset, json_top_atom_count

# optional dependencies
try:
    import mdtraj as mdj
except ModuleNotFoundError:
    warn("mdtraj is not installed and that functionality will not work", RuntimeWarning)

try:
    import pandas as pd
except ModuleNotFoundError:
    warn("pandas is not installed and that functionality will not work", RuntimeWarning)

## h5py settings

# we set the libver to always be the latest (which should be 1.10) so
# that we know we can always use SWMR and the newest features. We
# don't care about backwards compatibility with HDF5 1.8. Just update
# in a new virtualenv if this is a problem for you
H5PY_LIBVER = 'latest'

## Header and settings keywords

TOPOLOGY = 'topology'
"""Default header apparatus dataset. The molecular topology dataset."""

SETTINGS = '_settings'
"""Name of the settings group in the header group."""

RUNS = 'runs'
"""The group name for runs."""


## metadata fields
RUN_IDX = 'run_idx'
"""Metadata field for run groups for the run index within this file."""

RUN_START_SNAPSHOT_HASH = 'start_snapshot_hash'
"""Metadata field for a run that corresponds to the hash of the
starting simulation snapshot in orchestration."""

RUN_END_SNAPSHOT_HASH = 'end_snapshot_hash'
"""Metadata field for a run that corresponds to the hash of the
ending simulation snapshot in orchestration."""

TRAJ_IDX = 'traj_idx'
"""Metadata field for trajectory groups for the trajectory index in that run."""

## Misc. Names

CYCLE_IDX = 'cycle_idx'
"""String for setting the names of cycle indices in records and
miscellaneous situations."""


## Settings field names
SPARSE_FIELDS = 'sparse_fields'
"""Settings field name for sparse field trajectory field flags."""

N_ATOMS = 'n_atoms'
"""Settings field name group for the number of atoms in the default positions field."""

N_DIMS_STR = 'n_dims'
"""Settings field name for positions field spatial dimensions."""

MAIN_REP_IDXS = 'main_rep_idxs'
"""Settings field name for the indices of the full apparatus topology in
the default positions trajectory field."""

ALT_REPS_IDXS = 'alt_reps_idxs'
"""Settings field name for the different 'alt_reps'. The indices of
the atoms from the full apparatus topology for each."""

FIELD_FEATURE_SHAPES_STR = 'field_feature_shapes'
"""Settings field name for the trajectory field shapes."""

FIELD_FEATURE_DTYPES_STR = 'field_feature_dtypes'
"""Settings field name for the trajectory field data types."""

UNITS = 'units'
"""Settings field name for the units of the trajectory fields."""

RECORD_FIELDS = 'record_fields'
"""Settings field name for the record fields that are to be included
in the truncated listing of record group fields."""

CONTINUATIONS = 'continuations'
"""Settings field name for the continuations relationships between runs."""


## Run Fields Names
TRAJECTORIES = 'trajectories'
"""Run field name for the trajectories group."""

INIT_WALKERS = 'init_walkers'
"""Run field name for the initial walkers group."""

DECISION = 'decision'
"""Run field name for the decision enumeration group."""

## Record Groups Names
RESAMPLING = 'resampling'
"""Record group run field name for the resampling records """

RESAMPLER = 'resampler'
"""Record group run field name for the resampler records """

WARPING = 'warping'
"""Record group run field name for the warping records """

PROGRESS = 'progress'
"""Record group run field name for the progress records """

BC = 'boundary_conditions'
"""Record group run field name for the boundary conditions records """

## Record groups constants

# special datatypes strings
NONE_STR = 'None'
"""String signifying a field of unspecified shape. Used for
serializing the None python object."""

CYCLE_IDXS = '_cycle_idxs'
"""Group name for the cycle indices of sporadic records."""

# records can be sporadic or continual. Continual records are
# generated every cycle and are saved every cycle and are for all
# walkers.  Sporadic records are generated conditional on specific
# events taking place and thus may or may not be produced each
# cycle. There also is not a single record for each (cycle, step) like
# there would be for continual ones because they can occur for single
# walkers, boundary conditions, or resamplers.
SPORADIC_RECORDS = (RESAMPLER, WARPING, RESAMPLING, BC)
"""Enumeration of the record groups that are sporadic."""

## Trajectories Group

# Default Trajectory Constants

N_DIMS = 3
"""Number of dimensions for the default positions."""


# Required Trajectory Fields

WEIGHTS = 'weights'
"""The field name for the frame weights."""

# default fields for trajectories

POSITIONS = 'positions'
"""The field name for the default positions."""

BOX_VECTORS = 'box_vectors'
"""The field name for the default box vectors."""

VELOCITIES = 'velocities'
"""The field name for the default velocities."""

FORCES = 'forces'
"""The field name for the default forces."""

TIME = 'time'
"""The field name for the default time."""

KINETIC_ENERGY = 'kinetic_energy'
"""The field name for the default kinetic energy."""

POTENTIAL_ENERGY = 'potential_energy'
"""The field name for the default potential energy."""

BOX_VOLUME = 'box_volume'
"""The field name for the default box volume."""

PARAMETERS = 'parameters'
"""The field name for the default parameters."""

PARAMETER_DERIVATIVES = 'parameter_derivatives'
"""The field name for the default parameter derivatives."""

ALT_REPS = 'alt_reps'
"""The field name for the default compound field observables."""

OBSERVABLES = 'observables'
"""The field name for the default compound field observables."""

## Trajectory Field Constants

WEIGHT_SHAPE = (1,)
"""Weights feature vector shape."""

WEIGHT_DTYPE = np.float
"""Weights feature vector data type."""

# Default Trajectory Field Constants
FIELD_FEATURE_SHAPES = ((TIME, (1,)),
                        (BOX_VECTORS, (3,3)),
                        (BOX_VOLUME, (1,)),
                        (KINETIC_ENERGY, (1,)),
                        (POTENTIAL_ENERGY, (1,)),
                        )
"""Default shapes for the default fields."""

FIELD_FEATURE_DTYPES = ((POSITIONS, np.float),
                        (VELOCITIES, np.float),
                        (FORCES, np.float),
                        (TIME, np.float),
                        (BOX_VECTORS, np.float),
                        (BOX_VOLUME, np.float),
                        (KINETIC_ENERGY, np.float),
                        (POTENTIAL_ENERGY, np.float),
                        )
"""Default data types for the default fields."""

# Positions (and thus velocities and forces) are determined by the
# N_DIMS (which can be customized) and more importantly the number of
# particles which is always different. All the others are always wacky
# and different.
POSITIONS_LIKE_FIELDS = (VELOCITIES, FORCES)
"""Default trajectory fields which are the same shape as the main positions field."""

## Trajectory field features keys

# sparse trajectory fields
DATA = 'data'
"""Name of the dataset in sparse trajectory fields."""

SPARSE_IDXS = '_sparse_idxs'
"""Name of the dataset that indexes sparse trajectory fields."""

# utility for paths
def _iter_field_paths(grp):
    """Return all subgroup field name paths from a group.

    Useful for compound fields. For example if you have the group
    observables with multiple subfields:

    - observables
      - rmsd
      - sasa

    Passing the h5py group 'observables' will return the full field
    names for each subfield:

    - 'observables/rmsd'
    - 'observables/sasa'

    Parameters
    ----------
    grp : h5py.Group
        The group to enumerate subfield names for.

    Returns
    -------
    subfield_names : list of str
        The full names for the subfields of the group.

    """
    field_paths = []
    for field_name in grp:
        if isinstance(grp[field_name], h5py.Group):
            for subfield in grp[field_name]:

                # if it is a sparse field don't do the subfields since
                # they will be _sparse_idxs and data which are not
                # what we want here
                if field_name not in grp.file['_settings/sparse_fields']:
                    field_paths.append(field_name + '/' + subfield)
        else:
            field_paths.append(field_name)
    return field_paths

class WepyHDF5(object):
    """Wrapper for h5py interface to an HDF5 file object for creation and
    access of WepyHDF5 data.

    This is the primary implementation of the API for creating,
    accessing, and modifying data in an HDF5 file that conforms to the
    WepyHDF5 specification.

    """

    MODES = ('r', 'r+', 'w', 'w-', 'x', 'a')
    """The recognized modes for opening the WepyHDF5 file."""

    WRITE_MODES = ('r+', 'w', 'w-', 'x', 'a')


    #### dunder methods

    def __init__(self, filename, mode='x',
                 topology=None,
                 units=None,
                 sparse_fields=None,
                 feature_shapes=None, feature_dtypes=None,
                 n_dims=None,
                 alt_reps=None, main_rep_idxs=None,
                 swmr_mode=False,
                 expert_mode=False
    ):
        """Constructor for the WepyHDF5 class.

        Initialize a new Wepy HDF5 file. This will create an h5py.File
        object.

        The File will be closed after construction by default.

        mode:
        r        Readonly, file must exist
        r+       Read/write, file must exist
        w        Create file, truncate if exists
        x or w-  Create file, fail if exists
        a        Read/write if exists, create otherwise

        Parameters
        ----------
        filename : str
            File path

        mode : str
            Mode specification for opening the HDF5 file.

        topology : str
            JSON string representing topology of system being simulated.

        units : dict of str : str, optional
            Mapping of trajectory field names to string specs
            for units.

        sparse_fields : list of str, optional
            List of trajectory fields that should be initialized as sparse.

        feature_shapes : dict of str : shape_spec, optional
            Mapping of trajectory fields to their shape spec for initialization.

        feature_dtypes : dict of str : dtype_spec, optional
            Mapping of trajectory fields to their shape spec for initialization.

        n_dims : int, default: 3
            Set the number of spatial dimensions for the default
            positions trajectory field.

        alt_reps : dict of str : list of int, optional
            Specifies that there will be 'alt_reps' of positions each
            named by the keys of this mapping and containing the
            indices in each value list.

        main_rep_idxs : list of int, optional
            The indices of atom positions to save as the main 'positions'
            trajectory field. Defaults to all atoms.

        expert_mode : bool
            If True no initialization is performed other than the
            setting of the filename. Useful mainly for debugging.

        Raises
        ------

        AssertionError
            If the mode is not one of the supported mode specs.

        AssertionError
            If a topology is not given for a creation mode.

        Warns
        -----

        If initialization data was given but the file was opened in a read mode.

        """

        self._filename = filename
        self._swmr_mode = swmr_mode

        if expert_mode is True:
            self._h5 = None
            self._wepy_mode = None
            self._h5py_mode = None
            self.closed = None

            # terminate the constructor here
            return None

        assert mode in self.MODES, \
          "mode must be either one of: {}".format(', '.join(self.MODES))

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
            alt_rep_keys = ['{}/{}'.format(ALT_REPS, key) for key in self._alt_reps.keys()]
            self._sparse_fields.extend(alt_rep_keys)
        else:
            self._alt_reps = {}


        # open the file and then run the different constructors based
        # on the mode
        with h5py.File(filename, mode=self._h5py_mode,
                       libver=H5PY_LIBVER, swmr=self._swmr_mode) as h5:
            self._h5 = h5

            # set SWMR mode if asked for if we are in write mode also
            if self._swmr_mode is True and mode in self.WRITE_MODES:
                self._h5.swmr_mode = swmr_mode

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

    # TODO is this right? shouldn't we actually delete the data then close
    def __del__(self):
        self.close()

    # context manager methods

    def __enter__(self):
        self.open()
        # self._h5 = h5py.File(self._filename,
        #                      libver=H5PY_LIBVER, swmr=self._swmr_mode)
        # self.closed = False
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    @property
    def swmr_mode(self):
        return self._swmr_mode

    @swmr_mode.setter
    def swmr_mode(self, val):
        self._swmr_mode = val


    # TODO custom deepcopy to avoid copying the actual HDF5 object

    #### hidden methods (_method_name)

    ### constructors
    def _create_init(self):
        """Creation mode constructor.

        Completely overwrite the data in the file. Reinitialize the values
        and set with the new ones if given.
        """

        assert self._topology is not None, \
            "Topology must be given for a creation constructor"

        # initialize the runs group
        runs_grp = self._h5.create_group(RUNS)

        # initialize the settings group
        settings_grp = self._h5.create_group(SETTINGS)

        # create the topology dataset
        self._h5.create_dataset(TOPOLOGY, data=self._topology)

        # sparse fields
        if self._sparse_fields is not None:

            # make a dataset for the sparse fields allowed.  this requires
            # a 'special' datatype for variable length strings. This is
            # supported by HDF5 but not numpy.
            vlen_str_dt = h5py.special_dtype(vlen=str)

            # create the dataset with empty values for the length of the
            # sparse fields given
            sparse_fields_ds = settings_grp.create_dataset(SPARSE_FIELDS,
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
        settings_grp.create_dataset(N_DIMS_STR, data=np.array(self._n_dims))
        settings_grp.create_dataset(N_ATOMS, data=np.array(self._n_coords))

        # the main rep atom idxs
        settings_grp.create_dataset(MAIN_REP_IDXS, data=self._main_rep_idxs, dtype=np.int)

        # alt_reps settings
        alt_reps_idxs_grp = settings_grp.create_group(ALT_REPS_IDXS)
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
        shapes_grp = settings_grp.create_group(FIELD_FEATURE_SHAPES_STR)
        for field_path, field_shape in self._field_feature_shapes.items():
            if field_shape is None:
                # set it as a dimensionless array of NaN
                field_shape = np.array(np.nan)

            shapes_grp.create_dataset(field_path, data=field_shape)

        dtypes_grp = settings_grp.create_group(FIELD_FEATURE_DTYPES_STR)
        for field_path, field_dtype in self._field_feature_dtypes.items():
            if field_dtype is None:
                dt_str = NONE_STR
            else:
                # make a json string of the datatype that can be read
                # in again, we call np.dtype again because there is no
                # np.float.descr attribute
                dt_str = json.dumps(np.dtype(field_dtype).descr)

            dtypes_grp.create_dataset(field_path, data=dt_str)

        # initialize the units group
        unit_grp = self._h5.create_group(UNITS)

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

            unit_path = '{}/{}'.format(UNITS, field_path)

            unit_grp.create_dataset(unit_path, data=unit_value)


        # create the group for the run data records
        records_grp = settings_grp.create_group(RECORD_FIELDS)

        # create a dataset for the continuation run tuples
        # (continuation_run, base_run), where the first element
        # of the new run that is continuing the run in the second
        # position
        self._init_continuations()

    def _read_write_init(self):
        """Read-write mode constructor."""

        self._read_init()

    def _add_init(self):
        """The addition mode constructor.

        Create the dataset if it doesn't exist and put it in r+ mode,
        otherwise, just open in r+ mode.

        """

        if not any(self._exist_flags):
            self._create_init()
        else:
            self._read_write_init()

    def _read_init(self):
        """Read mode constructor."""

        pass

    def _set_default_init_field_attributes(self, n_dims=None):
        """Sets the feature_shapes and feature_dtypes to be the default for
        this module. These will be used to initialize field datasets when no
        given during construction (i.e. for sparse values)

        Parameters
        ----------
        n_dims : int

        """

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

    def _get_field_path_grp(self, run_idx, traj_idx, field_path):
        """Given a field path for the trajectory returns the group the field's
        dataset goes in and the key for the field name in that group.

        The field path for a simple field is just the name of the
        field and for a compound field it is the compound field group
        name with the subfield separated by a '/' like
        'observables/observable1' where 'observables' is the compound
        field group and 'observable1' is the subfield name.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_path : str

        Returns
        -------
        group : h5py.Group

        field_name : str

        """

        # check if it is compound
        if '/' in field_path:
            # split it
            grp_name, field_name = field_path.split('/')
            # get the hdf5 group
            grp = self.h5['{}/{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx, grp_name)]
        # its simple so just return the root group and the original path
        else:
            grp = self.h5
            field_name = field_path

        return grp, field_name

    def _init_continuations(self):
        """This will either create a dataset in the settings for the
        continuations or if continuations already exist it will reinitialize
        them and delete the data that exists there.

        Returns
        -------
        continuation_dset : h5py.Dataset

        """

        # if the continuations dset already exists we reinitialize the
        # data
        if CONTINUATIONS in self.settings_grp:
            cont_dset = self.settings_grp[CONTINUATIONS]
            cont_dset.resize( (0,2) )

        # otherwise we just create the data
        else:
            cont_dset = self.settings_grp.create_dataset(CONTINUATIONS, shape=(0,2), dtype=np.int,
                                    maxshape=(None, 2))

        return cont_dset


    def _add_run_init(self, run_idx, continue_run=None):
        """Routines for creating a run includes updating and setting object
        global variables, increasing the counter for the number of runs.

        Parameters
        ----------
        run_idx : int

        continue_run : int
            Index of the run to continue.

        """


        # add the run idx as metadata in the run group
        self._h5['{}/{}'.format(RUNS, run_idx)].attrs[RUN_IDX] = run_idx

        # if this is continuing another run add the tuple (this_run,
        # continues_run) to the continutations settings
        if continue_run is not None:

            self.add_continuation(run_idx, continue_run)

    def _add_init_walkers(self, init_walkers_grp, init_walkers):
        """Adds the run field group for the initial walkers.

        Parameters
        ----------
        init_walkers_grp : h5py.Group
            The group to add the walker data to.
        init_walkers : list of objects implementing the Walker interface
            The walkers to save in the group

        """

        # add the initial walkers to the group by essentially making
        # new trajectories here that will only have one frame
        for walker_idx, walker in enumerate(init_walkers):
            walker_grp = init_walkers_grp.create_group(str(walker_idx))

            # weights

            # get the weight from the walker and make a feature array of it
            weights = np.array([[walker.weight]])

            # then create the dataset and set it
            walker_grp.create_dataset(WEIGHTS, data=weights)

            # state fields data
            for field_key, field_value in walker.state.dict().items():

                # values may be None, just ignore them
                if field_value is not None:
                    # just create the dataset by making it a feature array
                    # (wrapping it in another list)
                    walker_grp.create_dataset(field_key, data=np.array([field_value]))


    def _init_run_sporadic_record_grp(self, run_idx, run_record_key, fields):
        """Initialize a sporadic record group for a run.

        Parameters
        ----------
        run_idx : int

        run_record_key : str
            The record group name.
        fields : list of field specs
            Each field spec is a 3-tuple of
            (field_name : str, field_shape : shape_spec, field_dtype : dtype_spec)

        Returns
        -------
        record_group : h5py.Group

        """

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
        """Initialize a continual record group for a run.

        Parameters
        ----------
        run_idx : int

        run_record_key : str
            The record group name.
        fields : list of field specs
            Each field spec is a 3-tuple of
            (field_name : str, field_shape : shape_spec, field_dtype : dtype_spec)

        Returns
        -------
        record_group : h5py.Group

        """

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
        """Initialize a single field for a run record group.

        Parameters
        ----------
        run_idx : int

        run_record_key : str
            The name of the record group.
        field_name : str
            The name of the field in the record group.
        field_shape : tuple of int
            The shape of the dataset for the field.
        field_dtype : dtype_spec
            An h5py recognized data type.

        Returns
        -------
        dataset : h5py.Dataset

        """

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
        """Tests whether a record group is sporadic or not.

        Parameters
        ----------
        run_record_key : str
            Record group name.

        Returns
        -------
        is_sporadic : bool
            True if the record group is sporadic False if not.

        """

        # assume it is continual and check if it is in the sporadic groups
        if run_record_key in SPORADIC_RECORDS:
            return True
        else:
            return False

    def _init_traj_field(self, run_idx, traj_idx, field_path, feature_shape, dtype):
        """Initialize a trajectory field.

        Initialize a data field in the trajectory to be empty but
        resizeable.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_path : str
            Field name specification.
        feature_shape : shape_spec
            Specification of shape of a feature vector of the field.
        dtype : dtype_spec
            Specification of the feature vector datatype.

        """

        # check whether this is a sparse field and create it
        # appropriately
        if field_path in self.sparse_fields:
            # it is a sparse field
            self._init_sparse_traj_field(run_idx, traj_idx, field_path, feature_shape, dtype)
        else:
            # it is not a sparse field (AKA simple)
            self._init_contiguous_traj_field(run_idx, traj_idx, field_path, feature_shape, dtype)

    def _init_contiguous_traj_field(self, run_idx, traj_idx, field_path, shape, dtype):
        """Initialize a contiguous (non-sparse) trajectory field.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_path : str
            Field name specification.
        feature_shape : tuple of int
            Shape of the feature vector of the field.
        dtype : dtype_spec
            H5py recognized datatype

        """

        traj_grp = self._h5['{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx)]

        # create the empty dataset in the correct group, setting
        # maxshape so it can be resized for new feature vectors to be added
        traj_grp.create_dataset(field_path, (0, *[0 for i in shape]), dtype=dtype,
                           maxshape=(None, *shape))


    def _init_sparse_traj_field(self, run_idx, traj_idx, field_path, shape, dtype):
        """

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_path : str
            Field name specification.
        feature_shape : shape_spec
            Specification for the shape of the feature.
        dtype : dtype_spec
            Specification for the dtype of the feature.

        """

        traj_grp = self._h5['{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx)]

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
            sparse_grp.create_dataset(DATA, (0, *[0 for i in shape]), dtype=dtype,
                               maxshape=(None, *shape))

            # create the dataset for the sparse indices
            sparse_grp.create_dataset(SPARSE_IDXS, (0,), dtype=np.int, maxshape=(None,))


    def _init_traj_fields(self, run_idx, traj_idx,
                          field_paths, field_feature_shapes, field_feature_dtypes):
        """Initialize a number of fields for a trajectory.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_paths : list of str
            List of field names.
        field_feature_shapes : list of shape_specs
        field_feature_dtypes : list of dtype_specs

        """
        for i, field_path in enumerate(field_paths):
            self._init_traj_field(run_idx, traj_idx,
                                  field_path, field_feature_shapes[i], field_feature_dtypes[i])

    def _add_traj_field_data(self,
                             run_idx,
                             traj_idx,
                             field_path,
                             field_data,
                             sparse_idxs=None,
    ):
        """Add a trajectory field to a trajectory.

        If the sparse indices are given the field will be created as a
        sparse field otherwise a normal one.

        Parameters
        ----------
        run_idx : int

        traj_idx : int

        field_path : str
            Field name.

        field_data : numpy.array
            The data array to set for the field.

        sparse_idxs : arraylike of int of shape (1,)
            List of cycle indices that the data corresponds to.

        """

        # get the traj group
        traj_grp = self._h5['{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx)]

        # if it is a sparse dataset we need to add the data and add
        # the idxs in a group
        if sparse_idxs is None:

            # first require that the dataset exist and is exactly the
            # same as the one that already exists (if indeed it
            # does). If it doesn't raise a specific error letting the
            # user know that they will have to delete the dataset if
            # they want to change it to something else
            try:
                dset = traj_grp.require_dataset(field_path, shape=field_data.shape, dtype=field_data.dtype,
                                         exact=True,
                                         maxshape=(None, *field_data.shape[1:]))
            except TypeError:
                raise TypeError("For changing the contents of a trajectory field it must be the same shape and dtype.")

            # if that succeeds then go ahead and set the data to the
            # dataset (overwriting if it is still there)
            dset[...] = field_data

        else:
            sparse_grp = traj_grp.create_group(field_path)
            # add the data to this group
            sparse_grp.create_dataset(DATA, data=field_data,
                                      maxshape=(None, *field_data.shape[1:]))
            # add the sparse idxs
            sparse_grp.create_dataset(SPARSE_IDXS, data=sparse_idxs,
                                      maxshape=(None,))

    def _extend_contiguous_traj_field(self, run_idx, traj_idx, field_path, field_data):
        """Add multiple new frames worth of data to the end of an existing
        contiguous (non-sparse)trajectory field.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_path : str
            Field name
        field_data : numpy.array
            The frames of data to add.

        """

        traj_grp = self.h5['{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx)]
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
        """Add multiple new frames worth of data to the end of an existing
        contiguous (non-sparse)trajectory field.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_path : str
            Field name
        values : numpy.array
            The frames of data to add.
        sparse_idxs : list of int
            The cycle indices the values correspond to.

        """

        field = self.h5['{}/{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx, field_path)]

        field_data = field[DATA]
        field_sparse_idxs = field[SPARSE_IDXS]

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

    def _add_sparse_field_flag(self, field_path):
        """Register a trajectory field as sparse in the header settings.

        Parameters
        ----------
        field_path : str
            Name of the trajectory field you want to flag as sparse

        """

        sparse_fields_ds = self._h5['{}/{}'.format(SETTINGS, SPARSE_FIELDS)]

        # make sure it isn't already in the sparse_fields
        if field_path in sparse_fields_ds[:]:
            warn("sparse field {} already a sparse field, ignoring".format(field_path))

        sparse_fields_ds.resize( (sparse_fields_ds.shape[0] + 1,) )
        sparse_fields_ds[sparse_fields_ds.shape[0] - 1] = field_path

    def _add_field_feature_shape(self, field_path, field_feature_shape):
        """Add the shape to the header settings for a trajectory field.

        Parameters
        ----------
        field_path : str
            The name of the trajectory field you want to set for.
        field_feature_shape : shape_spec
            The shape spec to serialize as a dataset.

        """
        shapes_grp = self._h5['{}/{}'.format(SETTINGS, FIELD_FEATURE_SHAPES_STR)]
        shapes_grp.create_dataset(field_path, data=np.array(field_feature_shape))

    def _add_field_feature_dtype(self, field_path, field_feature_dtype):
        """Add the data type to the header settings for a trajectory field.

        Parameters
        ----------
        field_path : str
            The name of the trajectory field you want to set for.
        field_feature_dtype : dtype_spec
            The dtype spec to serialize as a dataset.

        """
        feature_dtype_str = json.dumps(field_feature_dtype.descr)
        dtypes_grp = self._h5['{}/{}'.format(SETTINGS, FIELD_FEATURE_DTYPES_STR)]
        dtypes_grp.create_dataset(field_path, data=feature_dtype_str)


    def _set_field_feature_shape(self, field_path, field_feature_shape):
        """Add the trajectory field shape to header settings or set the value.

        Parameters
        ----------
        field_path : str
            The name of the trajectory field you want to set for.
        field_feature_shape : shape_spec
            The shape spec to serialize as a dataset.

        """
        # check if the field_feature_shape is already set
        if field_path in self.field_feature_shapes:
            # check that the shape was previously saved as "None" as we
            # won't overwrite anything else
            if self.field_feature_shapes[field_path] is None:
                full_path = '{}/{}/{}'.format(SETTINGS, FIELD_FEATURE_SHAPES_STR, field_path)
                # we have to delete the old data and set new data
                del self.h5[full_path]
                self.h5.create_dataset(full_path, data=field_feature_shape)
            else:
                raise AttributeError(
                    "Cannot overwrite feature shape for {} with {} because it is {} not {}".format(
                        field_path, field_feature_shape, self.field_feature_shapes[field_path],
                        NONE_STR))
        # it was not previously set so we must create then save it
        else:
            self._add_field_feature_shape(field_path, field_feature_shape)

    def _set_field_feature_dtype(self, field_path, field_feature_dtype):
        """Add the trajectory field dtype to header settings or set the value.

        Parameters
        ----------
        field_path : str
            The name of the trajectory field you want to set for.
        field_feature_dtype : dtype_spec
            The dtype spec to serialize as a dataset.

        """
        feature_dtype_str = json.dumps(field_feature_dtype.descr)
        # check if the field_feature_dtype is already set
        if field_path in self.field_feature_dtypes:
            # check that the dtype was previously saved as "None" as we
            # won't overwrite anything else
            if self.field_feature_dtypes[field_path] is None:
                full_path = '{}/{}/{}'.format(SETTINGS, FIELD_FEATURE_DTYPES_STR, field_path)
                # we have to delete the old data and set new data
                del self.h5[full_path]
                self.h5.create_dataset(full_path, data=feature_dtype_str)
            else:
                raise AttributeError(
                    "Cannot overwrite feature dtype for {} with {} because it is {} not ".format(
                        field_path, field_feature_dtype, self.field_feature_dtypes[field_path],
                        NONE_STR))
        # it was not previously set so we must create then save it
        else:
            self._add_field_feature_dtype(field_path, field_feature_dtype)

    def _extend_run_record_data_field(self, run_idx, run_record_key,
                                          field_name, field_data):
        """Primitive record append method.

        Adds data for a single field dataset in a run records group. This
        is done without paying attention to whether it is sporadic or
        continual and is supposed to be only the data write method.

        Parameters
        ----------
        run_idx : int
        run_record_key : str
            Name of the record group.
        field_name : str
            Name of the field in the record group to add to.
        field_data : arraylike
            The data to add to the field.

        """

        records_grp = self.h5['{}/{}/{}'.format(RUNS, run_idx, run_record_key)]
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


    def _run_record_namedtuple(self, run_record_key):
        """Generate a namedtuple record type for a record group.

        The class name will be formatted like '{}_Record' where the {}
        will be replaced with the name of the record group.

        Parameters
        ----------
        run_record_key : str
            Name of the record group

        Returns
        -------
        RecordType : namedtuple
            The record type to generate records for this record group.

        """

        Record = namedtuple('{}_Record'.format(run_record_key),
                            [CYCLE_IDX] + self.record_fields[run_record_key])

        return Record

    def _convert_record_field_to_table_column(self, run_idx, run_record_key, record_field):
        """Converts a dataset of feature vectors to more palatable values for
        use in external datasets.

        For single value feature vectors it unwraps them into single
        values.

        For 1-D feature vectors it casts them as tuples.

        Anything of higher rank will raise an error.

        Parameters
        ----------
        run_idx : int
        run_record_key : str
            Name of the record group
        record_field : str
            Name of the field of the record group

        Returns
        -------

        record_dset : list
            Table-ified values

        Raises
        ------

        TypeError
            If the field feature vector shape rank is greater than 1.

        """

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
        """Convert record group data to truncated namedtuple records.

        This uses the specified record fields from the header settings
        to choose which record group fields to apply this to.

        Does no checking to make sure the fields are
        "table-ifiable". If a field is not it will raise a TypeError.

        Parameters
        ----------
        run_idx : int
        run_record_key : str
            The name of the record group

        Returns
        -------
        table_fields : dict of str : list
            Mapping of the record group field to the table-ified values.

        """

        fields = {}
        for record_field in self.record_fields[run_record_key]:
            fields[record_field] = self._convert_record_field_to_table_column(
                                           run_idx, run_record_key, record_field)

        return fields

    def _make_records(self, run_record_key, cycle_idxs, fields):
        """Generate a list of proper (nametuple) records for a record group.

        Parameters
        ----------
        run_record_key : str
            Name of the record group
        cycle_idxs : list of int
            The cycle indices you want to get records for.
        fields : list of str
            The fields to make record entries for.

        Returns
        -------
        records : list of namedtuple objects

        """
        Record = self._run_record_namedtuple(run_record_key)

        # for each record we make a tuple and yield it
        records = []
        for record_idx in range(len(cycle_idxs)):

            # make a record for this cycle
            record_d = {CYCLE_IDX : cycle_idxs[record_idx]}
            for record_field, column in fields.items():
                datum = column[record_idx]
                record_d[record_field] = datum

            record = Record(*(record_d[key] for key in Record._fields))

            records.append(record)

        return records

    def _run_records_sporadic(self, run_idxs, run_record_key):
        """Generate records for a sporadic record group for a multi-run
        contig.

        If multiple run indices are given assumes that these are a
        contig (e.g. the second run index is a continuation of the
        first and so on). This method is considered low-level and does
        no checking to make sure this is true.

        The cycle indices of records from "continuation" runs will be
        modified so as the records will be indexed as if they are a
        single run.

        Uses the record fields settings to decide which fields to use.

        Parameters
        ----------
        run_idxs : list of int
            The indices of the runs in the order they are in the contig
        run_record_key : str
            Name of the record group

        Returns
        -------
        records : list of namedtuple objects

        """

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
            prev_run_cycle_total += self.num_run_cycles(run_idx)

        # then make the records from the fields
        records = self._make_records(run_record_key, cycle_idxs, fields)

        return records

    def _run_records_continual(self, run_idxs, run_record_key):
        """Generate records for a continual record group for a multi-run
        contig.

        If multiple run indices are given assumes that these are a
        contig (e.g. the second run index is a continuation of the
        first and so on). This method is considered low-level and does
        no checking to make sure this is true.

        The cycle indices of records from "continuation" runs will be
        modified so as the records will be indexed as if they are a
        single run.

        Uses the record fields settings to decide which fields to use.

        Parameters
        ----------
        run_idxs : list of int
            The indices of the runs in the order they are in the contig
        run_record_key : str
            Name of the record group

        Returns
        -------
        records : list of namedtuple objects

        """

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
            run_cycle_idxs = np.array(range(run_rec_grp[main_record_field].shape[0]))

            # add the total number of cycles that came before this run
            # to each of the cycle idxs to get the cycle_idxs in terms
            # of the full contig
            run_contig_cycle_idxs = run_cycle_idxs + prev_run_cycle_total

            # add these cycle indices to the records for the whole contig
            cycle_idxs = np.hstack( (cycle_idxs, run_contig_cycle_idxs) )

            # add the total number of cycle_idxs from this run to the
            # running total
            prev_run_cycle_total += self.num_run_cycles(run_idx)


        # then make the records from the fields
        records = self._make_records(run_record_key, cycle_idxs, fields)

        return records


    def _get_contiguous_traj_field(self, run_idx, traj_idx, field_path, frames=None):
        """Access actual data for a trajectory field.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_path : str
            Trajectory field name to access
        frames : list of int, optional
            The indices of the frames to return if you don't want all of them.

        Returns
        -------
        field_data : arraylike
            The data requested for the field.

        """

        full_path = '{}/{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx, field_path)

        if frames is None:
            field = self._h5[full_path][:]
        else:
            field = self._h5[full_path][list(frames)]

        return field

    def _get_sparse_traj_field(self, run_idx, traj_idx, field_path, frames=None, masked=True):
        """Access actual data for a trajectory field.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_path : str
            Trajectory field name to access

        frames : list of int, optional
            The indices of the frames to return if you don't want all of them.

        masked : bool
            If True returns the array data as numpy masked array, and
            only the available values if False.

        Returns
        -------
        field_data : arraylike
            The data requested for the field.

        """

        traj_path = '{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx)
        traj_grp = self.h5[traj_path]
        field = traj_grp[field_path]

        n_frames = traj_grp[POSITIONS].shape[0]

        if frames is None:
            data = field[DATA][:]

            # if it is to be masked make the masked array
            if masked:
                sparse_idxs = field[SPARSE_IDXS][:]

                filled_data = np.full( (n_frames, *data.shape[1:]), np.nan)
                filled_data[sparse_idxs] = data

                mask = np.full( (n_frames, *data.shape[1:]), True)
                mask[sparse_idxs] = False

                data = np.ma.masked_array(filled_data, mask=mask)

        else:

            # get the sparse idxs and the frames to slice from the
            # data
            sparse_idxs = field[SPARSE_IDXS][:]

            # we get a boolean array of the rows of the data table
            # that we are to slice from
            sparse_frame_idxs = np.argwhere(np.isin(sparse_idxs, frames))

            data = field[DATA][list(sparse_frame_idxs)]

            # if it is to be masked make the masked array
            if masked:
                # the empty arrays the size of the number of requested frames
                filled_data = np.full( (len(frames), *field[DATA].shape[1:]), np.nan)
                mask = np.full( (len(frames), *field[DATA].shape[1:]), True )

                # take the data which exists and is part of the frames
                # selection, and put it into the filled data where it is
                # supposed to be
                filled_data[np.isin(frames, sparse_idxs)] = data

                # unmask the present values
                mask[np.isin(frames, sparse_idxs)] = False

                data = np.ma.masked_array(filled_data, mask=mask)

        return data


    def _add_run_field(self,
                       run_idx,
                       field_path,
                       data,
                       sparse_idxs=None,
                       force=False):
        """Add a trajectory field to all trajectories in a run.

        By enforcing adding it to all trajectories at one time we
        promote in-run consistency.

        Parameters
        ----------
        run_idx : int
        field_path : str
            Name to set the trajectory field as. Can be compound.
        data : arraylike of shape (n_trajectories, n_cycles, feature_vector_shape[0],...)
            The data for all trajectories to be added.
        sparse_idxs : list of int
            If the data you are adding is sparse specify which cycles to apply them to.


        If 'force' is turned on, no checking for constraints will be done.

        """

        # TODO, SNIPPET: check that we have the right permissions
        # if field_exists:
        #     # if we are in a permissive write mode we delete the
        #     # old dataset and add the new one, overwriting old data
        #     if self.mode in ['w', 'w-', 'x', 'r+']:
        #         logging.info("Dataset already present. Overwriting.")
        #         del obs_grp[field_name]
        #         obs_grp.create_dataset(field_name, data=results)
        #     # this will happen in 'c' and 'c-' modes
        #     else:
        #         raise RuntimeError(
        #             "Dataset already exists and file is in concatenate mode ('c' or 'c-')")

        # check that the data has the correct number of trajectories
        if not force:
            assert len(data) == self.num_run_trajs(run_idx),\
                "The number of trajectories in data, {}, is different than the number"\
                "of trajectories in the run, {}.".format(len(data), self.num_run_trajs(run_idx))

            # for each trajectory check that the data is compliant
            for traj_idx, traj_data in enumerate(data):

                if not force:
                    # check that the number of frames is not larger than that for the run
                    if traj_data.shape[0] > self.num_run_cycles(run_idx):
                        raise ValueError("The number of frames in data for traj {} , {},"
                                          "is larger than the number of frames"
                                          "for this run, {}.".format(
                                                  traj_idx, data.shape[1], self.num_run_cycles(run_idx)))


                # if the number of frames given is the same or less than
                # the number of frames in the run
                elif (traj_data.shape[0] <= self.num_run_cycles(run_idx)):

                    # if sparse idxs were given we check to see there is
                    # the right number of them
                    #  and that they match the number of frames given
                    if data.shape[0] != len(sparse_idxs[traj_idx]):

                        raise ValueError("The number of frames provided for traj {}, {},"
                                          "was less than the total number of frames, {},"
                                          "but an incorrect number of sparse idxs were supplied, {}."\
                                         .format(traj_idx, traj_data.shape[0],
                                            self.num_run_cycles(run_idx), len(sparse_idxs[traj_idx])))


                    # if there were strictly fewer frames given and the
                    # sparse idxs were not given we need to raise an error
                    elif (traj_data.shape[0] < self.num_run_cycles(run_idx)):

                        raise ValueError("The number of frames provided for traj {}, {},"
                                          "was less than the total number of frames, {},"
                                          "but sparse_idxs were not supplied.".format(
                                                  traj_idx, traj_data.shape[0],
                                                  self.num_run_cycles(run_idx)))

        # add it to each traj
        for i, idx_tup in enumerate(self.run_traj_idx_tuples([run_idx])):
            if sparse_idxs is None:
                self._add_traj_field_data(*idx_tup, field_path, data[i])
            else:
                self._add_traj_field_data(*idx_tup, field_path, data[i],
                                          sparse_idxs=sparse_idxs[i])

    def _add_field(self, field_path, data, sparse_idxs=None,
                   force=False):
        """Add a trajectory field to all runs in a file.

        Parameters
        ----------
        field_path : str
            Name of trajectory field
        data : list of arraylike
            Each element of this list corresponds to a single run. The
            elements of which are arraylikes of shape (n_trajectories,
            n_cycles, feature_vector_shape[0],...) for each run.
        sparse_idxs : list of list of int
            The list of cycle indices to set for the sparse fields. If
            None, no trajectories are set as sparse.


        """

        for i, run_idx in enumerate(self.run_idxs):
            if sparse_idxs is not None:
                self._add_run_field(run_idx, field_path, data[i], sparse_idxs=sparse_idxs[i],
                                    force=force)
            else:
                self._add_run_field(run_idx, field_path, data[i],
                                    force=force)

    #### Public Methods

    ### File Utilities

    @property
    def filename(self):
        """The path to the underlying HDF5 file."""
        return self._filename

    def open(self, mode=None):
        """Open the underlying HDF5 file for access.

        Parameters
        ----------

        mode : str
           Valid mode spec. Opens the HDF5 file in this mode if given
           otherwise uses the existing mode.

        """

        if mode is None:
            mode = self.mode

        if self.closed:

            self.set_mode(mode)

            self._h5 = h5py.File(self._filename, mode,
                                 libver=H5PY_LIBVER, swmr=self.swmr_mode)
            self.closed = False
        else:
            raise IOError("This file is already open")

    def close(self):
        """Close the underlying HDF5 file. """
        if not self.closed:
            self._h5.flush()
            self._h5.close()
            self.closed = True

    @property
    def mode(self):
        """The WepyHDF5 mode this object was created with."""
        return self._wepy_mode

    @mode.setter
    def mode(self, mode):
        """Set the mode for opening the file with."""
        self.set_mode(mode)

    def set_mode(self, mode):
        """Set the mode for opening the file with."""

        if not self.closed:
            raise AttributeError("Cannot set the mode while the file is open.")

        self._set_h5_mode(mode)

        self._wepy_mode = mode

    @property
    def h5_mode(self):
        """The h5py.File mode the HDF5 file currently has."""
        return self._h5.mode

    def _set_h5_mode(self, h5_mode):
        """Set the mode to open the HDF5 file with.

        This really shouldn't be set without using the main wepy mode
        as they need to be aligned.

        """

        if not self.closed:
            raise AttributeError("Cannot set the mode while the file is open.")

        self._h5py_mode = h5_mode

    @property
    def h5(self):
        """The underlying h5py.File object."""
        return self._h5

    ### h5py object access

    def run(self, run_idx):
        """Get the h5py.Group for a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        run_group : h5py.Group

        """
        return self._h5['{}/{}'.format(RUNS, int(run_idx))]

    def traj(self, run_idx, traj_idx):
        """Get an h5py.Group trajectory group.

        Parameters
        ----------
        run_idx : int
        traj_idx : int

        Returns
        -------
        traj_group : h5py.Group

        """
        return self._h5['{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx)]

    def run_trajs(self, run_idx):
        """Get the trajectories group for a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        trajectories_grp : h5py.Group

        """
        return self._h5['{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES)]

    @property
    def runs(self):
        """The runs group."""
        return self.h5[RUNS]

    def run_grp(self, run_idx):
        """A group for a single run."""
        return self.runs["{}".format(run_idx)]

    def run_start_snapshot_hash(self, run_idx):
        """Hash identifier for the starting snapshot of a run from
        orchestration.

        """
        return self.run_grp(run_idx).attrs[RUN_START_SNAPSHOT_HASH]

    def run_end_snapshot_hash(self, run_idx):
        """Hash identifier for the ending snapshot of a run from
        orchestration.

        """
        return self.run_grp(run_idx).attrs[RUN_END_SNAPSHOT_HASH]

    def set_run_start_snapshot_hash(self, run_idx, snaphash):
        """Set the starting snapshot hash identifier for a run from
        orchestration.

        """

        if RUN_START_SNAPSHOT_HASH not in self.run_grp(run_idx).attrs:
            self.run_grp(run_idx).attrs[RUN_START_SNAPSHOT_HASH] = snaphash
        else:
            raise AttributeError("The snapshot has already been set.")

    def set_run_end_snapshot_hash(self, run_idx, snaphash):
        """Set the ending snapshot hash identifier for a run from
        orchestration.

        """
        if RUN_END_SNAPSHOT_HASH not in self.run_grp(run_idx).attrs:
            self.run_grp(run_idx).attrs[RUN_END_SNAPSHOT_HASH] = snaphash
        else:
            raise AttributeError("The snapshot has already been set.")

    @property
    def settings_grp(self):
        """The header settings group."""
        settings_grp = self.h5[SETTINGS]
        return settings_grp

    def decision_grp(self, run_idx):
        """Get the decision enumeration group for a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        decision_grp : h5py.Group

        """
        return self.run(run_idx)[DECISION]

    def init_walkers_grp(self, run_idx):
        """Get the group for the initial walkers for a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        init_walkers_grp : h5py.Group

        """

        return self.run(run_idx)[INIT_WALKERS]


    def records_grp(self, run_idx, run_record_key):
        """Get a record group h5py.Group for a run.

        Parameters
        ----------
        run_idx : int
        run_record_key : str
            Name of the record group

        Returns
        -------
        run_record_group : h5py.Group

        """
        path = '{}/{}/{}'.format(RUNS, run_idx, run_record_key)
        return self.h5[path]

    def resampling_grp(self, run_idx):
        """Get this record group for a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        run_record_group : h5py.Group

        """
        return self.records_grp(run_idx, RESAMPLING)

    def resampler_grp(self, run_idx):
        """Get this record group for a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        run_record_group : h5py.Group

        """
        return self.records_grp(run_idx, RESAMPLER)

    def warping_grp(self, run_idx):
        """Get this record group for a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        run_record_group : h5py.Group

        """
        return self.records_grp(run_idx, WARPING)

    def bc_grp(self, run_idx):
        """Get this record group for a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        run_record_group : h5py.Group

        """
        return self.records_grp(run_idx, BC)

    def progress_grp(self, run_idx):
        """Get this record group for a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        run_record_group : h5py.Group

        """
        return self.records_grp(run_idx, PROGRESS)

    def iter_runs(self, idxs=False, run_sel=None):
        """Generator for iterating through the runs of a file.

        Parameters
        ----------
        idxs : bool
            If True yields the run index in addition to the group.

        run_sel : list of int, optional
            If not None should be a list of the runs you want to iterate over.


        Yields
        ------
        run_idx : int, if idxs is True

        run_group : h5py.Group

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
        """Generator for iterating over trajectories in a file.

        Parameters
        ----------
        idxs : bool
            If True returns a tuple of the run index and trajectory
            index in addition to the trajectory group.
        traj_sel : list of int, optional
            If not None is a list of tuples of (run_idx, traj_idx)
            selecting which trajectories to iterate over.

        Yields
        ------
        traj_id : tuple of int, if idxs is True
            A tuple of (run_idx, traj_idx) for the group

        trajectory : h5py.Group
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
        """Iterate over the trajectories of a run.

        Parameters
        ----------
        run_idx : int
        idxs : bool
            If True returns a tuple of the run index and trajectory
            index in addition to the trajectory group.


        Returns
        -------
        iter_trajs_generator : generator for the iter_trajs method

        """
        run_sel = self.run_traj_idx_tuples([run_idx])
        return self.iter_trajs(idxs=idxs, traj_sel=run_sel)


    ### Settings

    @property
    def defined_traj_field_names(self):
        """A list of the settings defined field names all trajectories have in the file."""

        return list(self.field_feature_shapes.keys())

    @property
    def observable_field_names(self):
        """Returns a list of the names of the observables that all trajectories have.

        If this encounters observable fields that don't occur in all
        trajectories (inconsistency) raises an inconsistency error.

        """

        n_trajs = self.num_trajs
        field_names = Counter()
        for traj in self.iter_trajs():
            for name in list(traj['observables']):
                field_names[name] += 1

        # if any of the field names has not occured for every
        # trajectory we raise an error
        for field_name, count in field_names:
            if count != n_trajs:
                raise TypeError("observable field names are inconsistent")

        # otherwise return the field names for the observables
        return list(field_names.keys())

    def _check_traj_field_consistency(self, field_names):
        """Checks that every trajectory has the given fields across
        the entire dataset.

        Parameters
        ----------

        field_names : list of str
            The field names to check for.

        Returns
        -------
        consistent : bool
           True if all trajs have the fields, False otherwise

        """

        n_trajs = self.num_trajs
        field_names = Counter()
        for traj in self.iter_trajs():
            for name in field_names:
                if name in traj:
                    field_names[name] += 1

        # if any of the field names has not occured for every
        # trajectory we raise an error
        for field_name, count in field_names:
            if count != n_trajs:
                return False

        return True


    @property
    def record_fields(self):
        """The record fields for each record group which are selected for inclusion in the truncated records.

        These are the fields which are considered to be table-ified.

        Returns
        -------
        record_fields : dict of str : list of str
            Mapping of record group name to alist of the record group fields.
        """

        record_fields_grp = self.settings_grp[RECORD_FIELDS]

        record_fields_dict = {}
        for group_name, dset in record_fields_grp.items():
            record_fields_dict[group_name] = list(dset)

        return record_fields_dict

    @property
    def sparse_fields(self):
        """The trajectory fields that are sparse."""
        return self.h5['{}/{}'.format(SETTINGS, SPARSE_FIELDS)][:]

    @property
    def main_rep_idxs(self):
        """The indices of the atoms included from the full topology in the default 'positions' trajectory """

        if '{}/{}'.format(SETTINGS, MAIN_REP_IDXS) in self.h5:
            return self.h5['{}/{}'.format(SETTINGS, MAIN_REP_IDXS)][:]
        else:
            return None

    @property
    def alt_reps_idxs(self):
        """Mapping of the names of the alt reps to the indices of the atoms
        from the topology that they include in their datasets."""

        idxs_grp = self.h5['{}/{}'.format(SETTINGS, ALT_REPS_IDXS)]
        return {name : ds[:] for name, ds in idxs_grp.items()}

    @property
    def alt_reps(self):
        """Names of the alt reps."""

        idxs_grp = self.h5['{}/{}'.format(SETTINGS, ALT_REPS_IDXS)]
        return {name for name in idxs_grp.keys()}


    @property
    def field_feature_shapes(self):
        """Mapping of the names of the trajectory fields to their feature
        vector shapes."""

        shapes_grp = self.h5['{}/{}'.format(SETTINGS, FIELD_FEATURE_SHAPES_STR)]

        field_paths = _iter_field_paths(shapes_grp)

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
        """Mapping of the names of the trajectory fields to their feature
        vector numpy dtypes."""

        dtypes_grp = self.h5['{}/{}'.format(SETTINGS, FIELD_FEATURE_DTYPES_STR)]

        field_paths = _iter_field_paths(dtypes_grp)

        dtypes = {}
        for field_path in field_paths:
            dtype_str = dtypes_grp[field_path][()]
            # if there is 'None' flag for the dtype then return None
            if dtype_str == NONE_STR:
                dtypes[field_path] = None
            else:
                dtype_obj = json.loads(dtype_str)
                dtype_obj = [tuple(d) for d in dtype_obj]
                dtype = np.dtype(dtype_obj)
                dtypes[field_path] = dtype

        return dtypes

    @property
    def continuations(self):
        """The continuation relationships in this file."""
        return self.settings_grp[CONTINUATIONS][:]

    @property
    def metadata(self):
        """File metadata (h5py.attrs)."""
        return dict(self._h5.attrs)

    def decision_enum(self, run_idx):
        """Mapping of decision enumerated names to their integer representations.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        decision_enum : dict of str : int
            Mapping of the decision ID string to the integer representation.

        See Also
        --------
        WepyHDF5.decision_value_names : for the reverse mapping

        """

        enum_grp = self.decision_grp(run_idx)
        enum = {}
        for decision_name, dset in enum_grp.items():
            enum[decision_name] = dset[()]

        return enum

    def decision_value_names(self, run_idx):
        """Mapping of the integer values for decisions to the decision ID strings.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        decision_enum : dict of int : str
            Mapping of the decision integer to the decision ID string representation.

        See Also
        --------
        WepyHDF5.decision_enum : for the reverse mapping

        """
        enum_grp = self.decision_grp(run_idx)
        rev_enum = {}
        for decision_name, dset in enum_grp.items():
            value = dset[()]
            rev_enum[value] = decision_name

        return rev_enum

    ### Topology

    def get_topology(self, alt_rep=POSITIONS):
        """Get the JSON topology for a particular represenation of the positions.

        By default gives the topology for the main 'positions' field
        (when alt_rep 'positions'). To get the full topology the file
        was initialized with set `alt_rep` to `None`. Topologies for
        alternative representations (subfields of 'alt_reps') can be
        obtained by passing in the key for that alt_rep. For example,
        'all_atoms' for the field in alt_reps called 'all_atoms'.

        Parameters
        ----------
        alt_rep : str
            The base name of the alternate representation, or 'positions', or None.


        Returns
        -------
        topology : str
            The JSON topology string for the representation.

        """

        top = self.topology

        # if no alternative representation is given we just return the
        # full topology
        if alt_rep is None:
            pass

        # otherwise we either give the main representation topology
        # subset
        elif alt_rep == POSITIONS:
            top = json_top_subset(top, self.main_rep_idxs)

        # or choose one of the alternative representations
        elif alt_rep in self.alt_reps_idxs:
            top = json_top_subset(top, self.alt_reps_idxs[alt_rep])

        # and raise an error if the given alternative representation
        # is not given
        else:
            raise ValueError("alt_rep {} not found".format(alt_rep))

        return top

    @property
    def topology(self):
        """The topology for the full simulated system.

        May not be the main representation in the POSITIONS field; for
        that use the `get_topology` method.

        Returns
        -------
        topology : str
            The JSON topology string for the full representation.

        """
        return self._h5[TOPOLOGY][()]


    def get_mdtraj_topology(self, alt_rep=POSITIONS):
        """Get an mdtraj.Topology object for a system representation.

        By default gives the topology for the main 'positions' field
        (when alt_rep 'positions'). To get the full topology the file
        was initialized with set `alt_rep` to `None`. Topologies for
        alternative representations (subfields of 'alt_reps') can be
        obtained by passing in the key for that alt_rep. For example,
        'all_atoms' for the field in alt_reps called 'all_atoms'.

        Parameters
        ----------
        alt_rep : str
            The base name of the alternate representation, or 'positions', or None.

        Returns
        -------
        topology : str
            The JSON topology string for the full representation.

        """

        json_top = self.get_topology(alt_rep=alt_rep)
        return json_to_mdtraj_topology(json_top)

    ## Initial walkers

    def initial_walker_fields(self, run_idx, fields, walker_idxs=None):
        """Get fields from the initial walkers of the simulation.

        Parameters
        ----------
        run_idx : int
            Run to get initial walkers for.

        fields : list of str
            Names of the fields you want to retrieve.

        walker_idxs : None or list of int
            If None returns all of the walkers fields, otherwise a
            list of ints that are a selection from those walkers.

        Returns
        -------
        walker_fields : dict of str : array of shape
            Dictionary mapping fields to the values for all
            walkers. Frames will be either in counting order if no
            indices were requested or the order of the walker indices
            as given.

        """

        # set the walker indices if not specified
        if walker_idxs is None:
            walker_idxs = range(self.num_init_walkers(run_idx))

        init_walker_fields = {field : [] for field in fields}

        # for each walker go through and add the selected fields
        for walker_idx in walker_idxs:
            init_walker_grp = self.init_walkers_grp(run_idx)[str(walker_idx)]
            for field in fields:
                # we remove the first dimension because we just want
                # them as a single frame
                init_walker_fields[field].append(init_walker_grp[field][:][0])

        # convert the field values to arrays
        init_walker_fields = {field : np.array(val) for field, val in init_walker_fields.items()}

        return init_walker_fields

    def initial_walkers_to_mdtraj(self, run_idx, walker_idxs=None, alt_rep=POSITIONS):
        """Generate an mdtraj Trajectory from a trace of frames from the runs.

        Uses the default fields for positions (unless an alternate
        representation is specified) and box vectors which are assumed
        to be present in the trajectory fields.

        The time value for the mdtraj trajectory is set to the cycle
        indices for each trace frame.

        This is useful for converting WepyHDF5 data to common
        molecular dynamics data formats accessible through the mdtraj
        library.

        Parameters
        ----------
        run_idx : int
            Run to get initial walkers for.

        fields : list of str
            Names of the fields you want to retrieve.

        walker_idxs : None or list of int
            If None returns all of the walkers fields, otherwise a
            list of ints that are a selection from those walkers.

        alt_rep : None or str
            If None uses default 'positions' representation otherwise
            chooses the representation from the 'alt_reps' compound field.

        Returns
        -------
        traj : mdtraj.Trajectory

        """

        rep_path = self._choose_rep_path(alt_rep)

        init_walker_fields = self.initial_walker_fields(run_idx, [rep_path, BOX_VECTORS],
                                                        walker_idxs=walker_idxs)

        return self.traj_fields_to_mdtraj(init_walker_fields, alt_rep=alt_rep)


    ### Counts and Indexing

    @property
    def num_atoms(self):
        """The number of atoms in the full topology representation."""
        return self.h5['{}/{}'.format(SETTINGS, N_ATOMS)][()]

    @property
    def num_dims(self):
        """The number of spatial dimensions in the positions and alt_reps trajectory fields."""
        return self.h5['{}/{}'.format(SETTINGS, N_DIMS_STR)][()]

    @property
    def num_runs(self):
        """The number of runs in the file."""
        return len(self._h5[RUNS])

    @property
    def num_trajs(self):
        """The total number of trajectories in the entire file."""
        return len(list(self.run_traj_idx_tuples()))

    def num_init_walkers(self, run_idx):
        """The number of initial walkers for a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        n_walkers : int

        """

        return len(self.init_walkers_grp(run_idx))

    def num_walkers(self, run_idx, cycle_idx):
        """Get the number of walkers at a given cycle in a run.

        Parameters
        ----------
        run_idx : int

        cycle_idx : int

        Returns
        -------
        n_walkers : int

        """

        if cycle_idx >= self.num_run_cycles(run_idx):
            raise ValueError(
                f"Run {run_idx} has {self.num_run_cycles(run_idx)} cycles, {cycle_idx} requested")

        # TODO: currently we do not have a well-defined mechanism for
        # actually storing variable number of walkers in the
        # trajectory data so just return the number of trajectories
        return self.num_run_trajs(run_idx)

    def num_run_trajs(self, run_idx):
        """The number of trajectories in a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        n_trajs : int

        """
        return len(self._h5['{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES)])

    def num_run_cycles(self, run_idx):
        """The number of cycles in a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        n_cycles : int

        """
        return self.num_traj_frames(run_idx, 0)

    def num_traj_frames(self, run_idx, traj_idx):
        """The number of frames in a given trajectory.

        Parameters
        ----------
        run_idx : int
        traj_idx : int

        Returns
        -------
        n_frames : int

        """
        return self.traj(run_idx, traj_idx)[POSITIONS].shape[0]

    @property
    def run_idxs(self):
        """The indices of the runs in the file."""
        return list(range(len(self._h5[RUNS])))

    def run_traj_idxs(self, run_idx):
        """The indices of trajectories in a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        traj_idxs : list of int

        """
        return list(range(len(self._h5['{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES)])))

    def run_traj_idx_tuples(self, runs=None):
        """Get identifier tuples (run_idx, traj_idx) for all trajectories in
        all runs.

        Parameters
        ----------
        runs : list of int, optional
            If not None, a list of run indices to restrict to.

        Returns
        -------
        run_traj_tuples : list of tuple of int
            A listing of all trajectories by their identifying tuple
            of (run_idx, traj_idx).

        """
        tups = []
        if runs is None:
            run_idxs = self.run_idxs
        else:
            run_idxs = runs
        for run_idx in run_idxs:
            for traj_idx in self.run_traj_idxs(run_idx):
                tups.append((run_idx, traj_idx))

        return tups

    def get_traj_field_cycle_idxs(self, run_idx, traj_idx, field_path):
        """Returns the cycle indices for a sparse trajectory field.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_path : str
            Name of the trajectory field

        Returns
        -------
        cycle_idxs : arraylike of int

        """

        traj_path = '{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx)

        if not field_path in self._h5[traj_path]:
            raise KeyError("key for field {} not found".format(field_path))

        # if the field is not sparse just return the cycle indices for
        # that run
        if field_path not in self.sparse_fields:
            cycle_idxs = np.array(range(self.num_run_cycles(run_idx)))
        else:
            cycle_idxs = self._h5[traj_path][field_path][SPARSE_IDXS][:]

        return cycle_idxs

    def next_run_idx(self):
        """The index of the next run if it were to be added.

        Because runs are named as the integer value of the order they
        were added this gives the index of the next run that would be
        added.

        Returns
        -------
        next_run_idx : int

        """
        return self.num_runs

    def next_run_traj_idx(self, run_idx):
        """The index of the next trajectory for this run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        next_traj_idx : int

        """
        return self.num_run_trajs(run_idx)

    ### Aggregation

    def is_run_contig(self, run_idxs):
        """This method checks that if a given list of run indices is a valid
        contig or not.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that would make up the contig in order.

        Returns
        -------
        is_contig : bool

        """
        run_idx_continuations = [np.array([run_idxs[idx+1], run_idxs[idx]])
                            for idx in range(len(run_idxs)-1)]
        #gets the contigs array
        continuations = self.settings_grp[CONTINUATIONS][:]

        # checks if sub contigs are in contigs list or not.
        for run_continuous in run_idx_continuations:
            contig = False
            for continuous in continuations:
                if np.array_equal(run_continuous, continuous):
                    contig = True
            if not contig:
                return False

        return True

    def clone(self, path, mode='x'):
        """Clone the header information of this file into another file.

        Clones this WepyHDF5 file without any of the actual runs and run
        data. This includes the topology, units, sparse_fields,
        feature shapes and dtypes, alt_reps, and main representation
        information.

        This method will flush the buffers for this file.

        Does not preserve metadata pertaining to inter-run
        relationships like continuations.

        Parameters
        ----------
        path : str
            File path to save the new file.
        mode : str
            The mode to open the new file with.

        Returns
        -------
        new_file : h5py.File
            The handle to the new file. It will be closed.

        """

        assert mode in ['w', 'w-', 'x'], "must be opened in a file creation mode"

        # we manually construct an HDF5 and copy the groups over
        new_h5 = h5py.File(path, mode=mode, libver=H5PY_LIBVER)

        new_h5.require_group(RUNS)

        # flush the datasets buffers
        self.h5.flush()
        new_h5.flush()

        # copy the existing datasets to the new one
        h5py.h5o.copy(self._h5.id, TOPOLOGY.encode(), new_h5.id, TOPOLOGY.encode())
        h5py.h5o.copy(self._h5.id, UNITS.encode(), new_h5.id, UNITS.encode())
        h5py.h5o.copy(self._h5.id, SETTINGS.encode(), new_h5.id, SETTINGS.encode())

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

        # for the settings we need to get rid of the data for interun
        # relationships like the continuations, so we reinitialize the
        # continuations for the new file
        new_wepy_h5._init_continuations()

        # close the h5py.File and set the attribute to closed
        new_wepy_h5._h5.close()
        new_wepy_h5.closed = True


        # return the runless WepyHDF5 object
        return new_wepy_h5


    def link_run(self, filepath, run_idx, continue_run=None, **kwargs):
        """Add a run from another file to this one as an HDF5 external
        link.

        Parameters
        ----------
        filepath : str
            File path to the HDF5 file that the run is on.
        run_idx : int
            The run index from the target file you want to link.
        continue_run : int, optional
            The run from the linking WepyHDF5 file you want the target
            linked run to continue.

        kwargs : dict
            Adds metadata (h5py.attrs) to the linked run.

        Returns
        -------
        linked_run_idx : int
            The index of the linked run in the linking file.

        """

        # link to the external run
        ext_run_link = h5py.ExternalLink(filepath, '{}/{}'.format(RUNS, run_idx))

        # the run index in this file, as determined by the counter
        here_run_idx = self.next_run_idx()

        # set the local run as the external link to the other run
        self._h5['{}/{}'.format(RUNS, here_run_idx)] = ext_run_link

        # run the initialization routines for adding a run
        self._add_run_init(here_run_idx, continue_run=continue_run)

        run_grp = self._h5['{}/{}'.format(RUNS, here_run_idx)]

        # add metadata if given
        for key, val in kwargs.items():
            if key != RUN_IDX:
                run_grp.attrs[key] = val
            else:
                warn('run_idx metadata is set by wepy and cannot be used', RuntimeWarning)

        return here_run_idx

    def link_file_runs(self, wepy_h5_path):
        """Link all runs from another WepyHDF5 file.

        This preserves continuations within that file. This will open
        the file if not already opened.

        Parameters
        ----------
        wepy_h5_path : str
            Filepath to the file you want to link runs from.

        Returns
        -------
        new_run_idxs : list of int
            The new run idxs from the linking file.
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

    def extract_run(self, filepath, run_idx,
                 continue_run=None,
                 run_slice=None,
                 **kwargs):
        """Add a run from another file to this one by copying it and
        truncating it if necessary.

        Parameters
        ----------
        filepath : str
            File path to the HDF5 file that the run is on.
        run_idx : int
            The run index from the target file you want to link.
        continue_run : int, optional
            The run from the linking WepyHDF5 file you want the target
            linked run to continue.
        run_slice : 

        kwargs : dict
            Adds metadata (h5py.attrs) to the linked run.

        Returns
        -------
        linked_run_idx : int
            The index of the linked run in the linking file.

        """

        # close ourselves if not already done, so we can write using
        # the lower level API
        was_open = False
        if not self.closed:
            self.close()
            was_open = True

        # do the copying

        # open the other file and get the runs in it and the
        # continuations it has
        wepy_h5 = WepyHDF5(filepath, mode='r')



        with self:
            # normalize our HDF5s path
            self_path = osp.realpath(self.filename)
            # the run index in this file, as determined by the counter
            here_run_idx = self.next_run_idx()

        # get the group name for the new run in this HDF5
        target_grp_path = "/runs/{}".format(here_run_idx)

        with wepy_h5:
            # link the next run, and get its new run index
            new_h5 = wepy_h5.copy_run_slice(run_idx, self_path,
                                      target_grp_path,
                                      run_slice=run_slice,
                                      mode='r+')

            # close it since we are done
            new_h5.close()


        with self:

            # run the initialization routines for adding a run, just
            # sets some metadata
            self._add_run_init(here_run_idx, continue_run=continue_run)

            run_grp = self._h5['{}/{}'.format(RUNS, here_run_idx)]

            # add metadata if given
            for key, val in kwargs.items():
                if key != RUN_IDX:
                    run_grp.attrs[key] = val
                else:
                    warn('run_idx metadata is set by wepy and cannot be used', RuntimeWarning)

        if was_open:
            self.open()

        return here_run_idx


    def extract_file_runs(self, wepy_h5_path,
                          run_slices=None):
        """Extract (copying and truncating appropriately) all runs from
        another WepyHDF5 file.

        This preserves continuations within that file. This will open
        the file if not already opened.

        Parameters
        ----------
        wepy_h5_path : str
            Filepath to the file you want to link runs from.

        Returns
        -------
        new_run_idxs : list of int
            The new run idxs from the linking file.

        """

        if run_slices is None:
            run_slices = {}


        # open the other file and get the runs in it and the
        # continuations it has
        wepy_h5 = WepyHDF5(wepy_h5_path, mode='r')
        with wepy_h5:
            # the run idx in the external file
            ext_run_idxs = wepy_h5.run_idxs
            continuations = wepy_h5.continuations

        # then for each run in it copy them to this file
        new_run_idxs = []
        for ext_run_idx in ext_run_idxs:

            # get the run_slice spec for the run in the other file
            run_slice = run_slices[ext_run_idx]

            # get the index this run should be when it is added
            new_run_idx = self.extract_run(wepy_h5_path, ext_run_idx,
                                           run_slice=run_slice)

            # save that run idx
            new_run_idxs.append(new_run_idx)

        was_closed = False
        if self.closed:
            self.open()
            was_closed = True

        # copy the continuations over translating the run idxs,
        # for each continuation in the other files continuations
        for continuation in continuations:

            # translate each run index from the external file
            # continuations to the run idxs they were just assigned in
            # this file
            self.add_continuation(new_run_idxs[continuation[0]],
                                  new_run_idxs[continuation[1]])

        if was_closed:
            self.close()

        return new_run_idxs


    def join(self, other_h5):
        """Given another WepyHDF5 file object does a left join on this
        file, renumbering the runs starting from this file.

        This function uses the H5O function for copying. Data will be
        copied not linked.

        Parameters
        ----------
        other_h5 : h5py.File
            File handle to the file you want to join to this one.

        """

        with other_h5 as h5:
            for run_idx in h5.run_idxs:
                # the other run group handle
                other_run = h5.run(run_idx)
                # copy this run to this file in the next run_idx group
                self.h5.copy(other_run, '{}/{}'.format(RUNS, self.next_run_idx()))


    ### initialization and data generation

    def add_metadata(self, key, value):
        """Add metadata for the whole file.

        Parameters
        ----------
        key : str
        value : h5py value
            h5py valid metadata value.

        """
        self._h5.attrs[key] = value


    def init_record_fields(self, run_record_key, record_fields):
        """Initialize the settings record fields for a record group in the
        settings group.

        Save which records are to be considered from a run record group's
        datasets to be in the table like representation. This exists
        to allow there to large and small datasets for records to be
        stored together but allow for a more compact single table like
        representation to be produced for serialization.

        Parameters
        ----------
        run_record_key : str
            Name of the record group you want to set this for.
        record_fields : list of str
            Names of the fields you want to set as record fields.

        """

        record_fields_grp = self.settings_grp[RECORD_FIELDS]

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
        """Initialize the record fields for this record group.

        Parameters
        ----------
        resampler : object implementing the Resampler interface
            The resampler which contains the data for which record fields to set.

        """
        self.init_record_fields(RESAMPLING, resampler.resampling_record_field_names())

    def init_resampler_record_fields(self, resampler):
        """Initialize the record fields for this record group.

        Parameters
        ----------
        resampler : object implementing the Resampler interface
            The resampler which contains the data for which record fields to set.

        """
        self.init_record_fields(RESAMPLER, resampler.resampler_record_field_names())

    def init_bc_record_fields(self, bc):
        """Initialize the record fields for this record group.

        Parameters
        ----------
        bc : object implementing the BoundaryConditions interface
            The boundary conditions object which contains the data for which record fields to set.

        """
        self.init_record_fields(BC, bc.bc_record_field_names())

    def init_warping_record_fields(self, bc):
        """Initialize the record fields for this record group.

        Parameters
        ----------
        bc : object implementing the BoundaryConditions interface
            The boundary conditions object which contains the data for which record fields to set.

        """
        self.init_record_fields(WARPING, bc.warping_record_field_names())

    def init_progress_record_fields(self, bc):
        """Initialize the record fields for this record group.

        Parameters
        ----------
        bc : object implementing the BoundaryConditions interface
            The boundary conditions object which contains the data for which record fields to set.

        """
        self.init_record_fields(PROGRESS, bc.progress_record_field_names())

    def add_continuation(self, continuation_run, base_run):
        """Add a continuation between runs.

        Parameters
        ----------
        continuation_run : int
            The run index of the run that will be continuing another
        base_run : int
            The run that is being continued.

        """

        continuations_dset = self.settings_grp[CONTINUATIONS]
        continuations_dset.resize((continuations_dset.shape[0] + 1, continuations_dset.shape[1],))
        continuations_dset[continuations_dset.shape[0] - 1] = np.array([continuation_run, base_run])

    def new_run(self, init_walkers, continue_run=None, **kwargs):
        """Initialize a new run.

        Parameters
        ----------
        init_walkers : list of objects implementing the Walker interface
            The walkers that will be the start of this run.
        continue_run : int, optional
            If this run is a continuation of another set which one it is continuing.

        kwargs : dict
            Metadata to set for the run.

        Returns
        -------
        run_grp : h5py.Group
            The group of the newly created run.

        """

        # check to see if the continue_run is actually in this file
        if continue_run is not None:
            if continue_run not in self.run_idxs:
                raise ValueError("The continue_run idx given, {}, is not present in this file".format(
                    continue_run))

        # get the index for this run
        new_run_idx = self.next_run_idx()

        # create a new group named the next integer in the counter
        run_grp = self._h5.create_group('{}/{}'.format(RUNS, new_run_idx))


        # set the initial walkers group
        init_walkers_grp = run_grp.create_group(INIT_WALKERS)

        self._add_init_walkers(init_walkers_grp, init_walkers)

        # initialize the walkers group
        traj_grp = run_grp.create_group(TRAJECTORIES)


        # run the initialization routines for adding a run
        self._add_run_init(new_run_idx, continue_run=continue_run)


        # add metadata if given
        for key, val in kwargs.items():
            if key != RUN_IDX:
                run_grp.attrs[key] = val
            else:
                warn('run_idx metadata is set by wepy and cannot be used', RuntimeWarning)

        return run_grp

    # application level methods for setting the fields for run record
    # groups given the objects themselves
    def init_run_resampling(self, run_idx, resampler):
        """Initialize data for resampling records.

        Initialized the run record group as well as settings for the
        fields.

        This method also creates the decision group for the run.

        Parameters
        ----------
        run_idx : int
        resampler : object implementing the Resampler interface
            The resampler which contains the data for which record fields to set.

        Returns
        -------
        record_grp : h5py.Group

        """

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
        """Initialize the decision group for the run resampling records.

        Parameters
        ----------
        run_idx : int

        resampler : object implementing the Resampler interface
            The resampler which contains the data for which record fields to set.

        """

        self.init_run_fields_resampling_decision(run_idx, resampler.DECISION.enum_dict_by_name())

    def init_run_resampler(self, run_idx, resampler):
        """Initialize data for this record group in a run.

        Initialized the run record group as well as settings for the
        fields.

        Parameters
        ----------
        run_idx : int
        resampler : object implementing the Resampler interface
            The resampler which contains the data for which record fields to set.

        Returns
        -------
        record_grp : h5py.Group

        """

        fields = resampler.resampler_fields()

        grp = self.init_run_record_grp(run_idx, RESAMPLER, fields)

        return grp

    def init_run_warping(self, run_idx, bc):
        """Initialize data for this record group in a run.

        Initialized the run record group as well as settings for the
        fields.

        Parameters
        ----------
        run_idx : int

        bc : object implementing the BoundaryConditions interface
            The boundary conditions object which contains the data for which record fields to set.

        Returns
        -------
        record_grp : h5py.Group

        """

        fields = bc.warping_fields()
        grp = self.init_run_record_grp(run_idx, WARPING, fields)

        return grp

    def init_run_progress(self, run_idx, bc):
        """Initialize data for this record group in a run.

        Initialized the run record group as well as settings for the
        fields.

        Parameters
        ----------
        run_idx : int

        bc : object implementing the BoundaryConditions interface
            The boundary conditions object which contains the data for which record fields to set.

        Returns
        -------
        record_grp : h5py.Group

        """

        fields = bc.progress_fields()

        grp = self.init_run_record_grp(run_idx, PROGRESS, fields)

        return grp

    def init_run_bc(self, run_idx, bc):
        """Initialize data for this record group in a run.

        Initialized the run record group as well as settings for the
        fields.

        Parameters
        ----------
        run_idx : int

        bc : object implementing the BoundaryConditions interface
            The boundary conditions object which contains the data for which record fields to set.

        Returns
        -------
        record_grp : h5py.Group

        """

        fields = bc.bc_fields()

        grp = self.init_run_record_grp(run_idx, BC, fields)

        return grp

    # application level methods for initializing the run records
    # groups with just the fields and without the objects
    def init_run_fields_resampling(self, run_idx, fields):
        """Initialize this record group fields datasets.

        Parameters
        ----------
        run_idx : int
        fields : list of str
            Names of the fields to initialize

        Returns
        -------
        record_grp : h5py.Group

        """

        grp = self.init_run_record_grp(run_idx, RESAMPLING, fields)

        return grp

    def init_run_fields_resampling_decision(self, run_idx, decision_enum_dict):
        """Initialize the decision group for this run.

        Parameters
        ----------
        run_idx : int
        decision_enum_dict : dict of str : int
            Mapping of decision ID strings to integer representation.

        """

        decision_grp = self.run(run_idx).create_group(DECISION)
        for name, value in decision_enum_dict.items():
            decision_grp.create_dataset(name, data=value)


    def init_run_fields_resampler(self, run_idx, fields):
        """Initialize this record group fields datasets.

        Parameters
        ----------
        run_idx : int
        fields : list of str
            Names of the fields to initialize

        Returns
        -------
        record_grp : h5py.Group

        """

        grp = self.init_run_record_grp(run_idx, RESAMPLER, fields)

        return grp

    def init_run_fields_warping(self, run_idx, fields):
        """Initialize this record group fields datasets.

        Parameters
        ----------
        run_idx : int
        fields : list of str
            Names of the fields to initialize

        Returns
        -------
        record_grp : h5py.Group

        """

        grp = self.init_run_record_grp(run_idx, WARPING, fields)

        return grp

    def init_run_fields_progress(self, run_idx, fields):
        """Initialize this record group fields datasets.

        Parameters
        ----------
        run_idx : int
        fields : list of str
            Names of the fields to initialize

        Returns
        -------
        record_grp : h5py.Group

        """

        grp = self.init_run_record_grp(run_idx, PROGRESS, fields)

        return grp

    def init_run_fields_bc(self, run_idx, fields):
        """Initialize this record group fields datasets.

        Parameters
        ----------
        run_idx : int
        fields : list of str
            Names of the fields to initialize

        Returns
        -------
        record_grp : h5py.Group

        """

        grp = self.init_run_record_grp(run_idx, BC, fields)

        return grp


    def init_run_record_grp(self, run_idx, run_record_key, fields):
        """Initialize a record group for a run.

        Parameters
        ----------
        run_idx : int
        run_record_key : str
            The name of the record group.
        fields : list of str
            The names of the fields to set for the record group.
        """

        # initialize the record group based on whether it is sporadic
        # or continual
        if self._is_sporadic_records(run_record_key):
            grp = self._init_run_sporadic_record_grp(run_idx, run_record_key,
                                                     fields)
        else:
            grp = self._init_run_continual_record_grp(run_idx, run_record_key,
                                                      fields)


    # TODO: should've been removed already just double checking things are good without it
    # def traj_n_frames(self, run_idx, traj_idx):
    #     """

    #     Parameters
    #     ----------
    #     run_idx :
            
    #     traj_idx :
            

    #     Returns
    #     -------

    #     """
    #     return self.traj(run_idx, traj_idx)[POSITIONS].shape[0]

    def add_traj(self, run_idx, data, weights=None, sparse_idxs=None, metadata=None):
        """Add a full trajectory to a run.

        Parameters
        ----------
        run_idx : int
        data : dict of str : arraylike
            Mapping of trajectory fields to the data for them to add.
        weights : 1-D arraylike of float
            The weights of each frame. If None defaults all frames to 1.0.

        sparse_idxs : list of int
            Cycle indices the data corresponds to.

        metadata : dict of str : value
            Metadata for the trajectory.


        Returns
        -------
        traj_grp : h5py.Group

        """

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
                        '{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx))

        # add the run_idx as metadata
        traj_grp.attrs[RUN_IDX] = run_idx
        # add the traj_idx as metadata
        traj_grp.attrs[TRAJ_IDX] = traj_idx


        # add the rest of the metadata if given
        for key, val in metadata.items():
            if not key in [RUN_IDX, TRAJ_IDX]:
                traj_grp.attrs[key] = val
            else:
                warn("run_idx and traj_idx are used by wepy and cannot be set", RuntimeWarning)


        # check to make sure the positions are the right shape
        assert traj_data[POSITIONS].shape[1] == self.num_atoms, \
            "positions given have different number of atoms: {}, should be {}".format(
                traj_data[POSITIONS].shape[1], self.num_atoms)
        assert traj_data[POSITIONS].shape[2] == self.num_dims, \
            "positions given have different number of dims: {}, should be {}".format(
                traj_data[POSITIONS].shape[2], self.num_dims)

        # add datasets to the traj group

        # weights
        traj_grp.create_dataset(WEIGHTS, data=weights, dtype=WEIGHT_DTYPE,
                                maxshape=(None, *WEIGHT_SHAPE))
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

    def extend_traj(self, run_idx, traj_idx, data, weights=None):
        """Extend a trajectory with data for all fields.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        data : dict of str : arraylike
            The data to add for each field of the trajectory. Must all
            have the same first dimension.
        weights : arraylike
            Weights for the frames of the trajectory. If None defaults all frames to 1.0.

        """

        if self._wepy_mode == 'c-':
            assert self._append_flags[dataset_key], "dataset is not available for appending to"

        # convenient alias
        traj_data = data

        # number of frames to add
        n_new_frames = traj_data[POSITIONS].shape[0]

        n_frames = self.num_traj_frames(run_idx, traj_idx)

        # calculate the new sparse idxs for sparse fields that may be
        # being added
        sparse_idxs = np.array(range(n_frames, n_frames + n_new_frames))

        # get the trajectory group
        traj_grp = self._h5['{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx)]

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

            # if the field hasn't been initialized yet initialize it,
            # unless we are in SWMR mode
            if not field_path in traj_grp:

                # if in SWMR mode you cannot create groups so if we
                # are in SWMR mode raise a warning that the data won't
                # be recorded
                if self.swmr_mode:
                    warn("New datasets cannot be created while in SWMR mode.  The field {} will"
                    "not be saved. If you want to save this it must be"
                         "previously created".format(field_path))
                else:

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

    ## application level append methods for run records groups

    def extend_cycle_warping_records(self, run_idx, cycle_idx, warping_data):
        """Add records for each field for this record group.

        Parameters
        ----------
        run_idx : int
        cycle_idx : int
            The cycle index these records correspond to.
        warping_data : dict of str : arraylike
            Mapping of the record group fields to a collection of
            values for each field.

        """
        self.extend_cycle_run_group_records(run_idx, WARPING, cycle_idx, warping_data)

    def extend_cycle_bc_records(self, run_idx, cycle_idx, bc_data):
        """Add records for each field for this record group.

        Parameters
        ----------
        run_idx : int
        cycle_idx : int
            The cycle index these records correspond to.
        bc_data : dict of str : arraylike
            Mapping of the record group fields to a collection of
            values for each field.

        """

        self.extend_cycle_run_group_records(run_idx, BC, cycle_idx, bc_data)

    def extend_cycle_progress_records(self, run_idx, cycle_idx, progress_data):
        """Add records for each field for this record group.

        Parameters
        ----------
        run_idx : int
        cycle_idx : int
            The cycle index these records correspond to.
        progress_data : dict of str : arraylike
            Mapping of the record group fields to a collection of
            values for each field.

        """
        self.extend_cycle_run_group_records(run_idx, PROGRESS, cycle_idx, progress_data)

    def extend_cycle_resampling_records(self, run_idx, cycle_idx, resampling_data):
        """Add records for each field for this record group.

        Parameters
        ----------
        run_idx : int
        cycle_idx : int
            The cycle index these records correspond to.
        resampling_data : dict of str : arraylike
            Mapping of the record group fields to a collection of
            values for each field.

        """

        self.extend_cycle_run_group_records(run_idx, RESAMPLING, cycle_idx, resampling_data)

    def extend_cycle_resampler_records(self, run_idx, cycle_idx, resampler_data):
        """Add records for each field for this record group.

        Parameters
        ----------
        run_idx : int
        cycle_idx : int
            The cycle index these records correspond to.
        resampler_data : dict of str : arraylike
            Mapping of the record group fields to a collection of
            values for each field.

        """
        self.extend_cycle_run_group_records(run_idx, RESAMPLER, cycle_idx, resampler_data)

    def extend_cycle_run_group_records(self, run_idx, run_record_key, cycle_idx, fields_data):
        """Extend data for a whole records group.

        This must have the cycle index for the data it is appending as
        this is done for sporadic and continual datasets.

        Parameters
        ----------
        run_idx : int
        run_record_key : str
            Name of the record group.
        cycle_idx : int
            The cycle index these records correspond to.
        fields_data : dict of str : arraylike
            Mapping of the field name to the values for the records being added.

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

    ### Analysis Routines

    ## Record Getters

    def run_records(self, run_idx, run_record_key):
        """Get the records for a record group for a single run.

        Parameters
        ----------
        run_idx : int
        run_record_key : str
            The name of the record group.

        Returns
        -------
        records : list of namedtuple objects
            The list of records for the run's record group.

        """

        # wrap this in a list since the underlying functions accept a
        # list of records
        run_idxs = [run_idx]

        return self.run_contig_records(run_idxs, run_record_key)

    def run_contig_records(self, run_idxs, run_record_key):
        """Get the records for a record group for the contig that is formed by
        the run indices.

        This alters the cycle indices for the records so that they
        appear to have come from a single run. That is they are the
        cycle indices of the contig.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        run_record_key : str
            Name of the record group.

        Returns
        -------
        records : list of namedtuple objects
            The list of records for the contig's record group.

        """

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

    def run_records_dataframe(self, run_idx, run_record_key):
        """Get the records for a record group for a single run in the form of
        a pandas DataFrame.

        Parameters
        ----------
        run_idx : int
        run_record_key : str
            Name of record group.

        Returns
        -------
        record_df : pandas.DataFrame
        """
        records = self.run_records(run_idx, run_record_key)
        return pd.DataFrame(records)

    def run_contig_records_dataframe(self, run_idxs, run_record_key):
        """Get the records for a record group for a contig of runs in the form
	of a pandas DataFrame.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)
        run_record_key : str
            The name of the record group.

        Returns
        -------
        records_df : pandas.DataFrame

        """
        records = self.run_contig_records(run_idxs, run_record_key)
        return pd.DataFrame(records)

    # application level specific methods for each main group

    # resampling
    def resampling_records(self, run_idxs):
        """Get the records this record group for the contig that is formed by
        the run indices.

        This alters the cycle indices for the records so that they
        appear to have come from a single run. That is they are the
        cycle indices of the contig.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        records : list of namedtuple objects
            The list of records for the contig's record group.

        """

        return self.run_contig_records(run_idxs, RESAMPLING)

    def resampling_records_dataframe(self, run_idxs):
        """Get the records for this record group for a contig of runs in the
	form of a pandas DataFrame.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        records_df : pandas.DataFrame

        """

        return pd.DataFrame(self.resampling_records(run_idxs))

    # resampler records
    def resampler_records(self, run_idxs):
        """Get the records this record group for the contig that is formed by
        the run indices.

        This alters the cycle indices for the records so that they
        appear to have come from a single run. That is they are the
        cycle indices of the contig.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        records : list of namedtuple objects
            The list of records for the contig's record group.

        """

        return self.run_contig_records(run_idxs, RESAMPLER)

    def resampler_records_dataframe(self, run_idxs):
        """Get the records for this record group for a contig of runs in the
	form of a pandas DataFrame.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        records_df : pandas.DataFrame

        """

        return pd.DataFrame(self.resampler_records(run_idxs))

    # warping
    def warping_records(self, run_idxs):
        """Get the records this record group for the contig that is formed by
        the run indices.

        This alters the cycle indices for the records so that they
        appear to have come from a single run. That is they are the
        cycle indices of the contig.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        records : list of namedtuple objects
            The list of records for the contig's record group.

        """

        return self.run_contig_records(run_idxs, WARPING)

    def warping_records_dataframe(self, run_idxs):
        """Get the records for this record group for a contig of runs in the
	form of a pandas DataFrame.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        records_df : pandas.DataFrame

        """

        return pd.DataFrame(self.warping_records(run_idxs))

    # boundary conditions
    def bc_records(self, run_idxs):
        """Get the records this record group for the contig that is formed by
        the run indices.

        This alters the cycle indices for the records so that they
        appear to have come from a single run. That is they are the
        cycle indices of the contig.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        records : list of namedtuple objects
            The list of records for the contig's record group.

        """

        return self.run_contig_records(run_idxs, BC)

    def bc_records_dataframe(self, run_idxs):
        """Get the records for this record group for a contig of runs in the
	form of a pandas DataFrame.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        records_df : pandas.DataFrame

        """

        return pd.DataFrame(self.bc_records(run_idxs))

    # progress
    def progress_records(self, run_idxs):
        """Get the records this record group for the contig that is formed by
        the run indices.

        This alters the cycle indices for the records so that they
        appear to have come from a single run. That is they are the
        cycle indices of the contig.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        records : list of namedtuple objects
            The list of records for the contig's record group.

        """

        return self.run_contig_records(run_idxs, PROGRESS)

    def progress_records_dataframe(self, run_idxs):
        """Get the records for this record group for a contig of runs in the
	form of a pandas DataFrame.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        records_df : pandas.DataFrame

        """

        return pd.DataFrame(self.progress_records(run_idxs))


    def run_resampling_panel(self, run_idx):
        """Generate a resampling panel from the resampling records of a run.

        Parameters
        ----------
        run_idx : int

        Returns
        -------
        resampling_panel : list of list of list of namedtuple records
            The panel (list of tables) of resampling records in order
            (cycle, step, walker)

        """
        return self.run_contig_resampling_panel([run_idx])

    def run_contig_resampling_panel(self, run_idxs):
        """Generate a resampling panel from the resampling records of a
        contig, which is a series of runs.

        Parameters
        ----------
        run_idxs : list of int
            The run indices that form a contig. (i.e. element 1
            continues element 0)

        Returns
        -------
        resampling_panel : list of list of list of namedtuple records
            The panel (list of tables) of resampling records in order
            (cycle, step, walker)

        """
        # check the contig to make sure it is a valid contig
        if not self.is_run_contig(run_idxs):
            raise ValueError("The run_idxs provided are not a valid contig, {}.".format(
                run_idxs))

        # make the resampling panel from the resampling records for the contig
        contig_resampling_panel = resampling_panel(self.resampling_records(run_idxs),
                                                   is_sorted=False)

        return contig_resampling_panel


    # Trajectory Field Setters

    def add_run_observable(self, run_idx, observable_name, data, sparse_idxs=None):
        """Add a trajectory sub-field in the compound field "observables" for
        a single run.

        Parameters
        ----------
        run_idx : int
        observable_name : str
            What to name the observable subfield.
        data : arraylike of shape (n_trajs, feature_vector_shape[0], ...)
            The data for all of the trajectories that will be set to
            this observable field.
        sparse_idxs : list of int, optional
            If not None, specifies the cycle indices this data corresponds to.

        """
        obs_path = '{}/{}'.format(OBSERVABLES, observable_name)

        self._add_run_field(run_idx, obs_path, data, sparse_idxs=sparse_idxs)

    def add_traj_observable(self, observable_name, data, sparse_idxs=None):
        """Add a trajectory sub-field in the compound field "observables" for
        an entire file, on a trajectory basis.

        Parameters
        ----------
        observable_name : str
            What to name the observable subfield.

        data : list of arraylike
            The data for each run are the elements of this
            argument. Each element is an arraylike of shape
            (n_traj_frames, feature_vector_shape[0],...) where the
            n_run_frames is the number of frames in trajectory.

        sparse_idxs : list of list of int, optional
            If not None, specifies the cycle indices this data
            corresponds to. First by run, then by trajectory.

        """
        obs_path = '{}/{}'.format(OBSERVABLES, observable_name)

        run_results = []

        for run_idx in range(self.num_runs):

            run_num_trajs = self.num_run_trajs(run_idx)
            run_results.append([])

            for traj_idx in range(run_num_trajs):
                run_results[run_idx].append(data[(run_idx * run_num_trajs) + traj_idx])

        run_sparse_idxs = None
        if sparse_idxs is not None:
            run_sparse_idxs = []

            for run_idx in range(self.num_runs):

                run_num_trajs = self.num_run_trajs(run_idx)
                run_sparse_idxs.append([])

                for traj_idx in range(run_num_trajs):
                    run_sparse_idxs[run_idx].append(sparse_idxs[(run_idx * run_num_trajs) + traj_idx])

        self.add_observable(observable_name, run_results,
                            sparse_idxs=run_sparse_idxs)



    def add_observable(self, observable_name, data, sparse_idxs=None):
        """Add a trajectory sub-field in the compound field "observables" for
        an entire file, on a compound run and trajectory basis.

        Parameters
        ----------
        observable_name : str
            What to name the observable subfield.

        data : list of list of arraylike

            The data for each run are the elements of this
            argument. Each element is a list of the trajectory
            observable arraylikes of shape (n_traj_frames,
            feature_vector_shape[0],...).

        sparse_idxs : list of list of int, optional
            If not None, specifies the cycle indices this data
            corresponds to. First by run, then by trajectory.

        """
        obs_path = '{}/{}'.format(OBSERVABLES, observable_name)

        self._add_field(
            obs_path,
            data,
            sparse_idxs=sparse_idxs,
        )

    def compute_observable(self, func, fields, args,
                           map_func=map,
                           traj_sel=None,
                           save_to_hdf5=None, idxs=False, return_results=True):
        """Compute an observable on the trajectory data according to a
        function. Optionally save that data in the observables data group for
        the trajectory.

        Parameters
        ----------
        func : callable
            The function to apply to the trajectory fields (by
            cycle). Must accept a dictionary mapping string trajectory
            field names to a feature vector for that cycle and return
            an arraylike. May accept other positional arguments as well.

        fields : list of str
            A list of trajectory field names to pass to the mapped function.

        args : tuple
            A single tuple of arguments which will be expanded and
            passed to the mapped function for every evaluation.

        map_func : callable
            The mapping function. The implementation of how to map the
            computation function over the data. Default is the python
            builtin `map` function. Can be a parallel implementation
            for example.

        traj_sel : list of tuple, optional
            If not None, a list of trajectory identifier tuple
            (run_idx, traj_idx) to restrict the computation to.

        save_to_hdf5 : None or string, optional
            If not None, a string that specifies the name of the
            observables sub-field that the computed values will be saved to.

        idxs : bool
            If True will return the trajectory identifier tuple
            (run_idx, traj_idx) along with other return values.

        return_results : bool
            If True will return the results of the mapping. If not
            using the 'save_to_hdf5' option, be sure to use this or
            results will be lost.

        Returns
        -------

        traj_id_tuples : list of tuple of int, if 'idxs' option is True
            A list of the tuple identifiers for each trajectory result.

        results : list of arraylike, if 'return_results' option is True
            A list of arraylike feature vectors for each trajectory.

        """

        if save_to_hdf5 is not None:
            assert self.mode in ['w', 'w-', 'x', 'r+', 'c', 'c-'],\
                "File must be in a write mode"
            assert isinstance(save_to_hdf5, str),\
                "`save_to_hdf5` should be the field name to save the data in the `observables`"\
                " group in each trajectory"

            # the field name comes from this kwarg if it satisfies the
            # string condition above
            field_name = save_to_hdf5

        # calculate the results and accumulate them here
        results = []

        # and the indices of the results
        result_idxs = []


        # map over the trajectories and apply the function and save
        # the results
        for result in self.traj_fields_map(func, fields, args,
                                           map_func=map_func, traj_sel=traj_sel, idxs=True):

            idx_tup, obs_features = result

            results.append(obs_features)
            result_idxs.append(idx_tup)

        # we want to separate writing and computation so we can do it
        # in parallel without having multiple writers. So if we are
        # writing directly to the HDF5 we add the results to it.

        # if we are saving this to the trajectories observables add it as a dataset
        if save_to_hdf5:


            # reshape the results to be in the observable shape:
            observable = [[] for run_idx in self.run_idxs]

            for result_idx, traj_results in zip(result_idxs, results):

                run_idx, traj_idx = result_idx

                observable[run_idx].append(traj_results)

            self.add_observable(
                field_name,
                observable,
                sparse_idxs=None,
           )

        if return_results:
            if idxs:
                return result_idxs, results
            else:
                return results
    ## Trajectory Getters

    def get_traj_field(self, run_idx, traj_idx, field_path, frames=None, masked=True):
        """Returns a numpy array for the given trajectory field.

        You can control how sparse fields are returned using the
        `masked` option. When True (default) a masked numpy array will
        be returned such that you can get which cycles it is from,
        when False an unmasked array of the data will be returned
        which has no cycle information.

        Parameters
        ----------
        run_idx : int
        traj_idx : int
        field_path : str
            Name of the trajectory field to get

        frames : None or list of int
            If not None, a list of the frame indices of the trajectory
            to return values for.

        masked : bool
            If true will return sparse field values as masked arrays,
            otherwise just returns the compacted data.

        Returns
        -------
        field_data : arraylike
            The data for the trajectory field.

        """

        traj_path = '{}/{}/{}/{}'.format(RUNS, run_idx, TRAJECTORIES, traj_idx)

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

    def get_trace_fields(self, frame_tups, fields,
                         same_order=True):
        """Get trajectory field data for the frames specified by the trace.

        Parameters
        ----------
        frame_tups : list of tuple of int
            The trace values. Each tuple is of the form
            (run_idx, traj_idx, frame_idx).

        fields : list of str
            The names of the fields to get for each frame.

        same_order : bool
           (Default = True)
           If True will ensure that the results will be sorted exactly
           as the order of the frame_tups were. If False will return
           them in an arbitrary implementation determined order that
           should be more efficient.

        Returns
        -------
        trace_fields : dict of str : arraylike
            Mapping of the field names to the array of feature vectors
            for the trace.

        """

        # TODO optimize by doing reads in chunks
        opt_flag = True

        if opt_flag:

            def argsort(seq):
                return sorted(range(len(seq)), key=seq.__getitem__)

            def apply_argsorted(shuffled_seq, sorted_idxs):
                return [shuffled_seq[i] for i in sorted_idxs]

            # first sort the frame_tups so we can chunk them up by
            # (run, traj) to get more efficient reads since these are
            # chunked by these datasets.

            # we do an argsort here so that we can map fields back to
            # the order they came in (if requested)
            sorted_idxs = argsort(frame_tups)

            # then sort them as we will iterate through them
            sorted_frame_tups = apply_argsorted(frame_tups, sorted_idxs)

            # generate the chunks by (run, traj)
            read_chunks = defaultdict(list)
            for run_idx, traj_idx, frame_idx in sorted_frame_tups:
                read_chunks[(run_idx, traj_idx)].append(frame_idx)

            # go through each chunk and read data for each field
            frame_fields = {}
            for field in fields:

                # for each field collect the chunks
                field_chunks = []

                for chunk_key, frames in read_chunks.items():
                    run_idx, traj_idx = chunk_key

                    frames_field = self.get_traj_field(run_idx, traj_idx, field,
                                                       frames=frames)

                    field_chunks.append(frames_field)

                # then aggregate them
                field_unsorted = np.concatenate(field_chunks)
                del field_chunks; gc.collect()

                # if we want them sorted sort them back to the
                # original (unsorted) order, otherwise just return
                # them
                if same_order:
                    frame_fields[field] = field_unsorted[sorted_idxs]
                else:
                    frame_fields[field] = field_unsorted

                del field_unsorted; gc.collect()


        else:

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
        """Get trajectory field data for the frames specified by the trace
        within a single run.

        Parameters
        ----------
        run_idx : int

        frame_tups : list of tuple of int
            The trace values. Each tuple is of the form
            (traj_idx, frame_idx).

        fields : list of str
            The names of the fields to get for each frame.

        Returns
        -------
        trace_fields : dict of str : arraylike
            Mapping of the field names to the array of feature vectors
            for the trace.

        """
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


    def get_contig_trace_fields(self, contig_trace, fields):
        """Get field data for all trajectories of a contig for the frames
        specified by the contig trace.

        Parameters
        ----------
        contig_trace : list of tuple of int
            The trace values. Each tuple is of the form
            (run_idx, frame_idx).

        fields : list of str
            The names of the fields to get for each cycle.

        Returns
        -------
        contig_fields : dict of str : arraylike
                             of shape (n_cycles, n_trajs, field_feature_shape[0],...)
            Mapping of the field names to the array of feature vectors
            for contig trace.

        """

        # to be efficient we want to group our grabbing of fields by run

        # so we group them by run
        runs_frames = defaultdict(list)
        # and we get the runs in the order to fetch them
        run_idxs = []
        for run_idx, cycle_idx in contig_trace:
            runs_frames[run_idx].append(cycle_idx)

            if not run_idx in run_idxs:
                run_idxs.append(run_idx)


        # (there must be the same number of trajectories in each run)
        n_trajs_test = self.num_run_trajs(run_idxs[0])
        assert all([True if n_trajs_test == self.num_run_trajs(run_idx) else False
                    for run_idx in run_idxs])

        # then using this we go run by run and get all the
        # trajectories
        field_values = {}
        for field in fields:

            # we gather trajectories in "bundles" (think sticks
            # strapped together) and each bundle represents a run, we
            # will concatenate the ends of the bundles together to get
            # the full array at the end
            bundles = []
            for run_idx in run_idxs:

                run_bundle = []
                for traj_idx in self.run_traj_idxs(run_idx):

                    # get the values for this (field, run, trajectory)
                    traj_field_vals = self.get_traj_field(run_idx, traj_idx, field,
                                                          frames=runs_frames[run_idx],
                                                          masked=True)

                    run_bundle.append(traj_field_vals)

                # convert this "bundle" of trajectory values (think
                # sticks side by side) into an array
                run_bundle = np.array(run_bundle)
                bundles.append(run_bundle)

            # stick the bundles together end to end to make the value
            # for this field , the first dimension currently is the
            # trajectory_index, but we want to make the cycles the
            # first dimension. So we stack them along that axis then
            # transpose the first two axes (not the rest of them which
            # should stay the same). Pardon the log terminology, but I
            # don't know a name for a bunch of bundles taped together.
            field_log = np.hstack(tuple(bundles))
            field_log = np.swapaxes(field_log, 0, 1)

            field_values[field] = field_log


        return field_values

    def iter_trajs_fields(self, fields, idxs=False, traj_sel=None):
        """Generator for iterating over fields trajectories in a file.

        Parameters
        ----------
        fields : list of str
            Names of the trajectory fields you want to yield.

        idxs : bool
            If True will also return the tuple identifier of the
            trajectory the field data is from.

        traj_sel : list of tuple of int
            If not None, a list of trajectory identifiers to restrict
            iteration over.

        Yields
        ------
        traj_identifier : tuple of int if 'idxs' option is True
            Tuple identifying the trajectory the data belongs to
            (run_idx, traj_idx).

        fields_data : dict of str : arraylike
            Mapping of the field name to the array of feature vectors
            of that field for this trajectory.

        """

        for idx_tup, traj in self.iter_trajs(idxs=True, traj_sel=traj_sel):
            run_idx, traj_idx = idx_tup

            dsets = {}

            # DEBUG if we ask for debug prints send in the run and
            # traj index so the function can print this out TESTING if
            # this causes no problems (it doesn't seem like it would
            # from the code this will be removed permanently)

            # dsets['run_idx'] = run_idx
            # dsets[TRAJ_IDX] = traj_idx

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

    def traj_fields_map(self, func, fields, args,
                        map_func=map, idxs=False, traj_sel=None):
        """Function for mapping work onto field of trajectories.

        Parameters
        ----------
        func : callable
            The function to apply to the trajectory fields (by
            cycle). Must accept a dictionary mapping string trajectory
            field names to a feature vector for that cycle and return
            an arraylike. May accept other positional arguments as well.

        fields : list of str
            A list of trajectory field names to pass to the mapped function.

        args : None or or tuple
            A single tuple of arguments which will be
            passed to the mapped function for every evaluation.

        map_func : callable
            The mapping function. The implementation of how to map the
            computation function over the data. Default is the python
            builtin `map` function. Can be a parallel implementation
            for example.

        traj_sel : list of tuple, optional
            If not None, a list of trajectory identifier tuple
            (run_idx, traj_idx) to restrict the computation to.

        idxs : bool
            If True will return the trajectory identifier tuple
            (run_idx, traj_idx) along with other return values.

        Returns
        -------
        traj_id_tuples : list of tuple of int, if 'idxs' option is True
            A list of the tuple identifiers for each trajectory result.

        results : list of arraylike
            A list of arraylike feature vectors for each trajectory.

        """

        # check the args and kwargs to see if they need expanded for
        # mapping inputs
        #first go through each run and get the number of cycles
        n_cycles = 0
        for run_idx in self.run_idxs:
            n_cycles += self.num_run_cycles(run_idx)

        mapped_args = []
        for arg in args:
            # make a generator out of it to map as inputs
            mapped_arg = (arg for i in range(n_cycles))
            mapped_args.append(mapped_arg)

        # make a generator for the arguments to pass to the function
        # from the mapper, for the extra arguments we just have an
        # endless generator
        map_args = (self.iter_trajs_fields(fields, traj_sel=traj_sel, idxs=False),
                    *(it.repeat(arg) for arg in args))

        results = map_func(func, *map_args)

        if idxs:
            if traj_sel is None:
                traj_sel = self.run_traj_idx_tuples()
            return zip(traj_sel, results)
        else:
            return results

    def to_mdtraj(self, run_idx, traj_idx, frames=None, alt_rep=None):
        """Convert a trajectory to an mdtraj Trajectory object.

        Works if the right trajectory fields are defined. Minimally
        this is a representation, including the 'positions' field or
        an 'alt_rep' subfield.

        Will also set the unitcell lengths and angle if the
        'box_vectors' field is present.

        Will also set the time for the frames if the 'time' field is
        present, although this is likely not useful since walker
        segments have the time reset.

        Parameters
        ----------
        run_idx : int
        traj_idx : int

        frames : None or list of int
            If not None, a list of the frames to include.

        alt_rep : str
            If not None, an 'alt_reps' subfield name to use for
            positions instead of the 'positions' field.

        Returns
        -------
        traj : mdtraj.Trajectory

        """

        traj_grp = self.traj(run_idx, traj_idx)

        # the default for alt_rep is the main rep
        if alt_rep is None:
            rep_key = POSITIONS
            rep_path = rep_key
        else:
            rep_key = alt_rep
            rep_path = '{}/{}'.format(ALT_REPS, alt_rep)

        topology = self.get_mdtraj_topology(alt_rep=rep_key)


        # get the frames if they are not given
        if frames is None:
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
        """Generate an mdtraj Trajectory from a trace of frames from the runs.

        Uses the default fields for positions (unless an alternate
        representation is specified) and box vectors which are assumed
        to be present in the trajectory fields.

        The time value for the mdtraj trajectory is set to the cycle
        indices for each trace frame.

        This is useful for converting WepyHDF5 data to common
        molecular dynamics data formats accessible through the mdtraj
        library.

        Parameters
        ----------
        trace : list of tuple of int
            The trace values. Each tuple is of the form
            (run_idx, traj_idx, frame_idx).

        alt_rep : None or str
            If None uses default 'positions' representation otherwise
            chooses the representation from the 'alt_reps' compound field.

        Returns
        -------
        traj : mdtraj.Trajectory

        """

        rep_path = self._choose_rep_path(alt_rep)

        trace_fields = self.get_trace_fields(trace, [rep_path, BOX_VECTORS])

        return self.traj_fields_to_mdtraj(trace_fields, alt_rep=alt_rep)

    def run_trace_to_mdtraj(self, run_idx, trace, alt_rep=None):
        """Generate an mdtraj Trajectory from a trace of frames from the runs.

        Uses the default fields for positions (unless an alternate
        representation is specified) and box vectors which are assumed
        to be present in the trajectory fields.

        The time value for the mdtraj trajectory is set to the cycle
        indices for each trace frame.

        This is useful for converting WepyHDF5 data to common
        molecular dynamics data formats accessible through the mdtraj
        library.

        Parameters
        ----------
        run_idx : int
            The run the trace is over.

        run_trace : list of tuple of int
            The trace values. Each tuple is of the form
            (traj_idx, frame_idx).

        alt_rep : None or str
            If None uses default 'positions' representation otherwise
            chooses the representation from the 'alt_reps' compound field.

        Returns
        -------
        traj : mdtraj.Trajectory

        """

        rep_path = self._choose_rep_path(alt_rep)

        trace_fields = self.get_run_trace_fields(run_idx, trace, [rep_path, BOX_VECTORS])

        return self.traj_fields_to_mdtraj(trace_fields, alt_rep=alt_rep)

    def _choose_rep_path(self, alt_rep):
        """Given a positions specification string, gets the field name/path
        for it.

        Parameters
        ----------

        alt_rep : str
            The short name (non relative path) for a representation of
            the positions.

        Returns
        -------

        rep_path : str
            The relative field path to that representation.

        E.g.:

        If you give it 'positions' or None it will simply return
        'positions', however if you ask for 'all_atoms' it will return
        'alt_reps/all_atoms'.

        """

        # the default for alt_rep is the main rep
        if alt_rep == POSITIONS:
            rep_path = POSITIONS
        elif alt_rep is None:
            rep_key = POSITIONS
            rep_path = rep_key
        # if it is already a path we don't add more to it and just
        # return it.
        elif len(alt_rep.split('/')) > 1:
            if len(alt_rep.split('/')) > 2:
                raise ValueError("unrecognized alt_rep spec")
            elif alt_rep.split('/')[0] != ALT_REPS:
                raise ValueError("unrecognized alt_rep spec")
            else:
                rep_path = alt_rep
        else:
            rep_key = alt_rep
            rep_path = '{}/{}'.format(ALT_REPS, alt_rep)

        return rep_path


    def traj_fields_to_mdtraj(self, traj_fields, alt_rep=POSITIONS):
        """Create an mdtraj.Trajectory from a traj_fields dictionary.

        Parameters
        ----------

        traj_fields : dict of str : arraylike
            Dictionary of the traj fields to their values

        alt_reps : str
            The base alt rep name for the positions representation to
            use for the topology, should have the corresponding
            alt_rep field in the traj_fields

        Returns
        -------

        traj : mdtraj.Trajectory object

        This is mainly a convenience function to retrieve the correct
        topology for the positions which will be passed to the generic
        `traj_fields_to_mdtraj` function.

        """

        rep_path = self._choose_rep_path(alt_rep)

        json_topology = self.get_topology(alt_rep=rep_path)

        return traj_fields_to_mdtraj(traj_fields, json_topology, rep_key=rep_path)



    def copy_run_slice(self, run_idx, target_file_path, target_grp_path,
                       run_slice=None, mode='x'):
        """Copy this run to another HDF5 file (target_file_path) at the group
        (target_grp_path)"""

        assert mode in ['w', 'w-', 'x', 'r+'], "must be opened in write mode"

        if run_slice is not None:
            assert run_slice[1] >= run_slice[0], "Must be a contiguous slice"

            # get a list of the frames to use
            slice_frames = list(range(*run_slice))


        # we manually construct an HDF5 wrapper and copy the groups over
        new_h5 = h5py.File(target_file_path, mode=mode, libver=H5PY_LIBVER)

        # flush the datasets buffers
        self.h5.flush()
        new_h5.flush()

        # get the run group we are interested in
        run_grp = self.run(run_idx)

        # slice the datasets in the run and set them in the new file
        if run_slice is not None:

            # initialize the group for the run
            new_run_grp = new_h5.require_group(target_grp_path)


            # copy the init walkers group
            self.h5.copy(run_grp[INIT_WALKERS], new_run_grp,
                         name=INIT_WALKERS)

            # copy the decision group
            self.h5.copy(run_grp[DECISION], new_run_grp,
                         name=DECISION)


            # create the trajectories group
            new_trajs_grp = new_run_grp.require_group(TRAJECTORIES)

            # slice the trajectories and copy them
            for traj_idx in run_grp[TRAJECTORIES]:

                traj_grp = run_grp[TRAJECTORIES][traj_idx]

                traj_id = "{}/{}".format(TRAJECTORIES, traj_idx)

                new_traj_grp = new_trajs_grp.require_group(str(traj_idx))

                for field_name in _iter_field_paths(run_grp[traj_id]):
                    field_path = "{}/{}".format(traj_id, field_name)

                    data = self.get_traj_field(run_idx, traj_idx, field_name,
                                               frames=slice_frames)

                    # if it is a sparse field we need to create the
                    # dataset differently
                    if field_name in self.sparse_fields:

                        # create a group for the field
                        new_field_grp = new_traj_grp.require_group(field_name)

                        # slice the _sparse_idxs from the original
                        # dataset that are between the slice
                        cycle_idxs = self.traj(run_idx, traj_idx)[field_name]['_sparse_idxs'][:]

                        sparse_idx_idxs = np.argwhere(np.logical_and(
                            cycle_idxs[:] >= run_slice[0], cycle_idxs[:] < run_slice[1]
                        )).flatten().tolist()

                        # the cycle idxs there is data for
                        sliced_cycle_idxs = cycle_idxs[sparse_idx_idxs]

                        # get the data for these cycles
                        field_data = data[sliced_cycle_idxs]

                        # get the information on compression,
                        # chunking, and filters and use it when we set
                        # the new data
                        field_data_dset = traj_grp[field_name]['data']
                        data_dset_kwargs = {
                            'chunks' : field_data_dset.chunks,
                            'compression' : field_data_dset.compression,
                            'compression_opts' : field_data_dset.compression_opts,
                            'shuffle' : field_data_dset.shuffle,
                            'fletcher32' : field_data_dset.fletcher32,
                        }

                        # and for the sparse idxs although it is probably overkill
                        field_idxs_dset = traj_grp[field_name]['_sparse_idxs']
                        idxs_dset_kwargs = {
                            'chunks' : field_idxs_dset.chunks,
                            'compression' : field_idxs_dset.compression,
                            'compression_opts' : field_idxs_dset.compression_opts,
                            'shuffle' : field_idxs_dset.shuffle,
                            'fletcher32' : field_idxs_dset.fletcher32,
                        }

                        # then create the datasets
                        new_field_grp.create_dataset('_sparse_idxs',
                                                     data=sliced_cycle_idxs,
                                                     **idxs_dset_kwargs)
                        new_field_grp.create_dataset('data',
                                                     data=field_data,
                                                     **data_dset_kwargs)

                    else:

                        # get the information on compression,
                        # chunking, and filters and use it when we set
                        # the new data
                        field_dset = traj_grp[field_name]

                        # since we are slicing we want to make sure
                        # that the chunks are smaller than the
                        # slices. Normally chunks are (1, ...) for a
                        # field, but may not be for observables
                        # (perhaps they should but thats for another issue)
                        chunks = (1, *field_dset.chunks[1:])

                        dset_kwargs = {
                            'chunks' : chunks,
                            'compression' : field_dset.compression,
                            'compression_opts' : field_dset.compression_opts,
                            'shuffle' : field_dset.shuffle,
                            'fletcher32' : field_dset.fletcher32,
                        }

                        # require the dataset first to automatically build
                        # subpaths for compound fields if necessary
                        dset = new_traj_grp.require_dataset(field_name,
                                                            data.shape, data.dtype,
                                                            **dset_kwargs)

                        # then set the data depending on whether it is
                        # sparse or not
                        dset[:] = data

            # then do it for the records
            for rec_grp_name, rec_fields in self.record_fields.items():

                rec_grp = run_grp[rec_grp_name]

                # if this is a contiguous record we can skip the cycle
                # indices to record indices conversion that is
                # necessary for sporadic records
                if self._is_sporadic_records(rec_grp_name):

                    cycle_idxs = rec_grp[CYCLE_IDXS][:]

                    # get dataset info
                    cycle_idxs_dset = rec_grp[CYCLE_IDXS]

                    # we use autochunk, because I can't figure out how
                    # the chunks are set and I can't reuse them
                    idxs_dset_kwargs = {
                        'chunks' : True,
                        # 'chunks' : cycle_idxs_dset.chunks,
                        'compression' : cycle_idxs_dset.compression,
                        'compression_opts' : cycle_idxs_dset.compression_opts,
                        'shuffle' : cycle_idxs_dset.shuffle,
                        'fletcher32' : cycle_idxs_dset.fletcher32,
                    }

                    # get the indices of the records we are interested in
                    record_idxs = np.argwhere(np.logical_and(
                        cycle_idxs >= run_slice[0], cycle_idxs < run_slice[1]
                    )).flatten().tolist()

                    # set the cycle indices in the new run group
                    new_recgrp_cycle_idxs_path = '{}/{}/_cycle_idxs'.format(target_grp_path,
                                                                            rec_grp_name)
                    cycle_data = cycle_idxs[record_idxs]

                    cycle_dset = new_h5.require_dataset(new_recgrp_cycle_idxs_path,
                                                        cycle_data.shape, cycle_data.dtype,
                                                        **idxs_dset_kwargs)
                    cycle_dset[:] = cycle_data

                # if contiguous just set the record indices as the
                # range between the slice
                else:
                    record_idxs = list(range(run_slice[0], run_slice[1]))

                # then for each rec_field slice those and set them in the new file
                for rec_field in rec_fields:

                    field_dset = rec_grp[rec_field]

                    # get dataset info
                    field_dset_kwargs = {
                        'chunks' : True,
                        # 'chunks' : field_dset.chunks,
                        'compression' : field_dset.compression,
                        'compression_opts' : field_dset.compression_opts,
                        'shuffle' : field_dset.shuffle,
                        'fletcher32' : field_dset.fletcher32,
                    }


                    rec_field_path = "{}/{}".format(rec_grp_name, rec_field)
                    new_recfield_grp_path = '{}/{}'.format(target_grp_path, rec_field_path)

                    # if it is a variable length dtype make the dtype
                    # that for the dataset and we also slice the
                    # dataset differently
                    vlen_type = h5py.check_dtype(vlen=field_dset.dtype)
                    if vlen_type is not None:

                        dtype = h5py.special_dtype(vlen=vlen_type)

                    else:
                        dtype = field_dset.dtype



                    # if there are no records don't attempt to add them
                    # get the shape
                    shape = (len(record_idxs), *field_dset.shape[1:])

                    new_field_dset = new_h5.require_dataset(new_recfield_grp_path,
                                                            shape, dtype,
                                                            **field_dset_kwargs)

                    # if there aren't records just don't do anything,
                    # and if there are get them and add them
                    if len(record_idxs) > 0:
                        rec_data = field_dset[record_idxs]

                        # if it is a variable length data type we have
                        # to do it 1 by 1
                        if vlen_type is not None:
                            for i, vlen_rec in enumerate(rec_data):
                                new_field_dset[i] = rec_data[i]
                        # otherwise just set it all at once
                        else:
                            new_field_dset[:] = rec_data

        # just copy the whole thing over, since this will probably be
        # more efficient
        else:

            # split off the last bit of the target path, for copying we
            # need it's parent group but not it to exist
            target_grp_path_basename = target_grp_path.split('/')[-1]
            target_grp_path_prefix = target_grp_path.split('/')[:-1]

            new_run_prefix_grp = self.h5.require_group(target_grp_path_prefix)

            # copy the whole thing
            self.h5.copy(run_grp, new_run_prefix_grp,
                         name=target_grp_path_basename)

        # flush the datasets buffers
        self.h5.flush()
        new_h5.flush()

        return new_h5
