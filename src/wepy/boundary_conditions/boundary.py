"""Abstract base class and null boundary condition class for
conveniently making compliant boundary condition classes for use in
wepy.

"""

import sys
import logging
from copy import deepcopy
from collections import defaultdict
import random

import numpy as np

from wepy.walker import Walker

class BoundaryConditions(object):
    """Abstract base class for conveniently making compliant boundary condition classes.

    Includes empty record group definitions and useful getters for those.

    """


    # records of boundary condition changes (sporadic)
    BC_FIELDS = ()
    """String names of fields produced in this record group.

    Boundary condition (BC) records are typically used to report on
    changes to the state of the BC object.

    Notes
    -----

    These fields are not critical to the proper functioning of the
    rest of the wepy framework and can be modified freely.

    However, reporters specific to this boundary condition probably
    will make use of these records.

    """

    BC_SHAPES = ()
    """Numpy-style shapes of all fields produced in records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A tuple of ints that specify the shape of the field element
       array.

    B. Ellipsis, indicating that the field is variable length and
       limited to being a rank one array (e.g. (3,) or (1,)).

    C. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    Note that the shapes must be tuple and not simple integers for rank-1
    arrays.

    Option B will result in the special h5py datatype 'vlen' and
    should not be used for large datasets for efficiency reasons.

    """

    BC_DTYPES = ()
    """Specifies the numpy dtypes to be used for records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A `numpy.dtype` object.

    D. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    """

    BC_RECORD_FIELDS = ()
    """Optional, names of fields to be selected for truncated
    representation of the record group.

    These entries should be strings that are previously contained in
    the 'FIELDS' class constant.

    While strictly no constraints on to which fields can be added here
    you should only choose those fields whose features could fit into
    a plaintext csv or similar format.

    """

    # warping (sporadic)
    WARPING_FIELDS = ('walker_idx', 'target_idx', 'weight')
    """String names of fields produced in this record group.

    Warping records are typically used to report whenever a walker
    satisfied the boundary conditions and was warped and had its
    state changed.

    Warnings
    --------

    Be careful when modifying these fields as they may be integrated
    with other wepy framework features. Namely recognition of
    discontinuous warping events for making contiguous trajectories
    from cloning and merging lineages.

    The behavior of whether or not a warping event is discontinuous is
    given by a `BoundaryCondition` class's `warping_discontinuity`
    which likely depends on the existence of particular fields.

    """

    WARPING_SHAPES = ((1,), (1,), (1,))
    """Numpy-style shapes of all fields produced in records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A tuple of ints that specify the shape of the field element
       array.

    B. Ellipsis, indicating that the field is variable length and
       limited to being a rank one array (e.g. (3,) or (1,)).

    C. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    Note that the shapes must be tuple and not simple integers for rank-1
    arrays.

    Option B will result in the special h5py datatype 'vlen' and
    should not be used for large datasets for efficiency reasons.

    """

    WARPING_DTYPES = (np.int, np.int, np.float)
    """Specifies the numpy dtypes to be used for records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A `numpy.dtype` object.

    D. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    """

    WARPING_RECORD_FIELDS = ('walker_idx', 'target_idx', 'weight')
    """Optional, names of fields to be selected for truncated
    representation of the record group.

    These entries should be strings that are previously contained in
    the 'FIELDS' class constant.

    While strictly no constraints on to which fields can be added here
    you should only choose those fields whose features could fit into
    a plaintext csv or similar format.

    """

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ()
    """String names of fields produced in this record group.

    Progress records are typically used to report on measures of
    walkers at each cycle.

    Notes
    -----

    These fields are not critical to the proper functioning of the
    rest of the wepy framework and can be modified freely.

    However, reporters specific to this boundary condition probably
    will make use of these records.

    """

    PROGRESS_SHAPES = ()
    """Numpy-style shapes of all fields produced in records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A tuple of ints that specify the shape of the field element
       array.

    B. Ellipsis, indicating that the field is variable length and
       limited to being a rank one array (e.g. (3,) or (1,)).

    C. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    Note that the shapes must be tuple and not simple integers for rank-1
    arrays.

    Option B will result in the special h5py datatype 'vlen' and
    should not be used for large datasets for efficiency reasons.

    """

    PROGRESS_DTYPES = ()
    """Specifies the numpy dtypes to be used for records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A `numpy.dtype` object.

    D. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    """

    PROGRESS_RECORD_FIELDS = ()
    """Optional, names of fields to be selected for truncated
    representation of the record group.

    These entries should be strings that are previously contained in
    the 'FIELDS' class constant.

    While strictly no constraints on to which fields can be added here
    you should only choose those fields whose features could fit into
    a plaintext csv or similar format.

    """

    def __init__(self, **kwargs):
        """Null constructor accepts and ignores any key word arguments. """

        pass

    def bc_field_names(self):
        """Access the class level FIELDS constant for this record group."""
        return self.BC_FIELDS

    def bc_field_shapes(self):
        """Access the class level SHAPES constant for this record group."""
        return self.BC_SHAPES

    def bc_field_dtypes(self):
        """Access the class level DTYPES constant for this record group."""
        return self.BC_DTYPES

    def bc_fields(self):
        """Returns a list of zipped field specs.

        Returns
        -------

        record_specs : list of tuple
            A list of the specs for each field, a spec is a tuple of
            type (field_name, shape_spec, dtype_spec)
        """
        return list(zip(self.bc_field_names(),
                   self.bc_field_shapes(),
                   self.bc_field_dtypes()))

    def bc_record_field_names(self):
        """Access the class level RECORD_FIELDS constant for this record group."""
        return self.BC_RECORD_FIELDS

    def warping_field_names(self):
        """Access the class level FIELDS constant for this record group."""
        return self.WARPING_FIELDS

    def warping_field_shapes(self):
        """Access the class level SHAPES constant for this record group."""
        return self.WARPING_SHAPES

    def warping_field_dtypes(self):
        """Access the class level DTYPES constant for this record group."""
        return self.WARPING_DTYPES

    def warping_fields(self):
        """Returns a list of zipped field specs.

        Returns
        -------

        record_specs : list of tuple
            A list of the specs for each field, a spec is a tuple of
            type (field_name, shape_spec, dtype_spec)
        """
        return list(zip(self.warping_field_names(),
                   self.warping_field_shapes(),
                   self.warping_field_dtypes()))

    def warping_record_field_names(self):
        """Access the class level RECORD_FIELDS constant for this record group."""
        return self.WARPING_RECORD_FIELDS

    def progress_field_names(self):
        """Access the class level FIELDS constant for this record group."""
        return self.PROGRESS_FIELDS

    def progress_field_shapes(self):
        """Access the class level SHAPES constant for this record group."""
        return self.PROGRESS_SHAPES

    def progress_field_dtypes(self):
        """Access the class level DTYPES constant for this record group."""
        return self.PROGRESS_DTYPES

    def progress_fields(self):
        """Returns a list of zipped field specs.

        Returns
        -------

        record_specs : list of tuple
            A list of the specs for each field, a spec is a tuple of
            type (field_name, shape_spec, dtype_spec)
        """
        return list(zip(self.progress_field_names(),
                   self.progress_field_shapes(),
                   self.progress_field_dtypes()))

    def progress_record_field_names(self):
        """Access the class level RECORD_FIELDS constant for this record group."""
        return self.PROGRESS_RECORD_FIELDS

    def warp_walkers(self, walkers, cycle):
        """Apply boundary condition logic to walkers.

        If walkers satisfy the boundary conditions then they will be
        'warped' and have a corresponding state change take
        place. Each event recorded is returned as a single
        dictionary-style record in 'warp_data' list. These records
        correspond to the 'WARPING' record group.

        Additional data calculated on walkers may be returned in the
        single 'progress_data' dictionary-style record, which
        corresponds to the 'PROGRESS' record group.

        Any changes to the internal state of the boundary condition
        object (e.g. modification of parameters) should be recorded in
        at least one dictionary-style record in the 'bc_data'
        list. This corresponds to the 'BC' record group.

        Parameters
        ----------
        walkers : list of walkers
            A list of objects implementing the Walker interface

        cycle : int
            The index of the cycle this is for. Used to generate proper records.

        Returns
        -------
        new_walkers : list of walkers
            A list of objects implementing the Walker interface, that have had
            boundary condition logic applied.

        warp_data : list of dict of str : value
            A list of dictionary style records for each warping event that occured.

        bc_data : list of dict of str : value
            A list of dictionary style records for each boundary condition state change
            event record that occured.

        progress_data : dict of str : list of value
           Dictionary style progress records. The values should be lists
           corresponding to each walker.

        """

        raise NotImplementedError

    @classmethod
    def warping_discontinuity(cls, warping_record):
        """Given a warping record returns either True for a discontiuity
        occured or False if a discontinuity did not occur.

        Parameters
        ----------
        warping_record : tuple
            A tuple record of type 'WARPING'

        Returns
        -------
        is_discontinuous : bool
            True if discontinuous warping record False if continuous.

        """

        raise NotImplementedError



class NoBC(BoundaryConditions):
    """Boundary conditions class that does nothing.

    You may use this class as a stub in order to have an boundary
    condition class. However, this is not necessary since boundary
    conditions are optional in the sim_manager anyhow.
    """

    def warp_walkers(self, walkers, cycle):
        """Apply boundary condition logic to walkers, of which there is none.

        Simply returns all walkers provided with empty records data
        since there is nothing to do.

        Parameters
        ----------
        walkers : list of walkers
            A list of objects implementing the Walker interface

        cycle : int
            The index of the cycle this is for. Used to generate proper records.

        Returns
        -------
        new_walkers : list of walkers
            A list of objects implementing the Walker interface, that have had
            boundary condition logic applied.

        warp_data : list of dict of str : value
            A list of dictionary style records for each warping event that occured.

        bc_data : list of dict of str : value
            A list of dictionary style records for each boundary condition state change
            event record that occured.

        progress_data : dict of str : list of value
           Dictionary style progress records. The values should be lists
           corresponding to each walker.

        """

        warp_data = []
        bc_data = []
        progress_data = {}

        return walkers, warp_data, bc_data, progress_data

    @classmethod
    def warping_discontinuity(cls, warping_record):
        # documented in superclass

        # always return false
        return False


class RandomBC(BoundaryConditions):
    """Boundary conditions that randomly warps both continuously and
    discontinuously.

    Can be used with any system as it won't actually mutate states.

    """


    # records of boundary condition changes (sporadic)
    BC_FIELDS = ('ping',)
    BC_SHAPES = ((1,),)
    BC_DTYPES = (np.int,)

    BC_RECORD_FIELDS = ('ping',)

    # warping fields are directly inherited

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ('weight',)
    PROGRESS_SHAPES = (Ellipsis,)
    PROGRESS_DTYPES = (np.float,)

    PROGRESS_RECORD_FIELDS = ('weight',)

    DISCONTINUITY_TARGET_IDXS = (0,)

    def warp_walkers(self, walkers, cycle):

        ## warping walkers

        # just return the same walkers
        new_walkers = deepcopy(walkers)

        ## warping data

        warp_data = []
        # generate warping data: 50% of the time generate a warping
        # event, 25% is discontinuous (target 0), and 25% is
        # continuous (target 1)
        for walker_idx, walker in enumerate(walkers):

            # warping event?
            if random.random() >= 0.5:

                # discontinuous?
                if random.random() >= 0.5:
                    warp_record = {
                        'walker_idx' : np.array([walker_idx]),
                        'target_idx' : np.array([0]),
                        'weight' : np.array([walker.weight]),
                    }

                    warp_data.append(warp_record)

                # continuous
                else:
                    warp_record = {
                        'walker_idx' : np.array([walker_idx]),
                        'target_idx' : np.array([1]),
                        'weight' : np.array([walker.weight]),
                    }

                    warp_data.append(warp_record)

        ## BC data
        bc_data = []
        # choose whether to generate a bc record
        if random.random() >= 0.5:
            bc_data.append({'ping' : np.array([1])})

        ## Progress data

        # just set the walker progress to be its weight so there is a
        # number there
        progress_data = {'weight' :
                         [walker.weight for walker in walkers]
        }


        return new_walkers, warp_data, bc_data, progress_data

