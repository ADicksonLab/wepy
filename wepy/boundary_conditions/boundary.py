"""Abstract base class and null boundary condition class for
conveniently making compliant boundary condition classes for use in
wepy.

"""

import sys
import logging

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
    WARPING_FIELDS = ()
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

    WARPING_SHAPES = ()
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

    WARPING_DTYPES = ()
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

    WARPING_RECORD_FIELDS = ()
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

    def progress(self, walker):
        """Checks if a walker is in a boundary and returns which boundary it is in

        Parameters
        ----------
        walker :
            

        Returns
        -------

        """
        raise NotImplementedError

    def warp_walkers(self, walkers):
        """Checks walkers for membership in boundaries and processes them
        according to the rules of the boundary.

        Parameters
        ----------
        walkers :
            

        Returns
        -------

        """
        raise NotImplementedError



class NoBC(BoundaryConditions):
    """ """

    def check_boundaries(self, walker):
        """

        Parameters
        ----------
        walker :
            

        Returns
        -------

        """
        return False, {}

    def warp_walkers(self, walkers):
        """

        Parameters
        ----------
        walkers :
            

        Returns
        -------

        """
        # in order the walkers after applying warps:
        # warping, bc, progress
        warp_data = {}
        bc_data = {}
        progress_data = {}

        return walkers, warp_data, bc_data, progress_data
