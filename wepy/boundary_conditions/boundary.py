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
    """String names of fields produced in 'bc' records."""

    BC_SHAPES = ()
    """Numpy-style shapes of all fields produced in 'bc' records.

    Each entry should either be:

    A. A tuple of ints that specify the shape of the field element
       array.

    B. Ellipsis, indicating that the field is variable length and
       limited to being a rank one array (e.g. (3,) or (1,)).

    C. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    """

    BC_DTYPES = ()
    """Numpy-style """

    BC_RECORD_FIELDS = ()

    # warping (sporadic)
    WARPING_FIELDS = ()
    """String names of fields produced in 'warping' records."""

    WARPING_SHAPES = ()
    WARPING_DTYPES = ()

    WARPING_RECORD_FIELDS = ()

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ()
    """String names of fields produced in 'progress' records."""

    PROGRESS_SHAPES = ()
    PROGRESS_DTYPES = ()

    PROGRESS_RECORD_FIELDS = ()

    def __init__(self, **kwargs):

        pass

    def bc_field_names(self):
        """ """
        return self.BC_FIELDS

    def bc_field_shapes(self):
        """ """
        return self.BC_SHAPES

    def bc_field_dtypes(self):
        """ """
        return self.BC_DTYPES

    def bc_fields(self):
        """ """
        return list(zip(self.bc_field_names(),
                   self.bc_field_shapes(),
                   self.bc_field_dtypes()))

    def bc_record_field_names(self):
        """ """
        return self.BC_RECORD_FIELDS

    def warping_field_names(self):
        """ """
        return self.WARPING_FIELDS

    def warping_field_shapes(self):
        """ """
        return self.WARPING_SHAPES

    def warping_field_dtypes(self):
        """ """
        return self.WARPING_DTYPES

    def warping_fields(self):
        """ """
        return list(zip(self.warping_field_names(),
                   self.warping_field_shapes(),
                   self.warping_field_dtypes()))

    def warping_record_field_names(self):
        """ """
        return self.WARPING_RECORD_FIELDS

    def progress_field_names(self):
        """ """
        return self.PROGRESS_FIELDS

    def progress_field_shapes(self):
        """ """
        return self.PROGRESS_SHAPES

    def progress_field_dtypes(self):
        """ """
        return self.PROGRESS_DTYPES

    def progress_fields(self):
        """ """
        return list(zip(self.progress_field_names(),
                   self.progress_field_shapes(),
                   self.progress_field_dtypes()))

    def progress_record_field_names(self):
        """ """
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
