import sys
import logging

import numpy as np

from wepy.walker import Walker

class BoundaryConditions(object):

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ()
    BC_SHAPES = ()
    BC_DTYPES = ()

    BC_RECORD_FIELDS = ()

    # warping (sporadic)
    WARPING_FIELDS = ()
    WARPING_SHAPES = ()
    WARPING_DTYPES = ()

    WARPING_RECORD_FIELDS = ()

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ()
    PROGRESS_SHAPES = ()
    PROGRESS_DTYPES = ()

    PROGRESS_RECORD_FIELDS = ()

    def __init__(self, **kwargs):

        pass

    def bc_field_names(self):
        return self.BC_FIELDS

    def bc_field_shapes(self):
        return self.BC_SHAPES

    def bc_field_dtypes(self):
        return self.BC_DTYPES

    def bc_fields(self):
        return list(zip(self.bc_field_names(),
                   self.bc_field_shapes(),
                   self.bc_field_dtypes()))

    def bc_record_field_names(self):
        return self.BC_RECORD_FIELDS

    def warping_field_names(self):
        return self.WARPING_FIELDS

    def warping_field_shapes(self):
        return self.WARPING_SHAPES

    def warping_field_dtypes(self):
        return self.WARPING_DTYPES

    def warping_fields(self):
        return list(zip(self.warping_field_names(),
                   self.warping_field_shapes(),
                   self.warping_field_dtypes()))

    def warping_record_field_names(self):
        return self.WARPING_RECORD_FIELDS

    def progress_field_names(self):
        return self.PROGRESS_FIELDS

    def progress_field_shapes(self):
        return self.PROGRESS_SHAPES

    def progress_field_dtypes(self):
        return self.PROGRESS_DTYPES

    def progress_fields(self):
        return list(zip(self.progress_field_names(),
                   self.progress_field_shapes(),
                   self.progress_field_dtypes()))

    def progress_record_field_names(self):
        return self.PROGRESS_RECORD_FIELDS

    def progress(self, walker):
        """ Checks if a walker is in a boundary and returns which boundary it is in"""
        raise NotImplementedError

    def warp_walkers(self, walkers):
        """Checks walkers for membership in boundaries and processes them
        according to the rules of the boundary."""
        raise NotImplementedError



class NoBC(BoundaryConditions):

    def check_boundaries(self, walker):
        return False, {}

    def warp_walkers(self, walkers):
        # in order the walkers after applying warps:
        # warping, bc, progress
        warp_data = {}
        bc_data = {}
        progress_data = {}

        return walkers, warp_data, bc_data, progress_data
