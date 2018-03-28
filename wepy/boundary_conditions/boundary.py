import sys

import numpy as np

from wepy.walker import Walker

class BoundaryConditions(object):

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ()
    BC_SHAPES = ()
    BC_DTYPES = ()

    # warping (sporadic)
    WARP_FIELDS = ()
    WARP_SHAPES = ()
    WARP_DTYPES = ()

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ()
    PROGRESS_SHAPES = ()
    PROGRESS_DTYPES = ()

    def __init__(self, **kwargs):

        pass

    @classmethod
    def bc_field_names(cls):
        return BC_FIELDS

    @classmethod
    def bc_field_shapes(cls):
        return BC_SHAPES

    @classmethod
    def bc_field_dtypes(cls):
        return BC_DTYPES

    @classmethod
    def bc_fields(cls):
        return zip(self.bc_field_names(),
                   self.bc_field_shapes(),
                   self.bc_field_dtypes())

    @classmethod
    def warp_field_names(cls):
        return WARP_FIELDS

    @classmethod
    def warp_field_shapes(cls):
        return WARP_SHAPES

    @classmethod
    def warp_field_dtypes(cls):
        return WARP_DTYPES

    @classmethod
    def warp_fields(cls):
        return zip(self.warp_field_names(),
                   self.warp_field_shapes(),
                   self.warp_field_dtypes())

    @classmethod
    def progress_field_names(cls):
        return PROGRESS_FIELDS

    @classmethod
    def progress_field_shapes(cls):
        return PROGRESS_SHAPES

    @classmethod
    def progress_field_dtypes(cls):
        return PROGRESS_DTYPES

    @classmethod
    def progress_fields(cls):
        return zip(self.progress_field_names(),
                   self.progress_field_shapes(),
                   self.progress_field_dtypes())


    def check_boundaries(self, walker):
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
        # in order the walkers after applying warps, warp records,
        # warp aux data, boundary conditions records, boundary
        # conditions aux data
        return walkers, [], {}, [], {}
