import sys

import numpy as np

from wepy.walker import Walker

class BoundaryConditions(object):

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ()
    BC_SHAPES = ()
    BC_DTYPES = ()

    # warping (sporadic)
    WARPING_FIELDS = ()
    WARPING_SHAPES = ()
    WARPING_DTYPES = ()

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ()
    PROGRESS_SHAPES = ()
    PROGRESS_DTYPES = ()

    def __init__(self, **kwargs):

        pass

    @classmethod
    def bc_field_names(cls):
        return cls.BC_FIELDS

    @classmethod
    def bc_field_shapes(cls):
        return cls.BC_SHAPES

    @classmethod
    def bc_field_dtypes(cls):
        return cls.BC_DTYPES

    @classmethod
    def bc_fields(cls):
        return zip(cls.bc_field_names(),
                   cls.bc_field_shapes(),
                   cls.bc_field_dtypes())

    @classmethod
    def warping_field_names(cls):
        return cls.WARPING_FIELDS

    @classmethod
    def warping_field_shapes(cls):
        return cls.WARPING_SHAPES

    @classmethod
    def warping_field_dtypes(cls):
        return cls.WARPING_DTYPES

    @classmethod
    def warping_fields(cls):
        return list(zip(cls.warping_field_names(),
                   cls.warping_field_shapes(),
                   cls.warping_field_dtypes()))

    @classmethod
    def progress_field_names(cls):
        return cls.PROGRESS_FIELDS

    @classmethod
    def progress_field_shapes(cls):
        return cls.PROGRESS_SHAPES

    @classmethod
    def progress_field_dtypes(cls):
        return cls.PROGRESS_DTYPES

    @classmethod
    def progress_fields(cls):
        return list(zip(cls.progress_field_names(),
                   cls.progress_field_shapes(),
                   cls.progress_field_dtypes()))


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
