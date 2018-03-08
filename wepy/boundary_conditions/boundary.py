import sys

import numpy as np

from wepy.walker import Walker

class BoundaryConditions(object):

    def __init__(self, **kwargs):

        pass

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
