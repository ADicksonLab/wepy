"""Boundary conditions for random walk.
"""

# Standard Library
import itertools as it
import logging

logger = logging.getLogger(__name__)
# Standard Library
import time
from collections import defaultdict

# Third Party Library
import numpy as np
from geomm.centering import center_around
from geomm.distance import minimum_distance
from geomm.grouping import group_pair
from geomm.rmsd import calc_rmsd
from geomm.superimpose import superimpose

# First Party Library
from wepy.boundary_conditions.boundary import WarpBC
from wepy.util.util import box_vectors_to_lengths_angles
from wepy.walker import WalkerState


class RandomWalkBC(WarpBC):
    """Boundary condition for a random walk simulation with warping
    controlled by the sum of walker positions crossing a threshold.

    Implements the WarpBC superclass.

    This boundary condition will warp walkers to a number of initial
    states whenever a walker crosses the threshold distance from the
    origin.
    """

    # Records of boundary condition changes (sporadic)
    BC_FIELDS = WarpBC.BC_FIELDS + ("threshold_distance",)

    BC_SHAPES = WarpBC.BC_SHAPES + ((1,),)
    BC_DTYPES = WarpBC.BC_DTYPES + (int,)

    # warping (sporadic)
    WARPING_FIELDS = WarpBC.WARPING_FIELDS + ()
    WARPING_SHAPES = WarpBC.WARPING_SHAPES + ()
    WARPING_DTYPES = WarpBC.WARPING_DTYPES + ()

    WARPING_RECORD_FIELDS = WarpBC.WARPING_RECORD_FIELDS + ()

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = WarpBC.PROGRESS_FIELDS + ("distance",)
    PROGRESS_SHAPES = WarpBC.PROGRESS_SHAPES + (Ellipsis,)
    PROGRESS_DTYPES = WarpBC.PROGRESS_DTYPES + (int,)

    PROGRESS_RECORD_FIELDS = WarpBC.PROGRESS_RECORD_FIELDS + ("distance",)

    def __init__(
        self, threshold=None, initial_states=None, initial_weights=None, **kwargs
    ):
        """Constructor for RandomWalkBC.

        Arguments
        ---------

        threshold : int
            The threshold distance for recording a warping event.

        initial_states : list of objects implementing the State interface
            The list of possible states that warped walkers will assume.

        initial_weights : list of float, optional
            List of normalized probabilities of the initial_states
            provided. If not given, uniform probabilities will be
            used.

        ligand_idxs : arraylike of int
            The indices of the atom positions in the state considered
            the ligand.

        binding_site_idxs : arraylike of int
            The indices of the atom positions in the state considered
            the binding site.

        Raises
        ------
        AssertionError
            If any of the following kwargs are not given:
            threshold, initial_states.
        """

        super().__init__(
            initial_states=initial_states, initial_weights=initial_weights, **kwargs
        )

        # test inputs
        assert threshold is not None, "Must give a threshold distance"

        # save attributes
        self._threshold = threshold

    @property
    def threshold(self):
        """The cutoff RMSD for considering a walker bound."""
        return self._threshold

    def _progress(self, walker):
        """Calculate if the walker has bound and provide progress record.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        is_bound : bool
           Whether the walker is unbound (warped) or not

        progress_data : dict of str : value
           Dictionary of the progress record group fields
           for this walker alone.

        """

        pos = walker.state["positions"]

        distance = np.sum(pos)

        # test to see if the threshold was crossed
        crossed = False
        if distance >= self._threshold:
            crossed = True

        progress_data = {"distance": distance}

        return crossed, progress_data
