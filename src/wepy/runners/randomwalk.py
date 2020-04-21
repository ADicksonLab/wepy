"""The random walk dynamics runner.

In this system, the state of the walkers is defined as an
N-dimensional vector of non-negative values. The walkers start at
position zero (in N-dimensional space) and randomly move a step either
forward or backward with the given probabilities. This is done in each
dimension at each dynamic step. All moves that result in a negative
position are rejected.

One potentioanl use of the random walk system is to test the
performance of differnt resamplers as seen in these papers:

"WExplore: Hierarchical Exploration of High-Dimensional Spaces
Using the Weighted Ensemble Algorithm" and
"REVO: Resampling of Ensembles by Variation Optimization".

"""

import random as rand
import logging

import numpy as np
from pint import UnitRegistry

from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState

units = UnitRegistry()

# the names of the units. We pass them through pint just to validate
# them
UNIT_NAMES = (('positions_unit', str(units('microsecond').units)),
         ('time_unit', str(units('picosecond').units)),
        )

"""Mapping of units identifiers to the corresponding pint units."""


class RandomWalkRunner(Runner):
    """RandomWalk runner for random walk simulations."""

    def __init__(self, probability=0.25):
        """Constructor for RandomWalkRunner.

        Parameters
        ----------

        probabilty : float
            "Probability" is defined here as the forward-move
             probability only. The backward-move probability is
             1-probability.(Default = 0.25)

        """

        self._probability = probability


    @property
    def probability(self):
        """ The probability of forward-move in an N-dimensional space"""
        return self._probability


    def _walk(self, positions):
        """Run dynamics for the RandomWalk system for one step.

        Parameters
        ----------
        positions : arraylike of shape (1, dimension)
            Current position of the walker.

        Returns
        -------
        new_positions : arraylike of shape (1, dimension)
            The positions of the walker after one dynamic step.

        """

        # make the deep copy of current posiotion
        new_positions = positions.copy()

        # get the dimension of the random walk space
        dimension = new_positions.shape[1]

        # iterates over each dimension
        for dim_idx in range(dimension):
            # Generates an uniform random number to choose between
            # moving forward or backward.
            rand_num = rand.uniform(0, 1)

            # make a forward movement
            if rand_num < self.probability:
                new_positions[0][dim_idx] += 1
            # make a backward movement
            else:
                new_positions[0][dim_idx] -= 1

            # implement the boundary condition for movement, movements
            # to -1 are rejected
            if new_positions[0][dim_idx] < 0:
                new_positions[0][dim_idx] = 0

        return new_positions

    def run_segment(self, walker, segment_length,
                    **kwargs):
        """Runs a random walk simulation for the given number of steps.

        Parameters
        ----------
        walker : object implementing the Walker interface
            The walker for which dynamics will be propagated.


        segment_length : int
            The numerical value that specifies how much dynamical steps
            are to be run.

        Returns
        -------
        new_walker : object implementing the Walker interface
            Walker after dynamics was run, only the state should be modified.

        """

        # Gets the current posiotion of RandomWalk Walker
        positions = walker.state['positions']

        # Make movements for the segment_length steps
        for _ in range(segment_length):
            # calls walk function for one step movement
            new_positions = self._walk(positions)
            positions = new_positions

        # makes new state form new positions
        new_state = WalkerState(positions=new_positions, time=0.0)

        # creates new_walker from new state and current weight
        new_walker = Walker(new_state, walker.weight)

        return new_walker
