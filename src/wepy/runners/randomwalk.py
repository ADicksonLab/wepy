#TODO: docstring
""" """

import random as rand
import logging

import numpy as np

from simtk import unit

from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState

UNIT_NAMES = (('positions_unit', unit.nanometer.get_name()),
         ('time_unit', unit.picosecond.get_name()),
        )
"""Mapping of unit identifier strings to the serialized string spec of the unit."""

class RandomWalkRunner(Runner):
    """RandomWalkRunner is an object for implementing the dynamic of
    RandomWalk system. To use it, you need to provide the number of dimensions
    and the probability of movement.
    """

    def __init__(self, dimension=2, probability=0.25):
        """Initialize RandomWalk object with the number of
        dimension and probability.

        dimension : int
            Number of dimensions for the space of the random walk.
             (Default = 2)

        probability : float
           (Default = 0.25)
        """

        self._dimension = dimension
        self._probability = probability

    #TODO: docstring
    @property
    def dimension(self):
        """ """
        return self._dimension

    #TODO: docstring
    @property
    def probability(self):
        """ """
        return self._probability

    #TODO: docstring
    def _walk(self, positions):
        """Implement the dynamic of RandomWalk system for one step.

        Takes the current position vector as input and based on the probability
        generates new position for each dimension and returns new position
        vector.

        Parameters
        ----------
        positions : arraylike
            a numpy array of shape (1, dimension)

        Returns
        -------
        new_positions : arraylike
            a numpy array of shape (1, dimension)

        """
        # make the deep copy of current posiotion
        new_positions = positions.copy()

        # iterates over each dimension
        for dimension in range(self.dimension):
            # Generates an uniform random number to choose between increasing or decreasing position
            r = rand.uniform(0, 1)

            # make a forward movement
            if r < self.probability:
                new_positions[0][dimension] += 1
            # make a backward movement
            else:
                new_positions[0][dimension] -= 1

            # implement the boundary condition for movement, movements to -1 are rejected
            if new_positions[0][dimension] < 0:
                new_positions[0][dimension] = 0

        return new_positions

    def run_segment(self, walker, segment_length):
        # documented in superclass

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
