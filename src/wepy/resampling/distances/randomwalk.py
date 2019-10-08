"""This module here is part of the RandomWalk object that implements
the distance metric for the RandomWalk walk system. This distance
metric is a scaled version of the Manhattan Norm.

"""

import logging

import numpy as np

from wepy.resampling.distances.distance import Distance


class RandomWalkDistance(Distance):
    """A class to implement the RandomWalkDistance metric for measuring
    differences between walker states. This is a normalized Manhattan
    distance measured between the difference in positions of the walkers.

    """

    def __init__(self):
        """Construct a RandomWalkDistance metric."""
        pass


    def image(self, state):
        """Transform a state into a random walk image.

        A random walk image is just the position of a walker in the
        N-dimensional space.

        Parameters
        ----------

        state : object implementing WalkerState
            A walker state object with positions in a numpy array
            of shape (N), where N is the the dimension of the random
            walk system.

        Returns
        -------

        randomwalk_image : array of floats of shape (N)
            The positions of a walker in the N-dimensional space.

        """
        return state['positions']

    def image_distance(self, image_a, image_b):
        """Compute the distance between the image of the two walkers.

        Parameters
        ----------

        image_a : array of float of shape (1, N)
            Position of the first walker's state.

        image_b:  array of float of shape (1, N)
            Position of the second walker's state.

       Returns
        -------

        distance: float
            The normalized Manhattan distance.

        """
        return np.average(np.abs(image_a - image_b))
