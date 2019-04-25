"""
This module here is part of RandomWalk object that implements computing
distance between pairs of positions of RandomWalk walkers.
"""
import logging

import numpy as np

from wepy.resampling.distances.distance import Distance


class RandomWalkDistance(Distance):
    """Computes the distance between pairs of positions and returns a distance matrix
    where the element (d_ij) is the average of the difference between posiotion of
    walker i and j.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self):
        pass


    def image(self, state):
        """

        Parameters
        ----------
        state :
            

        Returns
        -------

        """
        return state['positions']

    def image_distance(self, image_a, image_b):
        """Compute the distance between posiotion of two states.

        Parameters
        ----------
        position_a :
            posiotion of first state
        position_b :
            posiotion of second state
        image_a :
            
        image_b :
            

        Returns
        -------
        float
            a distance value

        """
        return np.average(np.abs(image_a - image_b))
