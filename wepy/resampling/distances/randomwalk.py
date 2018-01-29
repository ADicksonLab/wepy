"""
This module here is part of RandomWalk object that implements computing
distance between pairs of positions of RandomWalk walkers.
"""

import numpy as np

from wepy.resampling.distances.distance import Distance


class RandomWalkDistance(Distance):

    def __init__(self):
        pass

    """
    Computes the distance between pairs of positions and returns a distance matrix
    where the element (d_ij) is the average of the difference between posiotion of
    walker i and j.
    """
    def distance(self, walker_a, walker_b):
        """Compute the distance between posiotion of two walkers.

        :param position_a: posiotion of first walker
        :param position_b: posiotion of second walker
        :returns: a distance value
        :rtype: float
        """
        return np.average(np.abs(walker_a['posiotion'][0] - walker_b['posiotion'][0]))


    # def distance(self, walkers):
    #     """
    #     Computes the distances between pairs of walkers and returns the
    #     distance matrix.

    #     :param walkers: list of RandomWalker objects
    #     :returns: the symmetric matrix of distances
    #     :rtype: numpy array of shape (n_walkers, n_walkers)
    #     """

    #     n_walkers = len (walkers)
    #     # creates and initialize the distance matrix
    #     distance_matrix = np.zeros((n_walkers, n_walkers))

    #     #calls the distance method for pairs of walkers
    #     for i in range(n_walkers-1):
    #         for j in range(i+1, n_walkers):

    #             distance_matrix[i][j] = self._distance(walkers[i].positions, walkers[j].positions)
    #             distance_matrix[j][i] = distance_matrix[i][j]

    #     return distance_matrix
