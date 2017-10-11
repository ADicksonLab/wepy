import numpy as np

from  wepy.distance_functions.distance import Distance


class RandomWalkDistance(Distance):

    def _distance(self, walker_a, walker_b):
        num_dimension= len(walker_a)
        return 1/num_dimension * sum ([ abs(walker_a[0][i] - walker_b[0][i]) for i in range(num_dimension)])


    def distance(self, walkers):
        n_walkers = len (walkers)
        distance_matrix = np.zeros((n_walkers, n_walkers))
        for i in range(n_walkers):
            for j in range(i+1, n_walkers):
                distance_matrix[i][j] = self._distance(walkers[i].positions, walkers[j].positions)
        return distance_matrix
