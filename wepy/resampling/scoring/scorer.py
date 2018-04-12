import random as rand
import itertools as it

import numpy as np

class Scorer(object):

    def score(self, walkers):
        raise NotImplementedError


class RandomScorer(Scorer):

    def __init__(self, seed=None):
        if seed is not None:
            self.seed = seed
            rand.seed(seed)

    def score(self, walkers):
        scores = []
        for walker in walkers:
            scores.append(rand.random())

        return scores, {}

class DistanceScorer(Scorer):

    def __init__(self, distance=None):
        assert distance is not None, "Must provide a Distance object"

        self.distance = distance

    def score(self, walkers):

        scores = []
        for walker in walkers:
            dist = self.distance.distance(walker)
            scores.append(dist)

        return scores, {}

class RankedDistanceScorer(DistanceScorer):

    def score(self, walkers):

        dists = []
        for walker in walkers:
            dists.append(self.distance.distance(walker))

        # report the orderings of the distances as the score
        scores = np.argsort(dists)

        return list(scores), {'distances' : np.array(dists)}


class AllToAllScorer(DistanceScorer):

    def _all_to_all_dist(self, walkers):

        # initialize an all-to-all matrix, with 0.0 for self distances
        dist_mat = np.zeros((len(walkers), len(walkers)))

        # make images for all the walker states for us to compute distances on
        images = []
        for walker in walkers:
            image = self.distance.image(walker.state)

        # get the combinations of indices for all walker pairs
        for i, j in it.combinations(range(len(walkers)), 2):

            # calculate the distance between the two walkers
            dist = self.distance.image_distance(images[i], images[j])

            # save this in the matrix in both spots
            dist_mat[i][j] = dist
            dist_mat[j][i] = dist

        # each row is the novelty vector for the walker, i.e. its
        # distance to all the other walkers
        return [walker_dists for walker_dists in dist_mat]

    def score(self, walkers):
        return self._all_to_all_dist(walkers)
