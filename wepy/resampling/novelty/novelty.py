import random as rand

import numpy as np

class NoveltyAssigner(object):

    def novelties(self, walkers):
        raise NotImplementedError


class RandomNoveltyAssigner(NoveltyAssigner):

    def __init__(self, seed=None):
        if seed is not None:
            self.seed = seed
            rand.seed(seed)

    def novelties(self, walkers):
        novelties = []
        for walker in walkers:
            novelties.append(rand.random())

        return novelties, {}

class DistanceNoveltyAssigner(NoveltyAssigner):

    def __init__(self, distance):
        self.distance = distance

    def novelties(self, walkers):

        novelties = []
        for walker in walkers:
            dist = self.distance.distance(walker)
            novelties.append(dist)

        return novelties, {}

class RankedDistanceNoveltyAssigner(DistanceNoveltyAssigner):

    def novelties(self, walkers):

        dists = []
        for walker in walkers:
            dists.append(self.distance.distance(walker))

        # report the orderings of the distances as the novelties
        novelties = np.argsort(dists)

        return list(novelties), {'distances' : np.array(dists)}
