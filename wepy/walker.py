import random as rand
import logging

def split(walker, number=2):
    # calculate the weight of all child walkers split uniformly
    split_prob = walker.weight / (number)
    # make the clones
    clones = []
    for i in range(number):
        clones.append(type(walker)(walker.state, split_prob))

    return clones

def keep_merge(walkers, keep_idx):

    weights = [walker.weight for walker in walkers]
    # but we add their weight to the new walker
    new_weight = sum(weights)
    # create a new walker with the keep_walker state
    new_walker = type(walkers[0])(walkers[keep_idx].state, new_weight)

    return new_walker

def merge(walkers):
    """Merge this walker with another keeping the state of one of them
    and adding the weights. """

    weights = [walker.weight for walker in walkers]
    # choose a walker according to their weights to keep its state
    keep_walker = rand.choices(walkers, weights=weights)
    keep_idx = walkers.index(keep_walker)

    # TODO do we need this?
    # the others are "squashed" and we lose their state
    # squashed_walkers = set(walkers).difference(keep_walker)

    # but we add their weight to the new walker
    new_weight = sum(weights)
    # create a new walker with the keep_walker state
    new_walker = type(walkers[0])(keep_walker.state, new_weight)

    return new_walker, keep_idx

class Walker(object):

    def __init__(self, state, weight):
        self.state = state
        self.weight = weight

    def clone(self, number=1):
        """Clone this walker by making a copy with the same state and split
        the probability uniformly between clones.

        The number is the increase in the number of walkers.

        e.g. number=1 will return 2 walkers with the same state as
        this object but with probability split 50/50 between them

        """

        # calculate the weight of all child walkers split uniformly
        split_prob = self.weight / (number+1)
        # make the clones
        clones = []
        for i in range(number+1):
            clones.append(type(self)(self.state, split_prob))

        return clones

    def squash(self, merge_target):
        new_weight = self.weight + merge_target.weight
        return type(self)(merge_target.state, new_weight)

    def merge(self, other_walkers):
        return merge([self]+other_walkers)

class WalkerState(object):

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getitem__(self, key):
        return self._data[key]

    def dict(self):
        return self._data
