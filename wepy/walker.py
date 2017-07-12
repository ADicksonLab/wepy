
def merge(walkers):
    """Merge this walker with another keeping the state of one of them
    and adding the weights. """

    weights = [walker.weight for walker in walkers]
    # choose a walker according to their weights
    keep_walker = random.choices(walkers, weights=weights)
    # index of the kept walker
    keep_idx = walkers.index(keep_walker)
    # the others are "squashed" and we lose their state
    squashed_walkers = set(walkers).difference(keep_walker)
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
        the probability uniformly between clones."""

        # calculate the weight of all child walkers split uniformly
        split_prob = self.weight / number+1
        # make the clones
        clones = []
        for i in range(number):
            clones.append(type(self)(self.state, self.weight*split_prob))

        return clones

    def squash(self, merge_target):
        new_weight = self.weight + merge_target.weight
        return type(self)(merge_target.state, new_weight)

    def merge(self, other_walkers):
        return merge([self]+other_walkers)
