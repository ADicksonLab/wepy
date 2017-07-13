from collections import namedtuple
import random as rand

from wepy.walker import merge
from wepy.resampling.decision import Decision

class Resampler(object):

    def resample(self, walkers, decisions):
        raise NotImplementedError

ResamplingRecord = namedtuple("ResamplingRecord", ['decision', 'value'])


# stubs and examples of resamplers for testing purposes
class NoResampler(Resampler):

    def resample(self, walkers):

        resampling_record = [(Decision.NOTHING, i) for i in len(walkers)]

        return walkers, resampling_record

class RandomCloneMergeResampler(Resampler):
    def __init__(self, seed):
        rand.seed(seed)

    def resample(self, walkers):

        # choose number of clone-merges between 1 and 10
        n_clonemerges = rand.randint(0,10)

        walker_actions = [[] for walker in walkers]
        resampling_records = []
        for i in range(n_clonemerges):

            # choose a random walker to clone
            clone_idx = rand.randint(0, len(walkers))
            walker_actions[i].append(Decision.CLONE)
            clone_walker = walkers[clone_idx]
            # clone the chosen walker
            walker_clone = clone_walker.clone()
            # choose a destination slot (index in the list) to put the clone in
            # the walker occupying that slot will be squashed
            # can't choose the same slot it is in
            squash_idx = rand.choice(set(range(n_walkers)).difference(clone_idx))
            walker_actions[i].append(Decision.SQUASH)
            squash_walker = walkers[squash_idx]

            # find a random merge target that is not either of the
            # cloned walkers
            merge_idx = rand.choice(set(range(n_walkers)).difference([clone_idx, squash_idx]))
            walker_actions[i].append(Decision.KEEP_MERGE)
            merge_walker = walkers[merge_idx]

            # merge the squashed walker with the keep_merge walker
            merged_walker = squash_walker.squash(merge_walker)

            # make a new list of walkers
            resampled_walkers = []
            for idx, walker in enumerate(walkers):
                if idx == squash_idx:
                    resampled_walkers.append(walker_clone)
                elif idx == merge_idx:
                    resampled_walkers.append(merged_walker)
                else:
                    resample_walkers.append(walker)


            return resampled_walkers, resampling_records

