from collections import namedtuple
import random as rand

from wepy.walker import merge
from wepy.resampling.decision import CloneMergeDecision

class Resampler(object):

    def resample(self, walkers, decisions):
        raise NotImplementedError

ResamplingRecord = namedtuple("ResamplingRecord", ['decision', 'value'])


# stubs and examples of resamplers for testing purposes
class NoResampler(Resampler):

    def resample(self, walkers):

        resampling_record = [(CloneMergeDecision.NOTHING, i) for i in len(walkers)]

        return walkers, resampling_record

class RandomCloneMergeResampler(Resampler):
    def __init__(self, seed):
        self.seed = seed
        rand.seed(seed)

    def resample(self, walkers, debug_prints=False):

        n_walkers = len(walkers)
        result_template_str = "|".join(["{:^10}" for i in range(n_walkers)])
        # choose number of clone-merges between 1 and 10
        n_clone_merges = rand.randint(0,10)

        if debug_prints:
            print("Number of clone-merges to perform: {}".format(n_clone_merges))

        resampling_actions = []
        for resampling_stage_idx in range(n_clone_merges):

            if debug_prints:
                print("Resampling Stage: {}".format(resampling_stage_idx))
                print("---------------------")

            # keep track of the actions for each walker in this step
            # of resampling, initialize to CloneMergeDecision.NOTHING
            walker_actions = [CloneMergeDecision.NOTHING for i in range(n_walkers)]

            # choose a random walker to clone
            clone_idx = rand.randint(0, len(walkers))
            walker_actions[clone_idx] = CloneMergeDecision.CLONE
            clone_walker = walkers[clone_idx]

            # clone the chosen walker
            clone_children = clone_walker.clone()

            # choose a destination slot (index in the list) to put the clone in
            # the walker occupying that slot will be squashed
            # can't choose the same slot it is in
            squash_available = set(range(n_walkers)).difference({clone_idx})
            squash_idx = rand.choice([walker_idx for walker_idx in squash_available])
            walker_actions[squash_idx] = CloneMergeDecision.SQUASH
            squash_walker = walkers[squash_idx]

            # find a random merge target that is not either of the
            # cloned walkers
            merge_available = set(range(n_walkers)).difference({clone_idx, squash_idx})
            merge_idx = rand.choice([walker_idx for walker_idx in merge_available])
            walker_actions[merge_idx] = CloneMergeDecision.KEEP_MERGE
            merge_walker = walkers[merge_idx]

            # merge the squashed walker with the keep_merge walker
            merged_walker = squash_walker.squash(merge_walker)

            # make a new list of walkers
            resampled_walkers = []
            for idx, walker in enumerate(walkers):
                if idx == clone_idx:
                    # put one of the cloned walkers in the cloned one's place
                    resampled_walkers.append(clone_children.pop())
                elif idx == squash_idx:
                    # put one of the clone children in the squashed one's place
                    resampled_walkers.append(clone_children.pop())
                elif idx == merge_idx:
                    # put the merged walker in the keep_merged walkers place
                    resampled_walkers.append(merged_walker)
                else:
                    # if they did not move put them back where they were
                    resampled_walkers.append(walker)


            # add the walker actions list to the list of actions for
            # this resampling cycle
            resampling_actions.append(walker_actions)

            if debug_prints:
                # print the state of the walkers at this stage of resampling
                walker_state_str = result_template_str.format(
                    *[str(walker.state) for walker in resampled_walkers])
                print(walker_state_str)
                walker_weight_str = result_template_str.format(
                    *[str(walker.weight) for walker in resampled_walkers])
                print(walker_weight_str)

        return resampled_walkers, walker_actions

