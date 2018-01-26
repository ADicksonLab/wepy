import random as rand

from wepy.resampling.decisions.clone_merge import CloneMergeDecision
from wepy.resampling.deciders.decider import Decider

class RandomCloneMergeDecider(Decider):
    """ WIP not working!! """

    N_CLONES = 2
    MIN_N_WALKERS = N_CLONES + 1
    MIN_MERGE = 2

    DECISION = CloneMergeDecision
    INSTRUCTIONS = dict(DECISION.INSTRUCTION_RECORDS)

    def __init__(self, seed=None):
        raise NotImplementedError("WIP do not use")
        if seed is not None:
            self.seed = seed
            rand.seed(seed)

    def decide(self, novelties):

        n_walkers = len(novelties)
        # decide the maximum number of splittings that can be done in
        # one decision step, while keeping at least one merge target
        max_n_splits = (n_walkers-1) // self.N_CLONES
        # check to make sure there is enough walkers to clone and merge
        if max_n_splits < 1:
            raise TypeError("There must be at least 3 walkers to do cloning and merging")

        # choose a number of splittings to perform
        n_splits = rand.randint(1, max_n_splits)

        # the number of merges will be the same

        decisions = [None for _ in novelties]

        # for the number of splittings to do choose walkers to split
        # randomly
        split_idxs = rand.sample(list(range(n_walkers)), n_splits)

        # choose the target slots for these walkers randomly
        avail_slot_idxs = set(range(n_walkers))
        for split_idx in split_idxs:

            # take some of these slots to put clones into
            split_target_idxs = rand.sample(avail_slot_idxs, self.N_CLONES)

            # remove these from the available target slots
            avail_slot_idxs.difference_update(split_target_idxs)

            # save this decision and instruction for this split
            # walker
            decisions[split_idx] = self.DECISION.record(self.DECISION.ENUM.CLONE.value,
                                                        split_target_idxs)

        # make a set of the walkers available for merging
        avail_walker_idxs = set(range(n_walkers)).difference(split_idxs)

        # choose the target slots for the merges
        for merge_idx in range(n_splits):

            # choose the walkers to squash into this merge group
            merge_grp_walker_idxs = set(rand.sample(avail_walker_idxs, self.N_CLONES))

            # remove these from the available walker idxs
            avail_walker_idxs.difference_update(merge_grp_walker_idxs)

            # choose the walker state to keep at random
            keep_idx = rand.sample(merge_grp_walker_idxs, 1)[0]

            squash_walker_idxs = merge_grp_walker_idxs.difference([keep_idx])

            # choose a target slot to put the merged walker in
            merge_target_idx = rand.sample(avail_slot_idxs, 1)[0]

            # remove that slot from the available slots
            avail_slot_idxs.difference_update([merge_target_idx])

            # make a record for the keep walker
            decisions[keep_idx] = self.DECISION.record(self.DECISION.ENUM.KEEP_MERGE.value,
                                                       (merge_target_idx,))

            # make records for the squashed walkers
            for squash_walker_idx in squash_walker_idxs:
                decisions[squash_walker_idx] = self.DECISION.record(self.DECISION.ENUM.SQUASH.value,
                                                                    (merge_target_idx,))

        # all leftover actionless walkers get assigned NOTHING records
        for walker_idx in avail_walker_idxs:

            # choose a slot for this walker
            target_idx = rand.sample(avail_slot_idxs, 1)[0]

            # remove this from the available walkers
            avail_slot_idxs.difference_update([target_idx])

            # make the record
            decisions[walker_idx] = self.DECISION.record(self.DECISION.ENUM.NOTHING.value,
                                                  (target_idx,))

        return decisions, {}
