import logging

from wepy.resampling.resamplers.resampler import Resampler

# for the framework
from wepy.resampling.deciders.clone_merge import RandomCloneMergeDecider
from wepy.resampling.scoring.scorer import RandomScorer

# for the monolithic resampler
import random as rand
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision

from wepy.resampling.decisions.clone_merge import CloneMergeDecision

class RandomCloneMergeDecider(Resampler):
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


class RandomCloneMergeResamplerMonolithic(Resampler):

    """Example of a monolithic resampler that does not use any
    framework. Everything is implemented from scratch in this class,
    thus overriding everything in the super class.

    """

    # constants for the class
    DECISION = MultiCloneMergeDecision
    MIN_N_WALKERS = 3

    def __init__(self, seed=None, n_resamplings=10):
        if seed is not None:
            self.seed = seed
            rand.seed(seed)
        self.n_resamplings = n_resamplings



    def resample(self, walkers):

        n_walkers = len(walkers)

        # check to make sure there is enough walkers to clone and merge
        if n_walkers < self.MIN_N_WALKERS:
            raise TypeError("There must be at least 3 walkers to do cloning and merging")


        # choose number of clone-merges between 1 and 10
        n_clone_merges = rand.randint(0, self.n_resamplings)

        result_template_str = "|".join(["{:^10}" for i in range(n_walkers+1)])
        logging.info("Number of clone-merges to perform: {}".format(n_clone_merges))

        resampling_actions = []
        for resampling_stage_idx in range(n_clone_merges):

            logging.info("Resampling Stage: {}".format(resampling_stage_idx))


            # choose a random walker to clone
            clone_idx = rand.randint(0, len(walkers)-1)

            clone_walker = walkers[clone_idx]

            # clone the chosen walker
            clone_children = clone_walker.clone()

            # choose a destination slot (index in the list) to put the clone in
            # the walker occupying that slot will be squashed
            # can't choose the same slot it is in
            squash_available = set(range(n_walkers)).difference({clone_idx})
            squash_idx = rand.choice([walker_idx for walker_idx in squash_available])
            squash_walker = walkers[squash_idx]

            # find a random merge target that is not either of the
            # cloned walkers
            merge_available = set(range(n_walkers)).difference({clone_idx, squash_idx})
            merge_idx = rand.choice([walker_idx for walker_idx in merge_available])
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

            # reset the walkers for the next step as the resampled walkers
            walkers = resampled_walkers

            # make the decision records for this stage of resampling
            # initialize to RandomCloneMergeDecision.NOTHING, and their starting index
            walker_actions = [self.DECISION.record(self.DECISION.ENUM.NOTHING.value, (i,)) for
                              i in range(n_walkers)]

            # for the cloned one make a record for the instruction
            walker_actions[clone_idx] = self.DECISION.record(self.DECISION.ENUM.CLONE.value,
                                                             (clone_idx, squash_idx,))

            # for the squashed walker
            walker_actions[squash_idx] = self.DECISION.record(self.DECISION.ENUM.SQUASH.value,
                                                             (merge_idx,))

            # for the keep-merged walker
            walker_actions[merge_idx] = self.DECISION.record(self.DECISION.ENUM.KEEP_MERGE.value,
                                                             (merge_idx,))

            resampling_actions.append(walker_actions)

            # walker slot indices
            slot_str = result_template_str.format("slot", *[i for i in range(n_walkers)])
            logging.info(slot_str)

            # the resampling actions
            decisions = []
            instructions = []
            for rec in walker_actions:
                decisions.append(str(rec.decision.name))
                if rec.decision is self.DECISION.ENUM.CLONE:
                    instructions.append(str(",".join([str(i) for i in rec.instruction])))
                else:
                    instructions.append(str(rec.instruction))

            decision_str = result_template_str.format("decision", *decisions)
            instruction_str = result_template_str.format("instruct", *instructions)
            logging.info(decision_str)
            logging.info(instruction_str)

            # the state of the walkers at this stage of resampling
            walker_state_str = result_template_str.format("state",
                *[str(walker.state) for walker in resampled_walkers])
            logging.info(walker_state_str)
            walker_weight_str = result_template_str.format("weight",
                *[str(walker.weight) for walker in resampled_walkers])
            logging.info(walker_weight_str)


        # return values: resampled_walkers, resampler_records, resampling_data
        # we return no extra data from this resampler
        if n_clone_merges == 0:
            return walkers, [], {}
        else:
            # return the final state of the resampled walkers after all
            # stages, and the records of resampling
            return resampled_walkers, resampling_actions, {}
