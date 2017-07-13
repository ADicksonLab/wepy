from collections import namedtuple
import random as rand

from wepy.resampling.decision import Decision
from wepy.resampling.resampler import Resampler

class CloneMergeDecision(Decision):
    NOTHING = 1
    CLONE = 2
    SQUASH = 3
    KEEP_MERGE = 4

NothingInstructionRecord = namedtuple("NothingInstructionRecord", ['slot'])
CloneInstructionRecord = namedtuple("CloneInstructionRecord", ['slot_a', 'slot_b'])
SquashInstructionRecord = namedtuple("SquashInstructionRecord", ['merge_slot'])
KeepMergeInstructionRecord = namedtuple("KeepMergeInstructionRecord", ['slot'])

CLONE_MERGE_DECISION_INSTRUCTION_MAP = {CloneMergeDecision.NOTHING : NothingInstructionRecord,
                                        CloneMergeDecision.CLONE : CloneInstructionRecord,
                                        CloneMergeDecision.SQUASH : SquashInstructionRecord,
                                        CloneMergeDecision.KEEP_MERGE : KeepMergeInstructionRecord}

class RandomCloneMergeResampler(Resampler):
    def __init__(self, seed):
        self.seed = seed
        rand.seed(seed)



    def resample(self, walkers, debug_prints=False):

        n_walkers = len(walkers)
        result_template_str = "|".join(["{:^10}" for i in range(n_walkers+1)])
        # choose number of clone-merges between 1 and 10
        n_clone_merges = rand.randint(0,10)

        if debug_prints:
            print("Number of clone-merges to perform: {}".format(n_clone_merges))

        resampling_actions = []
        for resampling_stage_idx in range(n_clone_merges):

            if debug_prints:
                print("Resampling Stage: {}".format(resampling_stage_idx))
                print("---------------------")


            # choose a random walker to clone
            clone_idx = rand.randint(0, len(walkers))

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

            # make the decision records for this stage of resampling
            # initialize to CloneMergeDecision.NOTHING, and their starting index
            walker_actions = [(CloneMergeDecision.NOTHING, i) for i in range(n_walkers)]
            walker_actions[clone_idx] = (CloneMergeDecision.CLONE, [clone_idx, squash_idx])
            walker_actions[squash_idx] = (CloneMergeDecision.SQUASH, merge_idx)
            walker_actions[merge_idx] = (CloneMergeDecision.KEEP_MERGE, merge_idx)

            resampling_actions.append(walker_actions)

            if debug_prints:

                # walker slot indices
                slot_str = result_template_str.format("slot", *[i for i in range(n_walkers)])
                print(slot_str)

                # the resampling actions
                action_str = result_template_str.format("decision",
                    *[str(tup[0].name) for tup in walker_actions])
                print(action_str)
                data_str = result_template_str.format("instruct",
                    *[str(tup[1]) for tup in walker_actions])
                print(data_str)

                # print the state of the walkers at this stage of resampling
                walker_state_str = result_template_str.format("state",
                    *[str(walker.state) for walker in resampled_walkers])
                print(walker_state_str)
                walker_weight_str = result_template_str.format("weight",
                    *[str(walker.weight) for walker in resampled_walkers])
                print(walker_weight_str)



        return resampled_walkers, resampling_actions






