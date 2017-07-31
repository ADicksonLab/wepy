from collections import namedtuple
import random as rand

from wepy.resampling.decision import Decision
from wepy.resampling.resampler import Resampler, ResamplingRecord

class CloneMergeDecision(Decision):
    NOTHING = 1
    CLONE = 2
    SQUASH = 3
    KEEP_MERGE = 4

class RandomCloneMergeResampler(Resampler):

    def __init__(self, seed, n_resamplings=10):
        self.seed = seed
        self.n_resamplings = n_resamplings
        rand.seed(seed)

    def resample(self, walkers, debug_prints=False):

        n_walkers = len(walkers)
        result_template_str = "|".join(["{:^10}" for i in range(n_walkers+1)])
        # choose number of clone-merges between 1 and 10
        n_clone_merges = rand.randint(0, self.n_resamplings)

        if debug_prints:
            print("Number of clone-merges to perform: {}".format(n_clone_merges))

        resampling_actions = []
        for resampling_stage_idx in range(n_clone_merges):

            if debug_prints:
                print("Resampling Stage: {}".format(resampling_stage_idx))
                print("---------------------")


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
            # initialize to CloneMergeDecision.NOTHING, and their starting index
            walker_actions = [ResamplingRecord(decision = CloneMergeDecision.NOTHING,
                                               instruction = i)
                              for i in range(n_walkers)]
            # for the cloned one make a record for the instruction
            walker_actions[clone_idx] = ResamplingRecord(
                decision = CloneMergeDecision.CLONE,
                instruction = (clone_idx, squash_idx))
            # for the squashed walker
            walker_actions[squash_idx] = ResamplingRecord(
                decision=CloneMergeDecision.SQUASH,
                instruction=merge_idx)
            # for the keep-merged walker
            walker_actions[merge_idx] = ResamplingRecord(
                decision=CloneMergeDecision.KEEP_MERGE,
                instruction=merge_idx)

            resampling_actions.append(walker_actions)

            if debug_prints:

                # walker slot indices
                slot_str = result_template_str.format("slot", *[i for i in range(n_walkers)])
                print(slot_str)

                # the resampling actions
                decisions = []
                instructions = []
                for rec in walker_actions:
                    decisions.append(str(rec.decision.name))
                    if rec.decision is CloneMergeDecision.CLONE:
                        instructions.append(str(",".join([str(i) for i in rec.instruction])))
                    else:
                        instructions.append(str(rec.instruction))

                decision_str = result_template_str.format("decision", *decisions)
                instruction_str = result_template_str.format("instruct", *instructions)
                print(decision_str)
                print(instruction_str)

                # print the state of the walkers at this stage of resampling
                walker_state_str = result_template_str.format("state",
                    *[str(walker.state) for walker in resampled_walkers])
                print(walker_state_str)
                walker_weight_str = result_template_str.format("weight",
                    *[str(walker.weight) for walker in resampled_walkers])
                print(walker_weight_str)


        if n_clone_merges == 0:
            return walkers, []
        else:
            # return the final state of the resampled walkers after all
            # stages, and the records of resampling
            return resampled_walkers, resampling_actions

def clone_parent_panel(clone_merge_resampling_record):
    # only the parents from the last stage of a cycle resampling
    full_parent_panel = []

    # each cycle
    for cycle_idx, cycle_stages in enumerate(clone_merge_resampling_record):

        # each stage in the resampling for that cycle
        # make a stage parent table
        stage_parent_table = []
        for stage_idx, stage in enumerate(cycle_stages):

            # the initial parents are just their own indices
            stage_parents = [i for i in range(len(stage))]
            # add it to the stage parents matrix
            stage_parent_table.append(stage_parents)
            # the rest of the stages parents are based on the previous stage
            for parent_idx, resampling_record in enumerate(stage):
                # the stage parents for the next stage, initialize to
                # the same idx
                stage_parents = [i for i in range(len(stage))]
                # if the parent is NOTHING or KEEP_MERGE it will have an index
                if resampling_record.decision in [CloneMergeDecision.NOTHING,
                                                  CloneMergeDecision.KEEP_MERGE]:
                    # single child
                    child_idx = resampling_record.instruction
                    # set the parent in the next row to this parent_idx
                    stage_parents[child_idx] = parent_idx
                # if it is CLONE it will have 2 indices
                elif resampling_record.decision is CloneMergeDecision.CLONE:
                    children = resampling_record.instruction[:]
                    for child_idx in children:
                        stage_parents[child_idx] = parent_idx
                elif resampling_record.decision  is CloneMergeDecision.SQUASH:
                    # do nothing this one has no children
                    pass
                else:
                    raise TypeError("Decision type not recognized")

                # for the full stage table save all the intermediate parents
                stage_parent_table.append(stage_parents)

        # for the full parent panel
        full_parent_panel.append(stage_parent_table)

    return full_parent_panel

def clone_parent_table(clone_merge_resampling_record):
    # all of the parent tables including within cycle stage parents,
    # this is 3D so I call it a panel
    net_parent_table = [[i for i in range(len(clone_merge_resampling_record[0][0]))]]

    # each cycle
    for cycle_idx, cycle_stages in enumerate(clone_merge_resampling_record):
        # each stage in the resampling for that cycle
        # make a stage parent table
        stage_parent_table = []
        for stage_idx, stage in enumerate(cycle_stages):

            # the initial parents are just their own indices,
            # this accounts for when there is no resampling done
            stage_parents = [i for i in range(len(stage))]
            # add it to the stage parents matrix
            stage_parent_table.append(stage_parents)

            # the rest of the stages parents are based on the previous stage
            for parent_idx, resampling_record in enumerate(stage):
                # the stage parents for the next stage, initialize to
                # the same idx
                stage_parents = [i for i in range(len(stage))]
                # if the parent is NOTHING or KEEP_MERGE it will have an index
                if resampling_record.decision in [CloneMergeDecision.NOTHING,
                                                  CloneMergeDecision.KEEP_MERGE]:
                    # single child
                    child_idx = resampling_record.instruction
                    # set the parent in the next row to this parent_idx
                    stage_parents[child_idx] = parent_idx
                # if it is CLONE it will have 2 indices
                elif resampling_record.decision is CloneMergeDecision.CLONE:
                    children = resampling_record.instruction[:]
                    for child_idx in children:
                        stage_parents[child_idx] = parent_idx
                elif resampling_record.decision  is CloneMergeDecision.SQUASH:
                    # do nothing this one has no children
                    pass
                else:
                    raise TypeError("Decision type not recognized")

                # for the full stage table save all the intermediate parents
                stage_parent_table.append(stage_parents)

            # for the net table we only want the end results,
            # we start at the last cycle and look at its parent
            stage_net_parents = []
            n_stages = len(stage_parent_table)
            for walker_idx, parent_idx in enumerate(stage_parent_table[-1]):
                # initialize the root_parent_idx which will be updated
                root_parent_idx = parent_idx

                # if no resampling skip the loop and just return the idx
                if n_stages > 0:
                    # go back through the stages getting the parent at each stage
                    for prev_stage_idx in range(n_stages):
                        prev_stage_parents = stage_parent_table[-(prev_stage_idx+1)]
                        root_parent_idx = prev_stage_parents[root_parent_idx]

                # when this is done we should have the index of the root parent,
                # save this as the net parent index
                stage_net_parents.append(root_parent_idx)

        # for this stage save the net parents
        net_parent_table.append(stage_net_parents)

    return net_parent_table
