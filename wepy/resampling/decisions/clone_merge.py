from collections import namedtuple, defaultdict
from enum import Enum

import numpy as np

from wepy.resampling.decisions.decision import Decision

from wepy.walker import split, keep_merge

# the possible types of decisions that can be made enumerated for
# storage, these each correspond to specific instruction type
class CloneMergeDecisionEnum(Enum):
    NOTHING = 1
    CLONE = 2
    SQUASH = 3
    KEEP_MERGE = 4


class CloneMergeDecision(Decision):

    ENUM = CloneMergeDecisionEnum

    # namedtuple records for each decision type
    INSTRUCTION_RECORDS = (
        (ENUM.NOTHING, namedtuple("NothingInstructionRecord", ['slot'])),
        (ENUM.CLONE, namedtuple("CloneInstructionRecord", ['slot_a', 'slot_b'])),
        (ENUM.SQUASH, namedtuple("SquashInstructionRecord", ['merge_slot'])),
        (ENUM.KEEP_MERGE, namedtuple("KeepMergeInstructionRecord", ['slot'])),
    )

    # datatypes for each instruction, useful for allocating memory in
    # databases. Datatypes may be converted to a numpy dtype by
    # calling `dtype(instruct_dtype)`
    INSTRUCTION_DTYPES = (
        (ENUM.NOTHING, [('pos', np.int)]),
        (ENUM.CLONE, [('slot_a', np.int), ('slot_b', np.int)]),
        (ENUM.SQUASH, [('merge_to', np.int)]),
        (ENUM.KEEP_MERGE, [('pos', np.int)]),
    )

    @classmethod
    def action(cls, walkers, decisions):
        """Performs cloning and merging according to a list of resampling
        records for some walkers."""

        # list for the modified walkers
        mod_walkers = [None for i in range(len(walkers))]

        # we need to collect groups of merges, one entry for each
        # merge, where the key is the walker_idx of the keep merge slot
        squash_walkers = defaultdict(list)
        keep_walkers = {}
        # go through each decision and perform the decision
        # instructions
        for walker_idx, decision in enumerate(decisions):
            decision_value, instruction = decision

            if decision_value == cls.ENUM.NOTHING.value:
                # check to make sure a walker doesn't already exist
                # where you are going to put it
                if mod_walkers[instruction[0]] is not None:
                    raise ValueError(
                        "Multiple walkers assigned to position {}".format(instruction[0]))

                # put the walker in the position specified by the
                # instruction
                mod_walkers[instruction[0]] = walkers[walker_idx]

            elif decision_value == cls.ENUM.CLONE.value:
                walker = walkers[walker_idx]
                clones = split(walker, number=len(instruction))

                for i, walker_idx in enumerate(instruction):
                    if mod_walkers[walker_idx] is not None:
                        raise ValueError(
                            "Multiple walkers assigned to position {}".format(instruction[0]))
                    mod_walkers[walker_idx] = clones[i]

            # if it is a decision for merging we must perform this
            # once we know all the merge targets for each merge group
            elif decision_value == cls.ENUM.SQUASH.value:

                # save this walker to the appropriate merge group to
                # merge after going through the list of walkers
                squash_walkers[instruction[0]].append(walker_idx)

            elif decision_value == cls.ENUM.KEEP_MERGE.value:
                keep_walkers[instruction[0]] = walker_idx

            else:
                raise ValueError("Decision not recognized")

        # do the merging for each merge group
        for target_idx, walker_idxs in squash_walkers.items():
            keep_idx = keep_walkers[target_idx]
            # collect the walkers in the merge group, the keep idx is
            # always the first in the list
            merge_grp = [walkers[keep_idx]] + [walkers[i] for i in walker_idxs]

            # merge the walkers
            merged_walker = keep_merge(merge_grp, 0)

            # make sure there is not already a walker in this slot
            if mod_walkers[target_idx] is not None:
                raise ValueError(
                    "Multiple walkers assigned to position {}".format(target_idx))

            # set it in the slot for the keep_idx
            mod_walkers[target_idx] = merged_walker

        return mod_walkers

class MultiCloneMergeDecision(Decision):


    ENUM = CloneMergeDecisionEnum


    # mapping of enumerations to the InstructionRecord names that will
    # be passed to either namedtuple or VariableLengthRecord.
    INSTRUCTION_NAMES = (
        (ENUM.NOTHING, "NothingInstructionRecord"),
        (ENUM.CLONE, "CloneInstructionRecord"),
        (ENUM.SQUASH, "SquashInstructionRecord"),
        (ENUM.KEEP_MERGE, "KeepMergeInstructionRecord"),
    )

    # mapping of enumerations to the field names for the record. An Ellipsis instead
    # of fields indicate there is a variable number of fields.
    INSTRUCTION_FIELDS = (
        (ENUM.NOTHING, ('pos',)),
        (ENUM.CLONE, (Ellipsis,)),
        (ENUM.SQUASH, ('merge_to',)),
        (ENUM.KEEP_MERGE, ('pos',)),
    )

    # datatypes for each instruction, useful for allocating memory in
    # databases. These correspond to the fields defined in
    # INSTRUCTION_FIELDS. The dtype mapped to an Ellipsis field will
    # be the dtype for all of the fields it may create in a variable
    # length record
    INSTRUCTION_FIELD_DTYPES = (
        (ENUM.NOTHING, (np.int,)),
        (ENUM.CLONE, (np.int,)),
        (ENUM.SQUASH, (np.int,)),
        (ENUM.KEEP_MERGE, (np.int,)),
    )


    # the decision types that pass on their state
    ANCESTOR_DECISION_IDS = (ENUM.NOTHING.value,
                             ENUM.KEEP_MERGE.value,
                             ENUM.CLONE.value,)

    @classmethod
    def action(cls, walkers, decisions):
        """Performs cloning and merging according to a list of resampling
        records for some walkers."""

        # list for the modified walkers
        mod_walkers = [None for i in range(len(walkers))]

        # perform clones and merges for each step of resampling
        for step_idx, step_recs in enumerate(decisions):

            # we need to collect groups of merges, one entry for each
            # merge, where the key is the walker_idx of the keep merge slot
            squash_walkers = defaultdict(list)
            keep_walkers = {}
            # go through each decision and perform the decision
            # instructions
            for walker_idx, decision in enumerate(step_recs):
                decision_value, instruction = decision

                if decision_value == cls.ENUM.NOTHING.value:
                    # check to make sure a walker doesn't already exist
                    # where you are going to put it
                    if mod_walkers[instruction[0]] is not None:
                        raise ValueError(
                            "Multiple walkers assigned to position {}".format(instruction[0]))

                    # put the walker in the position specified by the
                    # instruction
                    mod_walkers[instruction[0]] = walkers[walker_idx]

                elif decision_value == cls.ENUM.CLONE.value:
                    walker = walkers[walker_idx]
                    clones = split(walker, number=len(instruction))

                    for i, walker_idx in enumerate(instruction):
                        if mod_walkers[walker_idx] is not None:
                            raise ValueError(
                                "Multiple walkers assigned to position {}".format(instruction[0]))
                        mod_walkers[walker_idx] = clones[i]

                # if it is a decision for merging we must perform this
                # once we know all the merge targets for each merge group
                elif decision_value == cls.ENUM.SQUASH.value:

                    # save this walker to the appropriate merge group to
                    # merge after going through the list of walkers
                    squash_walkers[instruction[0]].append(walker_idx)

                elif decision_value == cls.ENUM.KEEP_MERGE.value:
                    keep_walkers[instruction[0]] = walker_idx

                else:
                    raise ValueError("Decision not recognized")

            # do the merging for each merge group
            for target_idx, walker_idxs in squash_walkers.items():
                keep_idx = keep_walkers[target_idx]
                # collect the walkers in the merge group, the keep idx is
                # always the first in the list
                merge_grp = [walkers[keep_idx]] + [walkers[i] for i in walker_idxs]

                # merge the walkers
                merged_walker = keep_merge(merge_grp, 0)

                # make sure there is not already a walker in this slot
                if mod_walkers[target_idx] is not None:
                    raise ValueError(
                        "Multiple walkers assigned to position {}".format(target_idx))

                # set it in the slot for the keep_idx
                mod_walkers[target_idx] = merged_walker

        return mod_walkers




    @classmethod
    def parent_panel(cls, resampling_panel):

        parent_panel = []
        for cycle_idx, cycle in enumerate(resampling_panel):

            # each stage in the resampling for that cycle
            # make a stage parent table
            parent_table = []

            # now iterate through the rest of the stages
            for step_idx, step in enumerate(cycle):

                # initialize a list for the parents of this stages walkers
                step_parents = [None for i in range(len(step))]

                # the rest of the stages parents are based on the previous stage
                for parent_idx, parent_rec in enumerate(step):

                    # if the decision is an ancestor then the instruction
                    # values will be the children
                    if parent_rec[0] in cls.ANCESTOR_DECISION_IDS:
                        child_idxs = parent_rec[1]
                        for child_idx in child_idxs:
                            step_parents[child_idx] = parent_idx

                # for the full stage table save all the intermediate parents
                parent_table.append(step_parents)

            # for the full parent panel
            parent_panel.append(parent_table)

        return parent_panel

    @classmethod
    def net_parent_table(cls, parent_panel):

        #net_parent_table = [[i for i in range(len(parent_panel[0][0]))]]
        net_parent_table = []

        # each cycle
        for cycle_idx, step_parent_table in enumerate(parent_panel):
            # for the net table we only want the end results,
            # we start at the last cycle and look at its parent
            step_net_parents = []
            n_steps = len(step_parent_table)
            for walker_idx, parent_idx in enumerate(step_parent_table[-1]):
                # initialize the root_parent_idx which will be updated
                root_parent_idx = parent_idx

                # if no resampling skip the loop and just return the idx
                if n_steps > 0:
                    # go back through the steps getting the parent at each step
                    for prev_step_idx in range(n_steps):
                        prev_step_parents = step_parent_table[-(prev_step_idx+1)]
                        root_parent_idx = prev_step_parents[root_parent_idx]

                # when this is done we should have the index of the root parent,
                # save this as the net parent index
                step_net_parents.append(root_parent_idx)

            # for this step save the net parents
            net_parent_table.append(step_net_parents)

        return net_parent_table
