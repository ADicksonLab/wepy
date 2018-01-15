from collections import namedtuple, defaultdict
from enum import Enum

import numpy as np

from wepy.walker import split, keep_merge

# the record type for all resampling records
ResamplingRecord = namedtuple("ResamplingRecord", ['decision', 'instruction'])

# ABC for the Decision class
class Decision(object):
    ENUM = None
    INSTRUCTION_RECORDS = None
    INSTRUCTION_DTYPES = None


    @classmethod
    def enum_dict_by_name(cls):
        d = {}
        for enum in cls.ENUM:
            d[enum.name] = enum
        return d

    @classmethod
    def enum_dict_by_value(cls):
        d = {}
        for enum in cls.ENUM:
            d[enum.value] = enum
        return d

    @classmethod
    def enum_by_value(cls, enum_value):
        d = cls.enum_dict_by_value()
        return d[enum_value]

    @classmethod
    def enum_by_name(cls, enum_name):
        d = cls.enum_dict_by_name()
        return d[enum_name]

    @classmethod
    def instruct_record(cls, enum_value, data):
        enum = cls.enum_by_value(enum_value)
        instruct_record = dict(cls.INSTRUCTION_RECORDS)[enum](*data)
        return instruct_record

    @classmethod
    def record(cls, enum_value, data):
        instruct_record = cls.instruct_record(enum_value, data)

        resampling_record = ResamplingRecord(decision=enum_value,
                                             instruction=instruct_record)

        return resampling_record

    @classmethod
    def action(cls, walkers, decisions):
        raise NotImplementedError

# an example of a Decision class that has the enumeration, instruction
# record namedtuple, and the instruction dtypes
class NoDecision(Decision):

    class NothingDecisionEnum(Enum):
        NOTHING = 0

    ENUM = NothingDecisionEnum

    INSTRUCTION_RECORDS = (
        (ENUM.NOTHING, namedtuple("NothingInstructionRecord", ['slot'])),
    )

    INSTRUCTION_DTYPES = (
        (ENUM.NOTHING, [('pos', np.int)]),
    )

    @classmethod
    def action(cls, walkers, decisions):
        # list for the modified walkers
        mod_walkers = [None for i in range(len(walkers))]
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

        return mod_walkers


class CloneMergeDecision(Decision):

    class CloneMergeDecisionEnum(Enum):
        NOTHING = 1
        CLONE = 2
        SQUASH = 3
        KEEP_MERGE = 4


    ENUM = CloneMergeDecisionEnum

    INSTRUCTION_RECORDS = (
        (ENUM.NOTHING, namedtuple("NothingInstructionRecord", ['slot'])),
        (ENUM.CLONE, namedtuple("CloneInstructionRecord", ['slot_a', 'slot_b'])),
        (ENUM.SQUASH, namedtuple("SquashInstructionRecord", ['merge_slot'])),
        (ENUM.KEEP_MERGE, namedtuple("KeepMergeInstructionRecord", ['slot'])),
    )

    INSTRUCTION_DTYPES = (
        (ENUM.NOTHING, [('pos', np.int)]),
        (ENUM.CLONE, [('slot_a', np.int), ('slot_b', np.int)]),
        (ENUM.SQUASH, [('merge_to', np.int)]),
        (ENUM.KEEP_MERGE, [('pos', np.int)]),
    )

    @classmethod
    def action(cls, walkers, decisions):

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
