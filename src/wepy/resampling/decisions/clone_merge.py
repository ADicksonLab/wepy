from collections import namedtuple, defaultdict
from enum import Enum
import logging

import numpy as np

from wepy.resampling.decisions.decision import Decision

from wepy.walker import split, keep_merge

# the possible types of decisions that can be made enumerated for
# storage, these each correspond to specific instruction type
class CloneMergeDecisionEnum(Enum):
    """Enum definition for cloning and merging decision values."

    - NOTHING : 1
    - CLONE : 2
    - SQUASH : 3
    - KEEP_MERGE : 4

    """

    NOTHING = 1
    """Do nothing with the walker sample. """

    CLONE = 2
    """Clone the walker into multiple children equally splitting the weight."""

    SQUASH = 3
    """Destroy the walker sample value (state) and donate the sample
    weight to another KEEP_MERGE walker sample."""

    KEEP_MERGE = 4
    """Do nothing with the sample value (state) but squashed walkers will
    donate their weight to it."""



class MultiCloneMergeDecision(Decision):
    """Decision encoding cloning and merging decisions for weighted ensemble.

    The decision records have in addition to the 'decision_id' a field
    called 'target_idxs'. This field has differing interpretations
    depending on the 'decision_id'.

    For NOTHING and KEEP_MERGE it indicates the walker index to assign
    this sample to after resampling. In this sense the walker is
    merely a vessel for the propagation of the state and acts as a
    slot.

    For SQUASH it indicates the walker that it's weight will be given
    to, which must have a KEEP_MERGE record for it.

    For CLONE it indicates the walker indices that clones of this one
    will be placed in. This field is variable length and the length
    corresponds to the number of clones.

    """


    ENUM = CloneMergeDecisionEnum

    DEFAULT_DECISION = ENUM.NOTHING

    FIELDS = Decision.FIELDS + ('target_idxs',)
    SHAPES = Decision.SHAPES + (Ellipsis,)
    DTYPES = Decision.DTYPES + (np.int,)

    RECORD_FIELDS = Decision.RECORD_FIELDS + ('target_idxs',)


    # the decision types that pass on their state
    ANCESTOR_DECISION_IDS = (ENUM.NOTHING.value,
                             ENUM.KEEP_MERGE.value,
                             ENUM.CLONE.value,)


    # TODO deprecate in favor of Decision implementation
    @classmethod
    def record(cls, enum_value, target_idxs):
        record = super().record(enum_value)
        record['target_idxs'] = target_idxs

        return record

    @classmethod
    def action(cls, walkers, decisions):

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
            for walker_idx, walker_rec in enumerate(step_recs):

                decision_value = walker_rec['decision_id']
                instruction = walker_rec['target_idxs']

                if decision_value == cls.ENUM.NOTHING.value:
                    # check to make sure a walker doesn't already exist
                    # where you are going to put it
                    if mod_walkers[instruction[0]] is not None:
                        raise ValueError(
                            "Multiple walkers assigned to position {}".format(instruction[0]))

                    # put the walker in the position specified by the
                    # instruction
                    mod_walkers[instruction[0]] = walkers[walker_idx]

                # for a clone
                elif decision_value == cls.ENUM.CLONE.value:

                    # get the walker to be cloned
                    walker = walkers[walker_idx]
                    # "clone" it by splitting it into walkers of the
                    # same state with even weights
                    clones = split(walker, number=len(instruction))

                    # then assign each of these clones to a target
                    # walker index in the next step
                    for clone_idx, target_idx in enumerate(instruction):

                        # check that there are not another walker
                        # already assigned to this position
                        if mod_walkers[target_idx] is not None:
                            raise ValueError(
                                "Multiple walkers assigned to position {}".format(instruction[0]))

                        # TODO this comment was just fixed so I
                        # believe that there was some serious problems
                        # before
                        # mod_walkers[walker_idx] = clones[clone_idx]

                        # assign the clone to the modified walkers of the next step
                        mod_walkers[target_idx] = clones[clone_idx]

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
                mod_walkers[keep_idx] = merged_walker


        if not all([False if walker is None else True for walker in mod_walkers]):

            raise ValueError("Some walkers were not created")

        return mod_walkers
