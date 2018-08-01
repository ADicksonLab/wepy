import itertools as it
from collections import defaultdict

import numpy as np

from wepy.resampling.decisions.decision import NoDecision


class Resampler(object):
    """Superclass for resamplers that use the the Novelty->Decider
    framework."""

    # data for resampling performed (continual)
    RESAMPLING_FIELDS = ()
    RESAMPLING_SHAPES = ()
    RESAMPLING_DTYPES = ()

    RESAMPLING_RECORD_FIELDS = ()

    # changes to the state of the resampler (sporadic)
    RESAMPLER_FIELDS = ()
    RESAMPLER_SHAPES = ()
    RESAMPLER_DTYPES = ()

    RESAMPLER_RECORD_FIELDS = ()


    def __init__(self, scorer, decider):
        self.scorer = scorer
        self.decider = decider
        self.decision = decider.DECISION

    def resampling_field_names(self):
        return self.RESAMPLING_FIELDS

    def resampling_field_shapes(self):
        return self.RESAMPLING_SHAPES

    def resampling_field_dtypes(self):
        return self.RESAMPLING_DTYPES

    def resampling_fields(self):
        return list(zip(self.resampling_field_names(),
                   self.resampling_field_shapes(),
                   self.resampling_field_dtypes()))

    def resampling_record_field_names(self):
        return self.RESAMPLING_RECORD_FIELDS

    def resampler_field_names(self):
        return self.RESAMPLER_FIELDS

    def resampler_field_shapes(self):
        return self.RESAMPLER_SHAPES

    def resampler_field_dtypes(self):
        return self.RESAMPLER_DTYPES

    def resampler_fields(self):
        return list(zip(self.resampler_field_names(),
                   self.resampler_field_shapes(),
                   self.resampler_field_dtypes()))

    def resampler_record_field_names(self):
        return self.RESAMPLER_RECORD_FIELDS

    def resample(self, walkers):

        aux_data = {}

        scores, scorer_aux = self.scorer.scores(walkers)
        decisions, decider_aux = self.decider.decide(scores)
        resampled_walkers = self.decider.decision.action(walkers, decisions)

        aux_data.update([scorer_aux, decider_aux])

        return resampled_walkers, resampling_records, resampler_records

    def assign_clones(self, merge_groups, walker_clone_nums):

        n_walkers = len(walker_clone_nums)

        # determine resampling actions
        walker_actions = [self.decision.record(self.decision.ENUM.NOTHING.value, (i,))
                    for i in range(n_walkers)]

        # keep track of which slots will be free due to squashing
        free_slots = []

        # go through the merge groups and write the records for them,
        # the index of a merge group determines the KEEP_MERGE walker
        # and the indices in the merge group are the walkers that will
        # be squashed
        for walker_idx, merge_group in enumerate(merge_groups):

            if len(merge_group) > 0:
                # add the squashed walker idxs to the list of open
                # slots
                free_slots.extend(merge_group)

                # for each squashed walker write a record and save it
                # in the walker actions
                for squash_idx in merge_group:
                    walker_actions[squash_idx] = self.decision.record(self.decision.ENUM.SQUASH.value,
                                                                      (walker_idx,))

                # make the record for the keep merge walker
                walker_actions[walker_idx] = self.decision.record(self.decision.ENUM.KEEP_MERGE.value,
                                                         (walker_idx,))

        # for each walker, if it is to be cloned assign open slots for it
        for walker_idx, num_clones in enumerate(walker_clone_nums):

            # if num_clones > 0 and len(merge_groups[walker_idx]) > 0:
            #     raise ValueError("Error! cloning and merging occuring with the same walker")

            # if this walker is to be cloned do so and consume the free
            # slot
            if num_clones > 0:

                # we first check to see if there are any free "slots"
                # for cloned walkers to go. If there are not we can
                # make room. The number of extra slots needed should
                # be default 0
                num_slots_needed = 0
                # initialize the lists of open slots
                this_free_slots = []
                new_slots = []
                if num_clones > len(free_slots):

                    # we get the difference in them in order to figure
                    # out how many extra slots will be created
                    num_slots_needed = num_clones - len(free_slots)

                    # collect free slots for this clone if there are any
                    if len(free_slots) > 0:
                        this_free_slots = [free_slots.pop() for clone in range(num_clones)]

                    # and make a list of the new slots
                    new_slots = [n_walkers + i for i in range(num_slots_needed)]

                    # then increase the number of walkers to match
                    n_walkers += num_slots_needed

                # then combine all the sources of slots, which
                # includes the spot this walker currently sits, slots
                # opened up by squashing, and newly created slots
                slots = [walker_idx] + \
                        this_free_slots + \
                        new_slots

                # DEBUG
                # check to see if there are any new slot assignments
                # if any([(True if idx >= n_walkers else False) for idx in slots]):
                #     # raise ValueError("Assignment of a clone to nonexistent slots")
                #     import ipdb; ipdb.set_trace()

                # make a record for this clone
                walker_actions[walker_idx] = self.decision.record(self.decision.ENUM.CLONE.value,
                                                         tuple(slots))

        return walker_actions

class NoResampler(Resampler):

    DECISION = NoDecision

    def __init__(self):
        self.decision = self.DECISION


    def resample(self, walkers, **kwargs):

        n_walkers = len(walkers)

        # determine resampling actions
        walker_actions = [self.decision.record(self.decision.ENUM.NOTHING.value, (i,))
                    for i in range(n_walkers)]

        return walkers, [walker_actions], {}
