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

    # changes to the state of the resampler (sporadic)
    RESAMPLER_FIELDS = ()
    RESAMPLER_SHAPES = ()
    RESAMPLER_DTYPES = ()


    def __init__(self, scorer, decider):
        self.scorer = scorer
        self.decider = decider
        self.decision = decider.DECISION

    @classmethod
    def resampling_field_names(cls):
        return cls.RESAMPLING_FIELDS

    @classmethod
    def resampling_field_shapes(cls):
        return cls.RESAMPLING_SHAPES

    @classmethod
    def resampling_field_dtypes(cls):
        return cls.RESAMPLING_DTYPES

    @classmethod
    def resampling_fields(cls):
        return list(zip(cls.resampling_field_names(),
                   cls.resampling_field_shapes(),
                   cls.resampling_field_dtypes()))


    @classmethod
    def resampler_field_names(cls):
        return cls.RESAMPLER_FIELDS

    @classmethod
    def resampler_field_shapes(cls):
        return cls.RESAMPLER_SHAPES

    @classmethod
    def resampler_field_dtypes(cls):
        return cls.RESAMPLER_DTYPES

    @classmethod
    def resampler_fields(cls):
        return list(zip(cls.resampler_field_names(),
                   cls.resampler_field_shapes(),
                   cls.resampler_field_dtypes()))

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
        # go through the merge groups and write the records for them
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

                if num_clones > len(free_slots):
                    raise ValueError("The number of clones exceeds the number of free slots.")

                # collect free slots for this clone, plus keeping one
                # at the original location
                slots = [free_slots.pop() for clone in range(num_clones)] + [walker_idx]

                # make a record for this clone
                walker_actions[walker_idx] = self.decision.record(self.decision.ENUM.CLONE.value,
                                                         tuple(slots))

        return walker_actions


    @staticmethod
    def resampling_actions_to_records(resampling_actions):

        resampling_records = defaultdict(list)
        for step_actions in resampling_actions:
            for walker_action in step_actions:
                for key, value in walker_action.items():
                    resampling_records[key].append(value)

        resampling_records = {key : np.array(values) for key, values in resampling_records.items()}

        return resampling_records

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
