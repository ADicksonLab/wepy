import itertools as it
from collections import defaultdict
from warnings import warn
import logging

import numpy as np

from wepy.resampling.decisions.decision import NoDecision

class ResamplerError(Exception):
    pass

class Resampler():
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


    # valid debug modes
    DEBUG_MODES = (True, False,)

    def __init__(self, min_num_walkers=Ellipsis,
                 max_num_walkers=Ellipsis,
                 debug_mode=False):

        # the min and max number of walkers that can be generated in
        # resampling.

        # Ellipsis means to keep bound it by the number of
        # walkers given to the resample method (e.g. if
        # max_num_walkers == Ellipsis and min_num_walkers == 5 and
        # resample is given 10 then the max will be set to 10 for that
        # resampling and the min will always be 5. If both are
        # Ellipsis then the number of walkers is kept the same)

        # None means that there is no bound, e.g. max_num_walkers ==
        # None then there is no maximum number of walkers, however a
        # min_num_walkers of None in practice is 1 since there must
        # always be at least 1 walker

        if min_num_walkers not in (Ellipsis, None):
            if min_num_walkers < 1:
                raise ResamplerError("The minimum number of walkers should be at least 1")

        self._min_num_walkers = min_num_walkers
        self._max_num_walkers = max_num_walkers

        # initialize debug mode
        self._debug_mode = False

        # set them to the args given
        self.set_debug_mode(debug_mode)

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

    @property
    def is_debug_on(self):
        return self._debug_mode

    def set_debug_mode(self, mode):

        if mode not in self.DEBUG_MODES:
            raise ValueError("debug mode, {}, not valid".format(mode))

        self._debug_mode = mode

        # if you want to use debug mode you have to have ipdb installed
        if self.is_debug_on:
            try:
                import ipdb
            except ModuleNotFoundError:
                raise ModuleNotFoundError("You must have ipdb installed to use the debug feature")

    def debug_on(self):
        if self.is_debug_on:
            warn("Debug mode is already on")

        self.set_debug_mode(True)

    def debug_off(self):
        if not self.is_debug_on:
            warn("Debug mode is already off")

        self.set_debug_mode(False)

    @property
    def max_num_walkers_setting(self):
        return self._max_num_walkers

    @property
    def min_num_walkers_setting(self):
        return self._min_num_walkers

    def max_num_walkers(self):
        """" Get the max number of walkers allowed currently"""

        # first check to make sure that a resampling is occuring and
        # we have a number of walkers to even reference
        if self._resampling_num_walkers is None:
            raise ResamplerError(
            "A resampling is currently not taking place so the"\
            " current number of walkers is not known.")

        # we are in a resampling so there is a current value for the
        # max number of walkers
        else:

            # if the max is None then there is no max number of
            # walkers so we just return None
            if self.max_num_walkers_setting is None:
                return None

            # if the max is Ellipsis then we just return what the
            # current number of walkers is
            elif self.max_num_walkers_setting is Ellipsis:
                return self._resampling_num_walkers

            # if it is not those then it is a hard number and we just
            # return it
            else:
                return self.max_num_walkers_setting

    def min_num_walkers(self):
        """" Get the min number of walkers allowed currently"""

        # first check to make sure that a resampling is occuring and
        # we have a number of walkers to even reference
        if self._resampling_num_walkers is None:
            raise ResamplerError(
            "A resampling is currently not taking place so the"\
            " current number of walkers is not known.")

        # we are in a resampling so there is a current value for the
        # min number of walkers
        else:

            # if the min is None then there is no min number of
            # walkers so we just return None
            if self.min_num_walkers_setting is None:
                return None

            # if the min is Ellipsis then we just return what the
            # current number of walkers is
            elif self.min_num_walkers_setting is Ellipsis:
                return self._resampling_num_walkers

            # if it is not those then it is a hard number and we just
            # return it
            else:
                return self.min_num_walkers_setting


    def _set_resampling_num_walkers(self, num_walkers):

        # there must be at least 1 walker in order to do resampling
        if num_walkers < 1:
            raise ResamplerError("No walkers were given to resample")

        # if the min number of walkers is not dynamic check to see if
        # this number violates the hard boundary
        if self._min_num_walkers in (None, Ellipsis):
            self._resampling_num_walkers = num_walkers
        elif num_walkers < self._min_num_walkers:
            raise ResamplerError(
                "The number of walkers given to resample is less than the minimum")

        # if the max number of walkers is not dynamic check to see if
        # this number violates the hard boundary
        if self._max_num_walkers in (None, Ellipsis):
            self._resampling_num_walkers = num_walkers
        elif num_walkers < self._max_num_walkers:
            raise ResamplerError(
                "The number of walkers given to resample is less than the maximum")

    def _unset_resampling_num_walkers(self):

        self._resampling_num_walkers = None



    def _resample_init(self, walkers):
        """Common initialization stuff for resamplers. """

        # first set how many walkers there are in this resampling
        self._set_resampling_num_walkers(len(walkers))

    def _resample_cleanup(self):

        # unset the number of walkers for this resampling
        self._unset_resampling_num_walkers()


    def resample(self, walkers, debug_mode=False):

        raise NotImplemented

        self._resample_init(walkers, debug_mode=debug_mode)


    def _init_walker_actions(self):
        # determine resampling actions
        walker_actions = [self.decision.record(enum_value=self.decision.ENUM.NOTHING.value,
                                               target_idxs=(i,))
                    for i in range(n_walkers)]

        return walker_actions


    def assign_clones(self, merge_groups, walker_clone_nums):

        n_walkers = len(walker_clone_nums)

        walker_actions = self._init_walker_actions()

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

            if num_clones > 0 and len(merge_groups[walker_idx]) > 0:
                raise ResamplerError("Error! cloning and merging occuring with the same walker")

            # if this walker is to be cloned do so and consume the free
            # slot
            if num_clones > 0:

                # we first check to see if there are any free "slots"
                # for cloned walkers to go. If there are not we can
                # make room. The number of extra slots needed should
                # be default 0


                # we choose the targets for this cloning, we start
                # with the walker initiating the cloning
                clone_targets = [walker_idx]

                # if there are any free slots, then we use those first
                if len(free_slots) > 0:
                    clone_targets.extend([free_slots.pop() for clone in range(num_clones)])

                # if there are more slots needed then we will have to
                # create them
                num_slots_needed = num_clones - len(clone_targets)
                if num_slots_needed > 0:

                    # initialize the lists of open slots
                    new_slots = []

                    # and make a list of the new slots
                    new_slots = [n_walkers + i for i in range(num_slots_needed)]

                    # then increase the number of walkers to match
                    n_walkers += num_slots_needed

                    # then add these to the clone targets
                    clone_targets.extend(new_slots)

                # make a record for this clone
                walker_actions[walker_idx] = self.decision.record(self.decision.ENUM.CLONE.value,
                                                         tuple(clone_targets))

        return walker_actions

class ScoreDecideResampler(Resampler):
    """Superclass for resamplers that use the the Novelty->Decider
    framework."""

    def __init__(self, scorer, decider):
        self.scorer = scorer
        self.decider = decider
        self.decision = decider.DECISION

    def resample(self, walkers):

        # first set how many walkers there are in this resampling
        self._set_resample_num_walkers(len(walkers))

        aux_data = {}

        scores, scorer_aux = self.scorer.scores(walkers)
        decisions, decider_aux = self.decider.decide(scores)
        resampled_walkers = self.decider.decision.action(walkers, decisions)

        aux_data.update([scorer_aux, decider_aux])

        # unset the number of walkers for this resampling
        self._unset_resampling_num_walkers()

        return resampled_walkers, resampling_records, resampler_records


class NoResampler(Resampler):

    DECISION = NoDecision

    def __init__(self):
        self.decision = self.DECISION


    def resample(self, walkers, **kwargs):

        n_walkers = len(walkers)

        # the walker actions are all nothings with the same walker
        # index which is the default initialization
        walker_actions = self._init_walker_actions()

        # we only have one step so our resampling_records are just the
        # single list of walker actions
        resampling_data = [walker_actions]

        # there is no change in state in the resampler so there are no
        # resampler records
        resampler_data = [{}]

        return walkers, resampling_data, resampler_data
