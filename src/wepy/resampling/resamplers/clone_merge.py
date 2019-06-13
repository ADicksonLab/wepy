import numpy as np

from wepy.resampling.resamplers.resampler import Resampler, ResamplerError
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision


class CloneMergeResampler(Resampler):
    """Abstract base class for resamplers using the clone-merge decision
    class.

    Provides some common functionality including handling minimum and
    maximum numbers of walkers and checking constraints on that.

    The 'assign_clones' method is a convenience method for generating
    decision records from two intermediate representation data
    structures which are convenient to produce: merge_groups and
    walker_clone_nums. See the docstring for more details on the
    format of these.

    """

    DECISION = MultiCloneMergeDecision

    RESAMPLING_FIELDS = DECISION.FIELDS + Resampler.CYCLE_FIELDS
    RESAMPLING_SHAPES = DECISION.SHAPES + Resampler.CYCLE_SHAPES
    RESAMPLING_DTYPES = DECISION.DTYPES + Resampler.CYCLE_DTYPES

    RESAMPLING_RECORD_FIELDS = DECISION.RECORD_FIELDS + Resampler.CYCLE_RECORD_FIELDS

    def __init__(self, pmin=1e-12, pmax=0.1,
                 min_num_walkers=Ellipsis,
                 max_num_walkers=Ellipsis,
                 **kwargs):
        """Constructor for CloneMegerResampler class.

        Parameters
        ----------

        pmin : float
            The minimum probability any walker is allowed to have.

        pmax : float
            The maximum probability any walker is allowed to have.

        """

        super().__init__(min_num_walkers=min_num_walkers,
                         max_num_walkers=max_num_walkers,
                         **kwargs)

        self._pmin=pmin
        self._pmax=pmax

    @property
    def pmin(self):
        return self._pmin

    @property
    def pmax(self):
        return self._pmax


    def _init_walker_actions(self, n_walkers):
        """Returns a list of default resampling records for a single
        resampling step.

        Parameters
        ----------

        n_walkers : int
            The number of walkers to generate records for

        Returns
        -------

        decision_records : list of dict of str: value
            A list of default decision records for one step of
            resampling.

        """

        # determine resampling actions
        walker_actions = [self.decision.record(
                                enum_value=self.decision.default_decision().value,
                                target_idxs=(i,))
                    for i in range(n_walkers)]

        return walker_actions

    def _check_resampled_walkers(self, resampled_walkers):
        """Check constraints on resampled walkers.

        Raises errors when constraints are violated.

        Parameters
        ----------
        resampled_walkers : list of Walker objects

        """

        walker_weights = np.array([walker.weight for walker in resampled_walkers])

        # check that all of the weights are less than or equal to the pmax
        overweight_walker_idxs = np.where(walker_weights > self.pmax)[0]
        if len(overweight_walker_idxs) > 0:

            raise ResamplerError("All walker weights must be less than the pmax, "
                                 "walkers {} are all overweight".format(
                                     ','.join([str(i) for i in overweight_walker_idxs])))

        # check that all walkers are greater than or equal to the pmin
        underweight_walker_idxs = np.where(walker_weights < self.pmin)[0]
        if len(underweight_walker_idxs) > 0:

            raise ResamplerError("All walker weights must be greater than the pmin, "
                                 "walkers {} are all underweight".format(
                                     ','.join([str(i) for i in underweight_walker_idxs])))



    def assign_clones(self, merge_groups, walker_clone_nums):
        """Convert two convenient data structures to a list of almost
        normalized resampling records.

        The two data structures are merge_groups and walker_clone_nums
        and are convenient to make.

        Each is a list with number of elements equal to the number of
        walkers that resampling will act on.

        Each element of the merge_groups is a list-like of integers
        indicating the indices of the walkers that will be merged into
        this one (i.e. squashed). A non-empty collection indicates a
        KEEP_MERGE decision.

        Each element of the walker_clone_nums is simply an integer
        specifying how many clones to make of this walker.

        These data structures simply declare requirements on what the
        actual decision records must achieve. The actual placement of
        walkers in slots (indices) is unspecified and immaterial.

        Parameters
        ----------
        merge_groups : list of list of int
            The specification of which walkers will be squashed and merged.

        walker_clone_nums : list of int
            The number of clones to make for each walker.

        Returns
        -------

        walker_actions : list of dict of str: values
            List of resampling record like dictionaries. These are not
            completely normalized for consumption by reporters, since
            they don't have the right list-like wrappers.

        """

        n_walkers = len(walker_clone_nums)

        walker_actions = self._init_walker_actions(n_walkers)

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
                                                                      target_idxs=(walker_idx,))

                # make the record for the keep merge walker
                walker_actions[walker_idx] = self.decision.record(self.decision.ENUM.KEEP_MERGE.value,
                                                         target_idxs=(walker_idx,))

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
                                                         target_idxs=tuple(clone_targets))

        return walker_actions
