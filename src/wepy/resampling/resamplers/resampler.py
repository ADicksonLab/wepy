import itertools as it
from collections import defaultdict
from warnings import warn

import logging
from eliot import start_action, log_call

import numpy as np

from wepy.resampling.decisions.decision import Decision, NoDecision

class ResamplerError(Exception):
    """Error raised when some constraint on resampling properties is
    violated."""
    pass

class Resampler():
    """Abstract base class for implementing resamplers.

    All subclasses of Resampler must implement the 'resample' method.

    If extra reporting on resampling and resampler updates desired
    subclassed resamplers should update the following class constants
    for specifying the decision class and the names, shapes, data
    types, and table-like records for each record group:

    - DECISION
    - RESAMPLING_FIELDS
    - RESAMPLING_SHAPES
    - RESAMPLING_DTYPES
    - RESAMPLING_RECORD_FIELDS
    - RESAMPLER_FIELDS
    - RESAMPLER_SHAPES
    - RESAMPLER_DTYPES
    - RESAMPLER_RECORD_FIELDS

    The DECISION constant should be a
    wepy.resampling.decisions.decision.Decision subclass.

    This base class provides some hidden methods that are useful for
    various purposes.

    To help maintain constraints on the number of walkers in a
    simulation the constructor allows for setting of a minimum and/or
    maximum of walkers.

    These values are allowed to be either integers, None, or Ellipsis.

    Integers set hard values for the minimum and maximum values.

    None indicates that the min and max are unbounded. For the min
    this translates to a value of 1 since there must always be one
    walker.

    Ellipsis is an indicator to determine the minimum and maximum
    dependent on the number of walkers provided for resampling.

    If the max_num_walkers is Ellipsis and the number of walkers given
    is 10 then the maximum will be set to 10 for that
    resampling. Conversely, for if the minimum is Ellipsis.

    If both the min and max are set to Ellipsis then the number of
    walkers is always kept the same.

    Note that this does not implement any algorithm that actually
    decides how many walkers there will be but just checks that these
    constraints are met by those implementations in subclasses.

    To allow for this checking the '_resample_init' method should be
    called at the beginning of the 'resample' method and the
    '_resample_cleanup' should be called at the end of the 'resample'
    method.



    """

    DECISION = Decision
    """The decision class for this resampler."""

    CYCLE_FIELDS = ('step_idx', 'walker_idx',)
    """The fields that get added to the decision record for all resampling
    records. This places a record within a single destructured listing
    of records for a single cycle of resampling using the step and
    walker index.
    """

    CYCLE_SHAPES = ((1,), (1,),)
    """Data shapes of the cycle fields."""

    CYCLE_DTYPES = (np.int, np.int,)
    """Data types of the cycle fields """

    CYCLE_RECORD_FIELDS = ('step_idx', 'walker_idx',)
    """Optional, names of fields to be selected for truncated
    representation of the record group.
    """

    # data for resampling performed (continual)
    RESAMPLING_FIELDS = DECISION.FIELDS + CYCLE_FIELDS
    """String names of fields produced in this record group.

    Resampling records are typically used to report on the details of
    how walkers are resampled for a given resampling step.

    Warning
    -------

    This is a critical function of many other components of the wepy
    framework and probably shouldn't be altered by most developers.

    Thi is where the information about cloning and merging of walkers
    is given. Seeing as this is a most of the value proposition of
    wepy as a tool getting rid of it will render most of the framework
    useless.

    But sticking to the 'loosely coupled, tightly integrated' mantra
    you are free to modify these fields. This would be useful for
    implementing resampling strategies that do not follow basic
    cloning and merging. Just beware, that most of the lineage based
    analysis will be broken without implementing a new Decision class.

    """

    RESAMPLING_SHAPES = DECISION.SHAPES + CYCLE_SHAPES
    """Numpy-style shapes of all fields produced in records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A tuple of ints that specify the shape of the field element
       array.

    B. Ellipsis, indicating that the field is variable length and
       limited to being a rank one array (e.g. (3,) or (1,)).

    C. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    Note that the shapes must be tuple and not simple integers for rank-1
    arrays.

    Option B will result in the special h5py datatype 'vlen' and
    should not be used for large datasets for efficiency reasons.

    """

    RESAMPLING_DTYPES = DECISION.DTYPES + CYCLE_DTYPES
    """Specifies the numpy dtypes to be used for records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A `numpy.dtype` object.

    D. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    """

    RESAMPLING_RECORD_FIELDS = DECISION.RECORD_FIELDS + CYCLE_RECORD_FIELDS
    """Optional, names of fields to be selected for truncated
    representation of the record group.

    These entries should be strings that are previously contained in
    the 'FIELDS' class constant.

    While strictly no constraints on to which fields can be added here
    you should only choose those fields whose features could fit into
    a plaintext csv or similar format.

    """

    # changes to the state of the resampler (sporadic)
    RESAMPLER_FIELDS = ()
    """String names of fields produced in this record group.

    Resampler records are typically used to report on changes in the
    state of the resampler.

    Notes
    -----

    These fields are not critical to the proper functioning of the
    rest of the wepy framework and can be modified freely.

    However, reporters specific to this resampler probably will make
    use of these records.

    """

    RESAMPLER_SHAPES = ()
    """Numpy-style shapes of all fields produced in records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A tuple of ints that specify the shape of the field element
       array.

    B. Ellipsis, indicating that the field is variable length and
       limited to being a rank one array (e.g. (3,) or (1,)).

    C. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    Note that the shapes must be tuple and not simple integers for rank-1
    arrays.

    Option B will result in the special h5py datatype 'vlen' and
    should not be used for large datasets for efficiency reasons.

    """

    RESAMPLER_DTYPES = ()
    """Specifies the numpy dtypes to be used for records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A `numpy.dtype` object.

    D. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    """

    RESAMPLER_RECORD_FIELDS = ()
    """Optional, names of fields to be selected for truncated
    representation of the record group.

    These entries should be strings that are previously contained in
    the 'FIELDS' class constant.

    While strictly no constraints on to which fields can be added here
    you should only choose those fields whose features could fit into
    a plaintext csv or similar format.

    """


    # valid debug modes
    DEBUG_MODES = (True, False,)

    def __init__(self,
                 min_num_walkers=Ellipsis,
                 max_num_walkers=Ellipsis,
                 debug_mode=False,
                 **kwargs):
        """Constructor for Resampler class

        Parameters
        ----------

        min_num_walkers : int or None or Ellipsis
            The minimum number of walkers allowed to have. None is
            unbounded, and Ellipsis preserves whatever number of
            walkers were given as input as the minimum.

        max_num_walkers : int or None or Ellipsis
            The maximum number of walkers allowed to have. None is
            unbounded, and Ellipsis preserves whatever number of
            walkers were given as input as the maximum.

        debug_mode : bool
            Expert mode stuff don't use unless you know what you are doing.


        """

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

        # this will be used to save the number of walkers given during
        # resampling, we initialize to None
        self._resampling_num_walkers = None

        # initialize debug mode
        self._debug_mode = False

        # set them to the args given
        self.set_debug_mode(debug_mode)


    @property
    def decision(self):
        """The decision class for this resampler."""
        return self.DECISION

    def resampling_field_names(self):
        """Access the class level FIELDS constant for this record group."""
        return self.RESAMPLING_FIELDS

    def resampling_field_shapes(self):
        """Access the class level SHAPES constant for this record group."""
        return self.RESAMPLING_SHAPES

    def resampling_field_dtypes(self):
        """Access the class level DTYPES constant for this record group."""
        return self.RESAMPLING_DTYPES

    def resampling_fields(self):
        """Returns a list of zipped field specs.

        Returns
        -------

        record_specs : list of tuple
            A list of the specs for each field, a spec is a tuple of
            type (field_name, shape_spec, dtype_spec)
        """
        return list(zip(self.resampling_field_names(),
                   self.resampling_field_shapes(),
                   self.resampling_field_dtypes()))

    def resampling_record_field_names(self):
        """Access the class level RECORD_FIELDS constant for this record group."""
        return self.RESAMPLING_RECORD_FIELDS

    def resampler_field_names(self):
        """Access the class level FIELDS constant for this record group."""
        return self.RESAMPLER_FIELDS

    def resampler_field_shapes(self):
        """Access the class level SHAPES constant for this record group."""
        return self.RESAMPLER_SHAPES

    def resampler_field_dtypes(self):
        """Access the class level DTYPES constant for this record group."""
        return self.RESAMPLER_DTYPES

    def resampler_fields(self):
        """Returns a list of zipped field specs.

        Returns
        -------

        record_specs : list of tuple
            A list of the specs for each field, a spec is a tuple of
            type (field_name, shape_spec, dtype_spec)
        """
        return list(zip(self.resampler_field_names(),
                   self.resampler_field_shapes(),
                   self.resampler_field_dtypes()))

    def resampler_record_field_names(self):
        """Access the class level RECORD_FIELDS constant for this record group."""
        return self.RESAMPLER_RECORD_FIELDS

    @property
    def is_debug_on(self):
        """ """
        return self._debug_mode

    def set_debug_mode(self, mode):
        """

        Parameters
        ----------
        mode

        Returns
        -------

        """

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
        """ """
        if self.is_debug_on:
            warn("Debug mode is already on")

        self.set_debug_mode(True)

    def debug_off(self):
        """ """
        if not self.is_debug_on:
            warn("Debug mode is already off")

        self.set_debug_mode(False)

    @property
    def max_num_walkers_setting(self):
        """The specification for the maximum number of walkers for the resampler."""
        return self._max_num_walkers

    @property
    def min_num_walkers_setting(self):
        """The specification for the minimum number of walkers for the resampler. """
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
        """Sets the concrete number of walkers constraints given a number of
        walkers and the settings for max and min.

        Parameters
        ----------
        num_walkers : int

        """

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



    def _resample_init(self, walkers, **kwargs):
        """Common initialization stuff for resamplers.

        Sets the number of walkers in this round of resampling.

        Parameters
        ----------
        walkers : list of Walker objects

        """

        # first set how many walkers there are in this resampling
        self._set_resampling_num_walkers(len(walkers))

    def _resample_cleanup(self, **kwargs):
        """Common cleanup stuff for resamplers.

        Unsets the number of walkers for this round of resampling.

        """

        # unset the number of walkers for this resampling
        self._unset_resampling_num_walkers()


    @log_call(include_args=[],
              include_result=False)
    def resample(self, walkers, debug_mode=False):
        """Perform resampling on the set of walkers.

        Parameters
        ----------
        walkers : list of Walker objects
            The walkers that are to be resampled.

        debug_mode : bool
            Expert mode debugging setting, only forif you know exactly
            what you are doing.

        Returns
        -------

        resampled_walkers : list of Walker objects
            The set of resampled walkers

        resampling_data : list of dict of str: value
            A list of destructured resampling records from this round
            of resampling.

        resampler_data : list of dict of str: value
            A list of records recording how the state of the resampler
            was updated.

        """

        raise NotImplemented

        self._resample_init(walkers, debug_mode=debug_mode)


class NoResampler(Resampler):
    """The resampler which does nothing."""

    DECISION = NoDecision

    # must reset these when you change the decision
    RESAMPLING_FIELDS = DECISION.FIELDS + Resampler.CYCLE_FIELDS
    RESAMPLING_SHAPES = DECISION.SHAPES + Resampler.CYCLE_SHAPES
    RESAMPLING_DTYPES = DECISION.DTYPES + Resampler.CYCLE_DTYPES

    RESAMPLING_RECORD_FIELDS = DECISION.RECORD_FIELDS + Resampler.CYCLE_RECORD_FIELDS

    @log_call(include_args=[],
              include_result=False)
    def resample(self, walkers, **kwargs):

        self._resample_init(walkers=walkers)

        n_walkers = len(walkers)

        # the walker actions are all nothings with the same walker
        # index which is the default initialization
        resampling_data = self._init_walker_actions(n_walkers)

        # normally decide is only for a single step and so does not
        # include the step_idx, so we add this to the records, and
        # convert the target idxs and decision_id to feature vector
        # arrays
        for walker_idx, walker_record in enumerate(resampling_data):
            walker_record['walker_idx'] = np.array([walker_idx])
            walker_record['step_idx'] = np.array([0])
            walker_record['walker_idx'] = np.array([walker_record['walker_idx']])
            walker_record['decision_id'] = np.array([walker_record['decision_id']])
            walker_record['target_idxs'] = np.array([walker_record['walker_idx']])

        # we only have one step so our resampling_records are just the
        # single list of walker actions
        resampling_data = resampling_data

        # there is no change in state in the resampler so there are no
        # resampler records
        resampler_data = [{}]

        # the resampled walkers are just the walkers

        self._resample_cleanup(resampling_data=resampling_data,
                               resampler_data=resampler_data,
                               walkers=walkers)

        return walkers, resampling_data, resampler_data

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
                              enum_value=self.decision.default_decision().value)
                    for i in range(n_walkers)]

        return walker_actions
