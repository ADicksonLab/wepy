"""Abstract base class and null boundary condition class for
conveniently making compliant boundary condition classes for use in
wepy.

"""

# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import random
import sys
from collections import defaultdict
from copy import deepcopy

# Third Party Library
import numpy as np

# First Party Library
from wepy.walker import Walker


class BoundaryConditions(object):
    """Abstract base class for conveniently making compliant boundary condition classes.

    Includes empty record group definitions and useful getters for those.

    """

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ()
    """String names of fields produced in this record group.

    Boundary condition (BC) records are typically used to report on
    changes to the state of the BC object.

    Notes
    -----

    These fields are not critical to the proper functioning of the
    rest of the wepy framework and can be modified freely.

    However, reporters specific to this boundary condition probably
    will make use of these records.

    """

    BC_SHAPES = ()
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

    BC_DTYPES = ()
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

    BC_RECORD_FIELDS = ()
    """Optional, names of fields to be selected for truncated
    representation of the record group.

    These entries should be strings that are previously contained in
    the 'FIELDS' class constant.

    While strictly no constraints on to which fields can be added here
    you should only choose those fields whose features could fit into
    a plaintext csv or similar format.

    """

    # warping (sporadic)
    WARPING_FIELDS = ("walker_idx", "target_idx", "weight")
    """String names of fields produced in this record group.

    Warping records are typically used to report whenever a walker
    satisfied the boundary conditions and was warped and had its
    state changed.

    Warnings
    --------

    Be careful when modifying these fields as they may be integrated
    with other wepy framework features. Namely recognition of
    discontinuous warping events for making contiguous trajectories
    from cloning and merging lineages.

    The behavior of whether or not a warping event is discontinuous is
    given by a `BoundaryCondition` class's `warping_discontinuity`
    which likely depends on the existence of particular fields.

    """

    WARPING_SHAPES = ((1,), (1,), (1,))
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

    WARPING_DTYPES = (int, int, float)
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

    WARPING_RECORD_FIELDS = ("walker_idx", "target_idx", "weight")
    """Optional, names of fields to be selected for truncated
    representation of the record group.

    These entries should be strings that are previously contained in
    the 'FIELDS' class constant.

    While strictly no constraints on to which fields can be added here
    you should only choose those fields whose features could fit into
    a plaintext csv or similar format.

    """

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ()
    """String names of fields produced in this record group.

    Progress records are typically used to report on measures of
    walkers at each cycle.

    Notes
    -----

    These fields are not critical to the proper functioning of the
    rest of the wepy framework and can be modified freely.

    However, reporters specific to this boundary condition probably
    will make use of these records.

    """

    PROGRESS_SHAPES = ()
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

    PROGRESS_DTYPES = ()
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

    PROGRESS_RECORD_FIELDS = ()
    """Optional, names of fields to be selected for truncated
    representation of the record group.

    These entries should be strings that are previously contained in
    the 'FIELDS' class constant.

    While strictly no constraints on to which fields can be added here
    you should only choose those fields whose features could fit into
    a plaintext csv or similar format.

    """

    def __init__(self, **kwargs):
        """Null constructor accepts and ignores any key word arguments."""

        pass

    def bc_field_names(self):
        """Access the class level FIELDS constant for this record group."""
        return self.BC_FIELDS

    def bc_field_shapes(self):
        """Access the class level SHAPES constant for this record group."""
        return self.BC_SHAPES

    def bc_field_dtypes(self):
        """Access the class level DTYPES constant for this record group."""
        return self.BC_DTYPES

    def bc_fields(self):
        """Returns a list of zipped field specs.

        Returns
        -------

        record_specs : list of tuple
            A list of the specs for each field, a spec is a tuple of
            type (field_name, shape_spec, dtype_spec)
        """
        return list(
            zip(self.bc_field_names(), self.bc_field_shapes(), self.bc_field_dtypes())
        )

    def bc_record_field_names(self):
        """Access the class level RECORD_FIELDS constant for this record group."""
        return self.BC_RECORD_FIELDS

    def warping_field_names(self):
        """Access the class level FIELDS constant for this record group."""
        return self.WARPING_FIELDS

    def warping_field_shapes(self):
        """Access the class level SHAPES constant for this record group."""
        return self.WARPING_SHAPES

    def warping_field_dtypes(self):
        """Access the class level DTYPES constant for this record group."""
        return self.WARPING_DTYPES

    def warping_fields(self):
        """Returns a list of zipped field specs.

        Returns
        -------

        record_specs : list of tuple
            A list of the specs for each field, a spec is a tuple of
            type (field_name, shape_spec, dtype_spec)
        """
        return list(
            zip(
                self.warping_field_names(),
                self.warping_field_shapes(),
                self.warping_field_dtypes(),
            )
        )

    def warping_record_field_names(self):
        """Access the class level RECORD_FIELDS constant for this record group."""
        return self.WARPING_RECORD_FIELDS

    def progress_field_names(self):
        """Access the class level FIELDS constant for this record group."""
        return self.PROGRESS_FIELDS

    def progress_field_shapes(self):
        """Access the class level SHAPES constant for this record group."""
        return self.PROGRESS_SHAPES

    def progress_field_dtypes(self):
        """Access the class level DTYPES constant for this record group."""
        return self.PROGRESS_DTYPES

    def progress_fields(self):
        """Returns a list of zipped field specs.

        Returns
        -------

        record_specs : list of tuple
            A list of the specs for each field, a spec is a tuple of
            type (field_name, shape_spec, dtype_spec)
        """
        return list(
            zip(
                self.progress_field_names(),
                self.progress_field_shapes(),
                self.progress_field_dtypes(),
            )
        )

    def progress_record_field_names(self):
        """Access the class level RECORD_FIELDS constant for this record group."""
        return self.PROGRESS_RECORD_FIELDS

    def warp_walkers(self, walkers, cycle):
        """Apply boundary condition logic to walkers.

        If walkers satisfy the boundary conditions then they will be
        'warped' and have a corresponding state change take
        place. Each event recorded is returned as a single
        dictionary-style record in 'warp_data' list. These records
        correspond to the 'WARPING' record group.

        Additional data calculated on walkers may be returned in the
        single 'progress_data' dictionary-style record, which
        corresponds to the 'PROGRESS' record group.

        Any changes to the internal state of the boundary condition
        object (e.g. modification of parameters) should be recorded in
        at least one dictionary-style record in the 'bc_data'
        list. This corresponds to the 'BC' record group.

        Parameters
        ----------
        walkers : list of walkers
            A list of objects implementing the Walker interface

        cycle : int
            The index of the cycle this is for. Used to generate proper records.

        Returns
        -------
        new_walkers : list of walkers
            A list of objects implementing the Walker interface, that have had
            boundary condition logic applied.

        warp_data : list of dict of str : value
            A list of dictionary style records for each warping event that occured.

        bc_data : list of dict of str : value
            A list of dictionary style records for each boundary condition state change
            event record that occured.

        progress_data : dict of str : list of value
           Dictionary style progress records. The values should be lists
           corresponding to each walker.

        """

        raise NotImplementedError

    @classmethod
    def warping_discontinuity(cls, warping_record):
        """Given a warping record returns either True for a discontiuity
        occured or False if a discontinuity did not occur.

        Parameters
        ----------
        warping_record : tuple
            A tuple record of type 'WARPING'

        Returns
        -------
        is_discontinuous : bool
            True if discontinuous warping record False if continuous.

        """

        raise NotImplementedError


class NoBC(BoundaryConditions):
    """Boundary conditions class that does nothing.

    You may use this class as a stub in order to have an boundary
    condition class. However, this is not necessary since boundary
    conditions are optional in the sim_manager anyhow.
    """

    def warp_walkers(self, walkers, cycle):
        """Apply boundary condition logic to walkers, of which there is none.

        Simply returns all walkers provided with empty records data
        since there is nothing to do.

        Parameters
        ----------
        walkers : list of walkers
            A list of objects implementing the Walker interface

        cycle : int
            The index of the cycle this is for. Used to generate proper records.

        Returns
        -------
        new_walkers : list of walkers
            A list of objects implementing the Walker interface, that have had
            boundary condition logic applied.

        warp_data : list of dict of str : value
            A list of dictionary style records for each warping event that occured.

        bc_data : list of dict of str : value
            A list of dictionary style records for each boundary condition state change
            event record that occured.

        progress_data : dict of str : list of value
           Dictionary style progress records. The values should be lists
           corresponding to each walker.

        """

        warp_data = []
        bc_data = []
        progress_data = {}

        return walkers, warp_data, bc_data, progress_data

    @classmethod
    def warping_discontinuity(cls, warping_record):
        # documented in superclass

        # always return false
        return False


class WarpBC(BoundaryConditions):
    """Base class for boundary conditions with warping."""

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ()
    BC_SHAPES = ()
    BC_DTYPES = ()

    BC_RECORD_FIELDS = ()

    # warping fields are directly inherited

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ()
    PROGRESS_SHAPES = ()
    PROGRESS_DTYPES = ()

    PROGRESS_RECORD_FIELDS = ()

    DISCONTINUITY_TARGET_IDXS = Ellipsis
    """Specifies which 'target_idxs' values are considered discontinuous targets.

    Values are either integer indices, Ellipsis (indicating all
    possible values are discontinuous), or None indicating no possible
    value is discontinuous.

    """

    def __init__(self, initial_states=None, initial_weights=None, **kwargs):
        """Base constructor for WarpBC.

        This should be called immediately in the subclass `__init__`
        method.

        If the initial weights for each initial state are not given
        uniform weights are assigned to them.

        Arguments
        ---------
        initial_states : list of objects implementing the State interface
            The list of possible states that warped walkers will assume.

        initial_weights : list of float, optional
            List of normalized probabilities of the initial_states
            provided. If not given, uniform probabilities will be
            used.

        Raises
        ------
        AssertionError
            If any of the following kwargs are not given:
            initial_states.

        """

        # make sure necessary inputs are given
        assert initial_states is not None, "Must give a set of initial states"

        self._initial_states = initial_states

        # we want to choose initial states conditional on their
        # initial probability if specified. If not specified assume
        # assume uniform probabilities.
        if initial_weights is None:
            self._initial_weights = [1 / len(initial_states) for _ in initial_states]
        else:
            self._initial_weights = initial_weights

    @property
    def initial_states(self):
        """The possible initial states warped walkers may assume."""
        return self._initial_states

    @property
    def initial_weights(self):
        """The probabilities of each initial state being chosen during a warping."""
        return self._initial_weights

    def _progress(self, walker):
        """The method that must be implemented in non-abstract subclasses.

        Should decide if a walker should be warped or not and what its
        progress is regardless.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        to_warp : bool
           Whether the walker should be warped or not.

        progress_data : dict of str : value
           Dictionary of the progress record group fields
           for this walker alone.

        """

        raise NotImplementedError

    def _warp(self, walker):
        """Perform the warping of a walker.

        Chooses an initial state to replace the walker's state with
        according to it's given weight.

        Returns a walker of the same type and weight.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        warped_walker : object implementing the Walker interface
            The walker with the state after the warping. Weight should
            be the same.

        warping_data : dict of str : value
           The dictionary-style 'WARPING' record for this
           event. Excluding the walker index which is done in the main
           `warp_walkers` method.

        """

        # choose a state randomly from the set of initial states
        target_idx = np.random.choice(
            range(len(self.initial_states)),
            1,
            p=self.initial_weights / np.sum(self.initial_weights),
        )[0]

        warped_state = self.initial_states[target_idx]

        # set the initial state into a new walker object with the same weight
        warped_walker = type(walker)(state=warped_state, weight=walker.weight)

        # the data for the warp
        warp_data = {
            "target_idx": np.array([target_idx]),
            "weight": np.array([walker.weight]),
        }

        return warped_walker, warp_data

    def _update_bc(self, new_walkers, warp_data, progress_data, cycle):
        """Perform an update to the boundary conditions.

        No updates to the bc are ever done in this null
        implementation.

        Parameters
        ----------
        new_walkers : list of walkers
            The walkers after warping.

        warp_data : list of dict

        progress_data : dict

        cycle : int

        Returns
        -------
        bc_data : list of dict
            The dictionary-style records for BC update events

        """

        # do nothing by default
        return []

    def warp_walkers(self, walkers, cycle):
        """Test the progress of all the walkers, warp if required, and update
        the boundary conditions.

        Arguments
        ---------

        walkers : list of objects implementing the Walker interface

        cycle : int
            The index of the cycle.

        Returns
        -------

        new_walkers : list of objects implementing the Walker interface
            The new set of walkers that may have been warped.

        warp_data : list of dict of str : value
            The dictionary-style records for WARPING update events


        bc_data : list of dict of str : value
            The dictionary-style records for BC update events

        progress_data : dict of str : arraylike
            The dictionary-style records for PROGRESS update events

        """

        new_walkers = []

        # sporadic, zero or many records per call
        warp_data = []
        bc_data = []

        # continual, one record per call
        progress_data = defaultdict(list)

        # calculate progress data
        all_progress_data = [self._progress(w) for w in walkers]

        for walker_idx, walker in enumerate(walkers):
            # unpack progress data
            to_warp, walker_progress_data = all_progress_data[walker_idx]

            # add that to the progress data record
            for key, value in walker_progress_data.items():
                progress_data[key].append(value)

            # if the walker is meets the requirements for warping warp
            # it
            if to_warp:
                # warp the walker
                warped_walker, walker_warp_data = self._warp(walker)

                # add the walker idx to the walker warp record
                walker_warp_data["walker_idx"] = np.array([walker_idx])

                # save warped_walker in the list of new walkers to return
                new_walkers.append(warped_walker)

                # save the instruction record of the walker
                warp_data.append(walker_warp_data)

                logger.info("WARP EVENT observed at {}".format(cycle))
                logger.info(
                    "Warped Walker Weight = {}".format(walker_warp_data["weight"])
                )

            # no warping so just return the original walker
            else:
                new_walkers.append(walker)

        # consolidate the progress data to an array of a single
        # feature vectors for the cycle
        for key, value in progress_data.items():
            progress_data[key] = value

        # if the boundary conditions need to be updated given the
        # cycle and state from warping perform that now and return any
        # record data for that
        bc_data = self._update_bc(new_walkers, warp_data, progress_data, cycle)

        return new_walkers, warp_data, bc_data, progress_data

    @classmethod
    def warping_discontinuity(cls, warping_record):
        """Tests whether a warping record generated by this class is
        discontinuous or not.

        Parameters
        ----------

        warping_record : tuple
            The WARPING type record.

        Returns
        -------

        is_discontinuous : bool
            True if a discontinuous warp False if continuous.

        """

        # if it is Ellipsis then all possible values are discontinuous
        if cls.DISCONTINUITY_TARGET_IDXS is Ellipsis:
            return True

        # if it is None then all possible values are continuous
        elif cls.DISCONTINUITY_TARGET_IDXS is None:
            return False

        # otherwise it will have a tuple of indices for the
        # target_idxs that are discontinuous targets
        elif warping_record[2] in cls.DISCONTINUITY_TARGET_IDXS:
            return True

        # otherwise it wasn't a discontinuous target
        else:
            return False


class RandomBC(BoundaryConditions):
    """Boundary conditions that randomly warps both continuously and
    discontinuously.

    Can be used with any system as it won't actually mutate states.

    """

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ("ping",)
    BC_SHAPES = ((1,),)
    BC_DTYPES = (int,)

    BC_RECORD_FIELDS = ("ping",)

    # warping fields are directly inherited

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ("weight",)
    PROGRESS_SHAPES = (Ellipsis,)
    PROGRESS_DTYPES = (float,)

    PROGRESS_RECORD_FIELDS = ("weight",)

    DISCONTINUITY_TARGET_IDXS = (0,)

    def warp_walkers(self, walkers, cycle):
        ## warping walkers

        # just return the same walkers
        new_walkers = deepcopy(walkers)

        ## warping data

        warp_data = []
        # generate warping data: 50% of the time generate a warping
        # event, 25% is discontinuous (target 0), and 25% is
        # continuous (target 1)
        for walker_idx, walker in enumerate(walkers):
            # warping event?
            if random.random() >= 0.5:
                # discontinuous?
                if random.random() >= 0.5:
                    warp_record = {
                        "walker_idx": np.array([walker_idx]),
                        "target_idx": np.array([0]),
                        "weight": np.array([walker.weight]),
                    }

                    warp_data.append(warp_record)

                # continuous
                else:
                    warp_record = {
                        "walker_idx": np.array([walker_idx]),
                        "target_idx": np.array([1]),
                        "weight": np.array([walker.weight]),
                    }

                    warp_data.append(warp_record)

        ## BC data
        bc_data = []
        # choose whether to generate a bc record
        if random.random() >= 0.5:
            bc_data.append({"ping": np.array([1])})

        ## Progress data

        # just set the walker progress to be its weight so there is a
        # number there
        progress_data = {"weight": [walker.weight for walker in walkers]}

        return new_walkers, warp_data, bc_data, progress_data
