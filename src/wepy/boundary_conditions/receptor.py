"""Boundary conditions for receptor based boundary conditions
including unbinding and rebinding.

"""

# Standard Library
import itertools as it
import logging

logger = logging.getLogger(__name__)
# Standard Library
import time
from collections import defaultdict

# Third Party Library
import numpy as np
from geomm.centering import center_around
from geomm.distance import minimum_distance
from geomm.grouping import group_pair
from geomm.rmsd import calc_rmsd
from geomm.superimpose import superimpose

# First Party Library
from wepy.boundary_conditions.boundary import WarpBC
from wepy.util.util import box_vectors_to_lengths_angles
from wepy.walker import WalkerState


class ReceptorBC(WarpBC):
    """Abstract base class for ligand-receptor based boundary conditions.

    Provides shared utilities for warping walkers to any number of
    optionally weighted initial structures through a shared
    `warp_walkers` method.

    Non-abstract implementations of this class need only implement the
    `_progress` method which should return a boolean signalling a
    warping event and the dictionary-style warping record of the
    progress for only a single walker. These records will be collated
    into a single progress record across all walkers.

    Additionally, the `_update_bc` method can be overriden to return
    'BC' group records. That method should accept the arguments shown
    in this ABC and return a list of dictionary-style 'BC' records.

    Warping of walkers with multiple initial states will be done
    according to a choice of initial states weighted on their weights,
    if given.

    """

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

    def __init__(
        self,
        initial_states=None,
        initial_weights=None,
        ligand_idxs=None,
        receptor_idxs=None,
        **kwargs
    ):
        """Base constructor for ReceptorBC.

        This should be called immediately in the subclass `__init__`
        method.

        If the initial weights for each initial state are not given
        uniform weights are assigned to them.

        Arguments
        ---------
        ligand_idxs : arraylike of int
            The indices of the atom positions in the state considered
            the ligand.

        receptor_idxs : arraylike of int
            The indices of the atom positions in the state considered
            the receptor.

        Raises
        ------
        AssertionError
            If any of the following kwargs are not given:
            ligand_idxs, receptor_idxs.

        """

        super().__init__(
            initial_states=initial_states, initial_weights=initial_weights, **kwargs
        )

        # make sure necessary inputs are given
        assert ligand_idxs is not None, "Must give ligand indices"
        assert receptor_idxs is not None, "Must give binding site indices"

        self._ligand_idxs = ligand_idxs
        self._receptor_idxs = receptor_idxs

    @property
    def ligand_idxs(self):
        """The indices of the atom positions in the state considered the ligand."""
        return self._ligand_idxs

    @property
    def receptor_idxs(self):
        """The indices of the atom positions in the state considered the receptor."""

        return self._receptor_idxs


class RebindingBC(ReceptorBC):
    """Boundary condition for doing re-binding simulations of ligands to a
    receptor.

    Implements the ReceptorBC superclass.

    This boundary condition will warp walkers to a number of initial
    states whenever a walker becomes very close to the native (bound)
    state.

    Thus the choice of the 'initial_states' argument should be walkers
    which are completely unbound (the choice of which are weighted by
    'initial_weight') and the choice of 'native_state' should be of a
    ligand bound to the receptor, e.g. X-ray crystallography or docked
    structure.

    The cutoff for the boundary is an RMSD of the walker to the native
    state which is calculated by first aligning and superimposing the
    entire structure according the atom indices specified in
    'binding_site_idxs', and as the name suggests should correspond to
    some approximation of the binding site of the ligand that occurs
    in the native state. Then the raw RMSD of the native and walker
    ligands is calculated. If this RMSD is less than the 'cutoff_rmsd'
    argument the walker is warped.

    PROGRESS is reported for each walker from this rmsd.

    The BC records are never updated.

    """

    # Records of boundary condition changes (sporadic)
    BC_FIELDS = ReceptorBC.BC_FIELDS + ("native_rmsd_cutoff",)
    """The 'native_rmsd_cutoff' is the cutoff used to determine when
    walkers have re-bound to the receptor, which is defined as the
    RMSD of the ligand to the native ligand bound state, when the
    binding sites are aligned and superimposed.

    """

    BC_SHAPES = ReceptorBC.BC_SHAPES + ((1,),)
    BC_DTYPES = ReceptorBC.BC_DTYPES + (float,)

    BC_RECORD_FIELDS = ReceptorBC.BC_RECORD_FIELDS + ("native_rmsd_cutoff",)

    # warping (sporadic)
    WARPING_FIELDS = ReceptorBC.WARPING_FIELDS + ()
    WARPING_SHAPES = ReceptorBC.WARPING_SHAPES + ()
    WARPING_DTYPES = ReceptorBC.WARPING_DTYPES + ()

    WARPING_RECORD_FIELDS = ReceptorBC.WARPING_RECORD_FIELDS + ()

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ReceptorBC.PROGRESS_FIELDS + ("native_rmsd",)
    PROGRESS_SHAPES = ReceptorBC.PROGRESS_SHAPES + (Ellipsis,)
    PROGRESS_DTYPES = ReceptorBC.PROGRESS_DTYPES + (float,)

    PROGRESS_RECORD_FIELDS = ReceptorBC.PROGRESS_RECORD_FIELDS + ("native_rmsd",)
    """Records for the state of this record group.

    The 'native_rmsd' is the is the RMSD of the ligand to the native
    ligand bound state, when the binding sites are aligned and
    superimposed.

    """

    def __init__(
        self,
        native_state=None,
        cutoff_rmsd=0.2,
        initial_states=None,
        initial_weights=None,
        ligand_idxs=None,
        binding_site_idxs=None,
        **kwargs
    ):
        """Constructor for RebindingBC.

        Arguments
        ---------

        native_state : object implementing the State interface
            The reference bound state. Will be automatically centered.

        cutoff_rmsd : float
            The cutoff RMSD for considering a walker bound.

        initial_states : list of objects implementing the State interface
            The list of possible states that warped walkers will assume.

        initial_weights : list of float, optional
            List of normalized probabilities of the initial_states
            provided. If not given, uniform probabilities will be
            used.

        ligand_idxs : arraylike of int
            The indices of the atom positions in the state considered
            the ligand.

        binding_site_idxs : arraylike of int
            The indices of the atom positions in the state considered
            the binding site.

        Raises
        ------
        AssertionError
            If any of the following kwargs are not given:
            native_state, initial_states, ligand_idxs, receptor_idxs.


        """

        super().__init__(
            initial_states=initial_states,
            initial_weights=initial_weights,
            ligand_idxs=ligand_idxs,
            receptor_idxs=binding_site_idxs**kwargs,
        )

        # test inputs
        assert native_state is not None, "Must give a native state"
        assert type(cutoff_rmsd) is float

        native_state_d = native_state.dict()

        # save the native state and center it around it's binding site
        native_state_d["positions"] = center_around(
            native_state["positions"], binding_site_idxs
        )

        native_state = WalkerState(**native_state_d)

        # save attributes
        self._native_state = native_state
        self._cutoff_rmsd = cutoff_rmsd

    @property
    def native_state(self):
        """The reference bound state to which walkers are compared."""
        return self._native_state

    @property
    def cutoff_rmsd(self):
        """The cutoff RMSD for considering a walker bound."""
        return self._cutoff_rmsd

    @property
    def binding_site_idxs(self):
        """The indices of the atom positions in the state considered the binding site."""

        return self._receptor_idxs

    def _progress(self, walker):
        """Calculate if the walker has bound and provide progress record.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        is_bound : bool
           Whether the walker is unbound (warped) or not

        progress_data : dict of str : value
           Dictionary of the progress record group fields
           for this walker alone.

        """

        # first recenter the ligand and the receptor in the walker
        box_lengths, box_angles = box_vectors_to_lengths_angles(
            walker.state["box_vectors"]
        )
        grouped_walker_pos = group_pair(
            walker.state["positions"],
            box_lengths,
            self.binding_site_idxs,
            self.ligand_idxs,
        )

        # center the positions around the center of the binding site
        centered_walker_pos = center_around(grouped_walker_pos, self.binding_site_idxs)

        # superimpose the walker state positions over the native state
        # matching the binding site indices only
        sup_walker_pos, _, _ = superimpose(
            self.native_state["positions"],
            centered_walker_pos,
            idxs=self.binding_site_idxs,
        )

        # calculate the rmsd of the walker ligand (superimposed
        # according to the binding sites) to the native state ligand
        native_rmsd = calc_rmsd(
            self.native_state["positions"], sup_walker_pos, idxs=self.ligand_idxs
        )

        # test to see if the ligand is re-bound
        rebound = False
        if native_rmsd <= self.cutoff_rmsd:
            rebound = True

        progress_data = {"native_rmsd": native_rmsd}

        return rebound, progress_data


class UnbindingBC(ReceptorBC):
    """Boundary condition for ligand unbinding.

    Walkers will be warped (discontinuously) if all atoms in the
    ligand are at least a certain distance away from the atoms in the
    receptor (i.e. the min-min of ligand-receptor distances > cutoff).

    Warping will replace the walker state with the initial state given
    as a parameter to this class.

    Also reports on the progress of that min-min for each walker.

    """

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ReceptorBC.BC_FIELDS + ("boundary_distance",)
    """
    Only occurs at the start of the simulation and just reports on the
    min-min cutoff distance.
    """

    BC_SHAPES = ReceptorBC.BC_SHAPES + ((1,),)
    BC_DTYPES = ReceptorBC.BC_DTYPES + (float,)
    BC_RECORD_FIELDS = ReceptorBC.BC_RECORD_FIELDS + ("boundary_distance",)

    # warping (sporadic)
    WARPING_FIELDS = ReceptorBC.WARPING_FIELDS + ()
    WARPING_SHAPES = ReceptorBC.WARPING_SHAPES + ()
    WARPING_DTYPES = ReceptorBC.WARPING_DTYPES + ()

    WARPING_RECORD_FIELDS = ReceptorBC.WARPING_RECORD_FIELDS + ()

    # progress record group
    PROGRESS_FIELDS = ReceptorBC.PROGRESS_FIELDS + ("min_distances",)
    """
    The 'min_distances' field reports on the min-min ligand-receptor
    distance for each walker.

    """

    PROGRESS_SHAPES = ReceptorBC.PROGRESS_SHAPES + (Ellipsis,)
    PROGRESS_DTYPES = ReceptorBC.PROGRESS_DTYPES + (float,)
    PROGRESS_RECORD_FIELDS = ReceptorBC.PROGRESS_RECORD_FIELDS + ("min_distances",)

    def __init__(
        self,
        initial_state=None,
        cutoff_distance=1.0,
        topology=None,
        ligand_idxs=None,
        receptor_idxs=None,
        periodic=True,
        **kwargs
    ):
        """Constructor for UnbindingBC class.

        All the key-word arguments are necessary.

        The 'initial_state' should be the initial state of your
        simulation for proper non-equilibrium simulations.

        Arguments
        ---------
        initial_state : object implementing State interface
            The state walkers will take on after unbinding.

        cutoff_distance : float
            The distance that specifies the boundary condition. When
            the min-min ligand-receptor distance is less than this it
            will be warped.

        topology : str
            A JSON string of topology. DEPRECATED: No longer needed

        ligand_idxs : list of int
           Indices of the atoms in the topology that correspond to the ligands.

        receptor_idxs : list of int
           Indices of the atoms in the topology that correspond to the
           receptor for the ligand.

        Raises
        ------
        AssertionError
            If any of the following are not provided: initial_state,
            ligand_idxs, receptor_idxs

        AssertionError
            If the cutoff distance is not a float.

        Warnings
        --------
        The 'initial_state' should be the initial state of your
        simulation for proper non-equilibrium simulations.

        """

        # since the super class can handle multiple initial states we
        # wrap the single initial state to a list.
        super().__init__(
            initial_states=[initial_state],
            ligand_idxs=ligand_idxs,
            receptor_idxs=receptor_idxs,
            **kwargs
        )

        # test input
        assert type(cutoff_distance) is float

        self._cutoff_distance = cutoff_distance
        self._topology = topology

        # whether or not to use the periodic box vectors in the
        # distance calculation
        self._periodic = periodic

    @property
    def cutoff_distance(self):
        """The distance a ligand must be to be unbound."""
        return self._cutoff_distance

    @property
    def topology(self):
        """JSON string topology of the system.
        Note: Deprecated and will be removed in future versions."""
        return self._topology

    def _calc_min_distance(self, walker):
        """Min-min distance for a walker.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        min_distance : float

        """

        # first recenter the ligand and the receptor in the walker
        box_lengths, box_angles = box_vectors_to_lengths_angles(
            walker.state["box_vectors"]
        )
        grouped_walker_pos = group_pair(
            walker.state["positions"], box_lengths, self.receptor_idxs, self.ligand_idxs
        )

        min_dist = minimum_distance(
            grouped_walker_pos[self.ligand_idxs], grouped_walker_pos[self.receptor_idxs]
        )

        return min_dist

    def _progress(self, walker):
        """Calculate whether a walker has unbound and also provide a
        dictionary for a single walker in the progress records.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        is_unbound : bool
           Whether the walker is unbound (warped) or not

        progress_data : dict of str : value
           Dictionary of the progress record group fields
           for this walker alone.

        """

        min_distance = self._calc_min_distance(walker)

        # test to see if the ligand is unbound
        unbound = False
        if min_distance >= self._cutoff_distance:
            unbound = True

        progress_data = {"min_distances": min_distance}

        return unbound, progress_data

    def _update_bc(self, new_walkers, warp_data, progress_data, cycle):
        """Perform an update to the boundary conditions.

        This is only used on the first cycle to keep a record of the
        cutoff parameter.

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

        # Only report a record on
        # the first cycle which gives the distance at which walkers
        # are warped
        if cycle == 0:
            return [
                {
                    "boundary_distance": np.array([self._cutoff_distance]),
                },
            ]
        else:
            return []
