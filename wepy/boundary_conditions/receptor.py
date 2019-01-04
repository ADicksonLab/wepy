"""Boundary conditions for receptor based boundary conditions
including unbinding and rebinding.

"""

from collections import defaultdict
import logging

import numpy as np

from geomm.recentering import recenter_pair
from geomm.superimpose import superimpose
from geomm.rmsd import calc_rmsd

import mdtraj as mdj

from wepy.util.util import box_vectors_to_lengths_angles

from wepy.boundary_conditions.boundary import BoundaryConditions

class ReceptorBC(BoundaryConditions):

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ()
    BC_SHAPES = ()
    BC_DTYPES = ()

    BC_RECORD_FIELDS = ()

    # warping (sporadic)
    WARPING_FIELDS = ('walker_idx', 'target_idx', 'weight')
    WARPING_SHAPES = ((1,), (1,), (1,))
    WARPING_DTYPES = (np.int, np.int, np.float)

    WARPING_RECORD_FIELDS = ('walker_idx', 'target_idx', 'weight')

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ()
    PROGRESS_SHAPES = ()
    PROGRESS_DTYPES = ()

    PROGRESS_RECORD_FIELDS = ()


    DISCONTINUITY_TARGET_IDXS = (Ellipsis)
    """Specifies which 'target_idxs' values are considered discontinuous targets.

    Values are either integer indices, Ellipsis (indicating all
    possible values are discontinuous), or None indicating no possible
    value is discontinuous.

    """

    def __init__(self, initial_states=None,
                 initial_weights=None,
                 ligand_idxs=None,
                 receptor_idxs=None):

        # make sure necessary inputs are given
        assert initial_states is not None, "Must give a set of initial states"
        assert ligand_idxs is not None, "Must give ligand indices"
        assert receptor_idxs is not None, "Must give binding site indices"

        self._initial_states = initial_states
        self._ligand_idxs = ligand_idxs
        self._receptor_idxs = receptor_idxs

        # we want to choose initial states conditional on their
        # initial probability if specified. If not specified assume
        # assume uniform probabilities.
        if initial_weights is None:
            self.initial_weights = [1/len(initial_states) for _ in initial_states]
        else:
            self._initial_weights = initial_weights

    @property
    def initial_states(self):
        return self._initial_states

    @property
    def initial_weights(self):
        return self._initial_weights

    @property
    def ligand_idxs(self):
        return self._ligand_idxs

    @property
    def receptor_idxs(self):
        return self._receptor_idxs


    def _progress(self, walker):

        raise NotImplementedError

    def _warp(self, walker):


        # choose a state randomly from the set of initial states
        target_idx = np.choice(range(len(self.initial_states)), 1,
                                  p=self.initial_weights/np.sum(self.initial_weights))[0]

        warped_state = self.initial_states[target_idx]

        # set the initial state into a new walker object with the same weight
        warped_walker = type(walker)(state=warped_state, weight=walker.weight)

        # the data for the warp
        warp_data = {'target_idx' : np.array([target_idx]),
                     'weight' : np.array([walker.weight])}

        return warped_walker, warp_data


    def _update_bc(self, new_walkers, warp_data, progress_data, cycle):

        # do nothing by default
        return []


    def warp_walkers(self, walkers, cycle):

        new_walkers = []

        # sporadic, zero or many records per call
        warp_data = []
        bc_data = []

        # continual, one record per call
        progress_data = defaultdict(list)

        for walker_idx, walker in enumerate(walkers):

            # check if it is unbound, also gives the progress record
            to_warp, walker_progress_data = self._progress(walker)

            # add that to the progress data record
            for key, value in walker_progress_data.items():
                progress_data[key].append(value)

            # if the walker is meets the requirements for warping warp
            # it
            if to_warp:
                # warp the walker
                warped_walker, walker_warp_data = self._warp(walker)

                # add the walker idx to the walker warp record
                walker_warp_data['walker_idx'] = np.array([walker_idx])

                # save warped_walker in the list of new walkers to return
                new_walkers.append(warped_walker)

                # save the instruction record of the walker
                warp_data.append(walker_warp_data)

                logging.info('WARP EVENT observed at {}'.format(cycle))
                logging.info('Warped Walker Weight = {}'.format(
                    walker_warp_data['weight']))

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
        """ """

        # if it is Ellipsis then all possible values are discontinuous
        if cls.DISCONTINUITY_TARGET_IDXS is Ellipsis:
            return True

        # if it is None then all possible values are discontinuous
        elif cls.DISCONTINUITY_TARGET_IDXS is None:
            return False

        # otherwise it will have a tuple of indices for the
        # target_idxs that are discontinuous targets
        elif warping_record[2] in cls.DISCONTINUITY_TARGET_IDXS:
            return True

        # otherwise it wasn't a discontinuous target
        else:
            return False


class RebindingBC(ReceptorBC):
    """ """

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ReceptorBC.BC_FIELDS + ('native_rmsd_cutoff', )
    BC_SHAPES = ReceptorBC.BC_SHAPES + ((1,), )
    BC_DTYPES = ReceptorBC.BC_DTYPES + (np.float, )

    BC_RECORD_FIELDS = ReceptorBC.BC_RECORD_FIELDS + ('native_rmsd_cutoff', )

    # warping (sporadic)
    WARPING_FIELDS = ReceptorBC.WARPING_FIELDS + ()
    WARPING_SHAPES = ReceptorBC.WARPING_SHAPES + ()
    WARPING_DTYPES = ReceptorBC.WARPING_DTYPES + ()

    WARPING_RECORD_FIELDS = ReceptorBC.WARPING_RECORD_FIELDS + ()

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ReceptorBC.PROGRESS_FIELDS + ('native_rmsd',)
    PROGRESS_SHAPES = ReceptorBC.PROGRESS_SHAPES + (Ellipsis,)
    PROGRESS_DTYPES = ReceptorBC.PROGRESS_DTYPES + (np.float,)

    PROGRESS_RECORD_FIELDS = ReceptorBC.PROGRESS_RECORD_FIELDS + ('native_rmsd', )

    def __init__(self, initial_states=None,
                 native_state=None,
                 initial_weights=None,
                 cutoff_rmsd=0.2,
                 ligand_idxs=None,
                 binding_site_idxs=None):

        super().__init__(initial_states=initial_states,
                         initial_weights=initial_weights,
                         ligand_idxs=ligand_idxs,
                         receptor_idxs=binding_site_idxs
                         )

        # test inputs
        assert native_state is not None, "Must give a native state"
        assert type(cutoff_distance) is float

        # save attributes
        self._native_state = native_state
        self._cutoff_rmsd = cutoff_rmsd

    @property
    def native_state(self):
        return self._native_state

    @property
    def cutoff_rmsd(self):
        return self._cutoff_rmsd

    @property
    def binding_site_idxs(self):
        return self._receptor_idxs

    def _check_boundaries(self, walker):
        """

        Parameters
        ----------
        walker

        Returns
        -------

        """

        # first recenter the ligand and the receptor in the walker
        box_lengths, box_angles = box_vectors_to_lengths_angles(walker.state['box_vectors'])
        rece_walker_pos = recenter_pair(walker.state['positions'], box_lengthsm
                                        self.binding_site_idxs, self.ligand_idxs)

        # superimpose the walker state positions over the native state
        # matching the binding site indices only
        sup_walker = superimpose(self.native_state['positions'], rece_walker_pos,
                                 idxs=self.binding_site_idxs)

        # calculate the rmsd of the walker ligand (superimposed
        # according to the binding sites) to the native state ligand
        native_rmsd = calc_rmsd(self.native_state['positions'], rece_walker_pos,
                                idxs=self.ligand_idxs)

        # test to see if the ligand is re-bound
        rebound = False
        if native_rmsd <= self.cutoff_rmsd:
            rebound = True

        boundary_data = {'native_rmsd' : native_rmsd}

        return rebound, progress_data


class UnbindingBC(BoundaryConditions):
    """Boundary condition for ligand unbinding.

    Walkers will be warped (discontinuously) if all atoms in the
    ligand are at least a certain distance away from the atoms in the
    receptor (i.e. the min-min of ligand-receptor distances > cutoff).

    Warping will replace the walker state with the initial state given
    as a parameter to this class.

    Also reports on the progress of that min-min for each walker.

    """

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ReceptorBC.BC_FIELDS + ('boundary_distance', )
    """Records for the state of this record group.

    Only occurs at the start of the simulation and just reports on the
    min-min cutoff distance.

    See Also
    --------
    boundary_conditions.boundary.BC_FIELDS : For explanation of format.
    """

    BC_SHAPES = ReceptorBC.BC_SHAPES + ((1,), )
    """Shapes of record group features.

    See Also
    --------
    boundary_conditions.boundary.BC_SHAPES : For explanation of format.

    """

    BC_DTYPES = ReceptorBC.BC_DTYPES + (np.float, )
    """Datatypes of record group features.

    See Also
    --------
    boundary_conditions.boundary.BC_DTYPES : For explanation of format.

    """

    BC_RECORD_FIELDS = ReceptorBC.BC_RECORD_FIELDS + ('boundary_distance', )
    """Fields included in truncated record group.

    See Also
    --------
    boundary_conditions.boundary.BC_RECORD_FIELDS : For explanation of format.

    """

    # warping (sporadic)
    WARPING_FIELDS = ReceptorBC.WARPING_FIELDS + ()
    """Records for the state of this record group.

    The 'walker_idx' is the index of the walker that was warped and
    'weight' is the weight of that walker.

    The 'target_idx' specifies the target of the warping, which is
    always 0 since all warped walkers have their state replaced with
    the initial state.

    See Also
    --------
    boundary_conditions.boundary.WARPING_FIELDS : For explanation of format.

    """

    WARPING_SHAPES = ReceptorBC.WARPING_SHAPES + ()
    """Shapes of record group features.

    See Also
    --------
    boundary_conditions.boundary.WARPING_SHAPES : For explanation of format.

    """

    WARPING_DTYPES = ReceptorBC.WARPING_DTYPES + ()
    """Datatypes of record group features.

    See Also
    --------
    boundary_conditions.boundary.WARPING_DTYPES : For explanation of format.

    """

    WARPING_RECORD_FIELDS = ReceptorBC.WARPING_RECORD_FIELDS + ()
    """Fields included in truncated record group.

    See Also
    --------
    boundary_conditions.boundary.WARPING_RECORD_FIELDS : For explanation of format.

    """

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ReceptorBC.PROGRESS_FIELDS + ('min_distances',)
    """Records for the state of this record group.

    The 'min_distances' field reports on the min-min ligand-receptor
    distance for each walker.

    See Also
    --------
    boundary_conditions.boundary.PROGRESS_FIELDS : For explanation of format.

    """

    PROGRESS_SHAPES = ReceptorBC.PROGRESS_SHAPES + (Ellipsis,)
    """Shapes of record group features.

    See Also
    --------
    boundary_conditions.boundary.PROGRESS_SHAPES : For explanation of format.

    """

    PROGRESS_DTYPES = ReceptorBC.PROGRESS_DTYPES + (np.float,)
    """Datatypes of record group features.

    See Also
    --------
    boundary_conditions.boundary.PROGRESS_DTYPES : For explanation of format.

    """

    PROGRESS_RECORD_FIELDS = ReceptorBC.PROGRESS_RECORD_FIELDS + ('min_distances', )
    """Fields included in truncated record group.

    See Also
    --------
    boundary_conditions.boundary.PROGRESS_RECORD_FIELDS : For explanation of format.

    """

    def __init__(self, initial_state=None,
                 cutoff_distance=1.0,
                 topology=None,
                 ligand_idxs=None,
                 receptor_idxs=None):
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
            A JSON string of topology.

        ligand_idxs : list of int
           Indices of the atoms in the topology that correspond to the ligands.

        receptor_idxs : list of int
           Indices of the atoms in the topology that correspond to the
           receptor for the ligand.

        Raises
        ------
        AssertionError
            If any of the following are not provided: initial_state, topology,
            ligand_idxs, receptor_idxs

        AssertionError
            If the cutoff distance is not a float.

        Warnings
        --------
        The 'initial_state' should be the initial state of your
        simulation for proper non-equilibrium simulations.

        Notes
        -----

        The topology argument is necessary due to an implementation
        detail that uses mdtraj and may not be required in the future.

        """

        # since the super class can handle multiple initial states we
        # wrap the single initial state to a list.
        super().__init__(initial_states=[initial_state],
                         ligand_idxs=ligand_idxs,
                         receptor_idxs=receptor_idxs)

        # test input
        assert topology is not None, "Must give a reference topology"
        assert type(cutoff_distance) is float

        self._cutoff_distance = cutoff_distance
        self._topology = topology

    def _calc_min_distance(self, walker):
        """Min-min distance for a walker.

        Parameters
        ----------
        walker

        Returns
        -------
        min_distance : float

        """

        cell_lengths, cell_angles = box_vectors_to_lengths_angles(walker.state['box_vectors'])

        # convert the json topology to an mdtraj one
        mdj_top = json_to_mdtraj_topology(self._topology)

        # make a traj out of it so we can calculate distances through
        # the periodic boundary conditions
        walker_traj = mdj.Trajectory(walker.state['positions'],
                                     topology=mdj_top,
                                     unitcell_lengths=cell_lengths,
                                     unitcell_angles=cell_angles)

        # calculate the distances through periodic boundary conditions
        # and get hte minimum distance
        min_distance = np.min(mdj.compute_distances(walker_traj,
                                                    it.product(self.ligand_idxs,
                                                               self.receptor_idxs)))
        return min_distance

    def _progress(self, walker):
        """Calculate whether a walker has unbound and also provide a
        dictionary for a single walker in the progress records.

        Parameters
        ----------
        walker

        Returns
        -------
        unbound : bool
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

        progress_data = {'min_distances' : min_distance}

        return unbound, progress_data

    def _update_bc(self, new_walkers, warp_data, progress_data, cycle):
        """

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
            return [{'boundary_distance' : np.array([self._cutoff_distance]),},]
        else:
            return []
