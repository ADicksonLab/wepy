"""Boundary conditions for ligand unbinding simulations."""

import sys
import itertools as it
from collections import defaultdict
from copy import copy
import logging

import numpy as np
import numpy.linalg as la

import mdtraj as mdj

from wepy.boundary_conditions.boundary import BoundaryConditions
from wepy.util.mdtraj import json_to_mdtraj_topology

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
    BC_FIELDS = ('boundary_distance', )
    """Records for the state of this record group.

    Only occurs at the start of the simulation and just reports on the
    min-min cutoff distance.

    See Also
    --------
    boundary_conditions.boundary.BC_FIELDS : For explanation of format.
    """

    BC_SHAPES = ((1,), )
    """Shapes of record group features.

    See Also
    --------
    boundary_conditions.boundary.BC_SHAPES : For explanation of format.

    """

    BC_DTYPES = (np.float, )
    """Datatypes of record group features.

    See Also
    --------
    boundary_conditions.boundary.BC_DTYPES : For explanation of format.

    """

    BC_RECORD_FIELDS = ('boundary_distance', )
    """Fields included in truncated record group.

    See Also
    --------
    boundary_conditions.boundary.BC_RECORD_FIELDS : For explanation of format.

    """

    # warping (sporadic)
    WARPING_FIELDS = ('walker_idx', 'target_idx', 'weight')
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

    WARPING_SHAPES = ((1,), (1,), (1,))
    """Shapes of record group features.

    See Also
    --------
    boundary_conditions.boundary.WARPING_SHAPES : For explanation of format.

    """

    WARPING_DTYPES = (np.int, np.int, np.float)
    """Datatypes of record group features.

    See Also
    --------
    boundary_conditions.boundary.WARPING_DTYPES : For explanation of format.

    """

    WARPING_RECORD_FIELDS = ('walker_idx', 'target_idx', 'weight')
    """Fields included in truncated record group.

    See Also
    --------
    boundary_conditions.boundary.WARPING_RECORD_FIELDS : For explanation of format.

    """

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ('min_distances',)
    """Records for the state of this record group.

    The 'min_distances' field reports on the min-min ligand-receptor
    distance for each walker.

    See Also
    --------
    boundary_conditions.boundary.PROGRESS_FIELDS : For explanation of format.

    """

    PROGRESS_SHAPES = (Ellipsis,)
    """Shapes of record group features.

    See Also
    --------
    boundary_conditions.boundary.PROGRESS_SHAPES : For explanation of format.

    """

    PROGRESS_DTYPES = (np.float,)
    """Datatypes of record group features.

    See Also
    --------
    boundary_conditions.boundary.PROGRESS_DTYPES : For explanation of format.

    """

    PROGRESS_RECORD_FIELDS = ('min_distances', )
    """Fields included in truncated record group.

    See Also
    --------
    boundary_conditions.boundary.PROGRESS_RECORD_FIELDS : For explanation of format.

    """


    # for boundary conditions that warp things around certain targets
    # mayb not introduce discontiuities, these target idxs do
    DISCONTINUITY_TARGET_IDXS = (0,)
    """Specifies which 'target_idxs' values are considered discontinuous targets."""

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

    def _calc_angle(self, v1, v2):
        """Calculates the angle between two vectors.

        Parameters
        ----------
        v1 : arraylike of rank 1
        v2 : arraylike of rank 1

        Returns
        -------

        angle : float
            In degrees.

        """
        return np.degrees(np.arccos(np.dot(v1, v2)/(la.norm(v1) * la.norm(v2))))

    def _calc_length(self, v):
        """

        Parameters
        ----------
        v : arraylike of rank 1

        Returns
        -------
        length : float

        """
        return la.norm(v)

    def _calc_min_distance(self, walker):
        """Min-min distance for a walker.

        Parameters
        ----------
        walker

        Returns
        -------
        min_distance : float

        """
        # convert box_vectors to angles and lengths for mdtraj
        # calc box length
        cell_lengths = np.array([[self._calc_length(v) for v in walker.state['box_vectors']]])

        # TODO order of cell angles
        # calc angles
        cell_angles = np.array([[self._calc_angle(walker.state['box_vectors'][i],
                                                 walker.state['box_vectors'][j])
                                 for i, j in [(0,1), (1,2), (2,0)]]])

        # convert the json topology to an mdtraj one
        mdj_top = json_to_mdtraj_topology(self.topology)

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
        if min_distance >= self.cutoff_distance:
            unbound = True

        progress_data = {'min_distances' : min_distance}

        return unbound, progress_data

    def _warp(self, walker):
        """Perform the warping on a walker. Replaces its state
        with the initial_state.

        Parameters
        ----------
        walker

        Returns
        -------
        warped_walker
           Walker with initial_state state

        warp_data : dict
           Dictionary-style record for this warping event.

        """

        # we always start at the initial state
        warped_state = self.initial_state

        # set the initial state into a new walker object with the same
        # weight
        warped_walker = type(walker)(state=warped_state, weight=walker.weight)

        # thus there is only value for a record
        target_idx = 0

        # the data for the warp
        warp_data = {'target_idx' : np.array([target_idx]),
                     'weight' : np.array([walker.weight])}

        return warped_walker, warp_data

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
            return [{'boundary_distance' : np.array([self.cutoff_distance]),},]
        else:
            return []

    def warp_walkers(self, walkers, cycle):
        # documented in superclass

        new_walkers = []

        # sporadic, zero or many records per call
        warp_data = []
        bc_data = []

        # continual, one record per call
        progress_data = defaultdict(list)

        for walker_idx, walker in enumerate(walkers):
            # check if it is unbound, also gives the minimum distance
            # between guest and host
            unbound, walker_progress_data = self._progress(walker)

            # add that to the progress data record
            for key, value in walker_progress_data.items():
                progress_data[key].append(value)

            # if the walker is unbound we need to warp it
            if unbound:
                # warp the walker
                warped_walker, walker_warp_data = self._warp(walker)

                # add the walker idx to the walker warp record
                walker_warp_data['walker_idx'] = np.array([walker_idx])

                # save warped_walker in the list of new walkers to return
                new_walkers.append(warped_walker)

                # save the instruction record of the walker
                warp_data.append(walker_warp_data)

                logging.info('EXIT POINT observed at {}'.format(cycle))
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
        # documented in superclass

        # the target_idxs are one of the discontinuous targets
        if warping_record[2] in cls.DISCONTINUITY_TARGET_IDXS:
            return True
        else:
            return False
