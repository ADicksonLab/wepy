import sys
import itertools as it
from collections import defaultdict
from copy import copy
import logging

import numpy as np
import numpy.linalg as la

import mdtraj as mdj

from wepy.boundary_conditions.boundary import BoundaryConditions

class UnbindingBC(BoundaryConditions):

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ('boundary_distance', )
    BC_SHAPES = ((1,), )
    BC_DTYPES = (np.float, )

    BC_RECORD_FIELDS = ('boundary_distance', )

    # warping (sporadic)
    WARPING_FIELDS = ('walker_idx', 'target_idx', 'weight')
    WARPING_SHAPES = ((1,), (1,), (1,))
    WARPING_DTYPES = (np.int, np.int, np.float)

    WARPING_RECORD_FIELDS = ('walker_idx', 'target_idx', 'weight')

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ('min_distances',)
    PROGRESS_SHAPES = (Ellipsis,)
    PROGRESS_DTYPES = (np.float,)

    PROGRESS_RECORD_FIELDS = ('min_distances', )


    # for boundary conditions that warp things around certain targets
    # mayb not introduce discontiuities, these target idxs do
    DISCONTINUITY_TARGET_IDXS = (0,)

    def __init__(self, initial_state=None,
                 cutoff_distance=1.0,
                 topology=None,
                 ligand_idxs=None,
                 receptor_idxs=None):

        super().__init__()

        # test input
        assert initial_state is not None, "Must give an initial state"
        assert topology is not None, "Must give a reference topology"
        assert ligand_idxs is not None
        assert receptor_idxs is not None
        assert type(cutoff_distance) is float

        self.initial_state = initial_state
        self.cutoff_distance = cutoff_distance
        self.topology = topology

        self.ligand_idxs = ligand_idxs
        self.receptor_idxs = receptor_idxs

    def _calc_angle(self, v1, v2):
        return np.degrees(np.arccos(np.dot(v1, v2)/(la.norm(v1) * la.norm(v2))))

    def _calc_length(self, v):
        return la.norm(v)

    def _calc_min_distance(self, walker):
        # convert box_vectors to angles and lengths for mdtraj
        # calc box length
        cell_lengths = np.array([[self._calc_length(v) for v in walker.state['box_vectors']]])

        # TODO order of cell angles
        # calc angles
        cell_angles = np.array([[self._calc_angle(walker.state['box_vectors'][i],
                                                 walker.state['box_vectors'][j])
                                 for i, j in [(0,1), (1,2), (2,0)]]])

        # make a traj out of it so we can calculate distances through
        # the periodic boundary conditions
        walker_traj = mdj.Trajectory(walker.state['positions'],
                                     topology=self.topology,
                                     unitcell_lengths=cell_lengths,
                                     unitcell_angles=cell_angles)

        # calculate the distances through periodic boundary conditions
        # and get hte minimum distance
        min_distance = np.min(mdj.compute_distances(walker_traj,
                                                    it.product(self.ligand_idxs,
                                                               self.receptor_idxs)))
        return min_distance

    def progress(self, walker):

        min_distance = self._calc_min_distance(walker)

        # test to see if the ligand is unbound
        unbound = False
        if min_distance >= self.cutoff_distance:
            unbound = True

        progress_data = {'min_distances' : min_distance}

        return unbound, progress_data

    def warp(self, walker):

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

    def update_bc(self, new_walkers, warp_data, progress_data, cycle):

        # TODO just for testing if this works. only report a record on
        # the first cycle which gives the distance at which walkers
        # are warped
        if cycle == 0:
            return [{'boundary_distance' : np.array([self.cutoff_distance]),},]
        else:
            return []

    def warp_walkers(self, walkers, cycle):

        new_walkers = []

        # sporadic, zero or many records per call
        warp_data = []
        bc_data = []

        # continual, one record per call
        progress_data = defaultdict(list)

        for walker_idx, walker in enumerate(walkers):
            # check if it is unbound, also gives the minimum distance
            # between guest and host
            unbound, walker_progress_data = self.progress(walker)

            # add that to the progress data record
            for key, value in walker_progress_data.items():
                progress_data[key].append(value)

            # if the walker is unbound we need to warp it
            if unbound:
                # warp the walker
                warped_walker, walker_warp_data = self.warp(walker)

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
        bc_data = self.update_bc(new_walkers, warp_data, progress_data, cycle)

        return new_walkers, warp_data, bc_data, progress_data

    @classmethod
    def warping_discontinuity(cls, warping_record):
        """Given a warping record returns either True for a discontiuity
        occured or False if a discontinuity did not occur."""

        # the target_idxs are one of the discontinuous targets
        if warping_record[2] in cls.DISCONTINUITY_TARGET_IDXS:
            return True
        else:
            return False
