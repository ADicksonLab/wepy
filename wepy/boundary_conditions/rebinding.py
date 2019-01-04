import sys
import itertools as it
from collections import defaultdict
from random import random
import logging

import numpy as np
import numpy.linalg as la
from numpy.random import choice

import mdtraj as mdj

from geomm.recentering import recenter_pair
from geomm.superimpose import superimpose
from geomm.rmsd import calc_rmsd

from wepy.boundary_conditions.receptor import ReceptorBC
from wepy.util.util import box_vectors_to_lengths_angles

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
                         )

        # make sure necessary inputs are given
        assert native_state is not None, "Must give a native state"
        assert ligand_idxs is not None, "Must give ligand indices"
        assert binding_site_idxs is not None, "Must give binding site indices"

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

    @property
    def distance_metric(self):
        return self._distance_metric


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

    def _warp(self, walker, cycle):
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

        # choose a state randomly from the set of initial states
        warped_state = choice(self.initial_states, 1,
                              p=self.initial_weights/np.sum(self.initial_weights))[0]

        # set the initial state into a new walker object with the same weight
        warped_walker = type(walker)(state=warped_state, weight=walker.weight)

        # thus there is only one record
        warp_record = (0,)

        # collect the passage time

        # time is returned as an array because it is a feature of the
        # walker, and domain specific. I.e. domain specific values are
        # of type `array` while weights will always be floats in all
        # applications.
        time = walker.time_value()
        warp_data = {'cycle' : np.array([cycle]), 'passage_time' : time,
                     'warped_walker_weight' : np.array([walker.weight])}

        # make the warp data mapping


        return warped_walker, warp_record, warp_data

    def warp_walkers(self, walkers, cycle):
        # docstring in superclass

        new_walkers = []
        warped_walkers_records = []
        cycle_bc_records = []

        # boundary data is collected for each walker every cycle
        cycle_boundary_data = defaultdict(list)
        # warp data is collected each time a warp occurs
        cycle_warp_data = defaultdict(list)



        for walker_idx, walker in enumerate(walkers):
            # check if it is unbound, also gives the minimum distance
            # between guest and host
            rebound, boundary_data = self._check_boundaries(native_rmsds[walker_idx])

            # add boundary data for this walker
            for key, value in boundary_data.items():
                cycle_boundary_data[key].append(value)

            # if the walker is unbound we need to warp it
            if rebound:
                # warp the walker
                warped_walker, warp_record, warp_data = self._warp(walker,cycle)

                # save warped_walker in the list of new walkers to return
                new_walkers.append(warped_walker)

                # save the record of the walker
                warped_walkers_records.append( (walker_idx, warp_record) )

                # save warp data
                for key, value in warp_data.items():
                    cycle_warp_data[key].append(value)

                logging.info('REBINDING observed at {}'.format(
                    warp_data['passage_time']))
                logging.info('Warped Walker Weight = {}'.format(
                    warp_data['warped_walker_weight']))

            # no warping so just return the original walker
            else:
                new_walkers.append(walker)

        # convert aux datas to np.arrays
        for key, value in cycle_warp_data.items():
            cycle_warp_data[key] = np.array(value)
        for key, value in cycle_boundary_data.items():
            cycle_boundary_data[key] = np.array(value)

        return new_walkers, warped_walkers_records, cycle_warp_data, \
                 cycle_bc_records, cycle_boundary_data


    @classmethod
    def warping_discontinuity(cls, warping_record):
        # documented in superclass

        # the target_idxs are one of the discontinuous targets
        if warping_record[2] in cls.DISCONTINUITY_TARGET_IDXS:
            return True
        else:
            return False
