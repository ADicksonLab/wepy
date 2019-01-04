import sys
import itertools as it
from collections import defaultdict
from random import random
import logging

import numpy as np
import numpy.linalg as la
from numpy.random import choice

import mdtraj as mdj

from wepy.boundary_conditions.boundary import BoundaryConditions
from wepy.resampling.distances.receptor import RebindingDistance

class RebindingBC(BoundaryConditions):
    """ """

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

    def __init__(self, initial_states=None,
                 native_state=None,
                 initial_weights=None,
                 cutoff_distance=0.2,
                 ligand_idxs=None,
                 binding_site_idxs=None):

        # make sure necessary inputs are given
        assert initial_states is not None, "Must give a set of initial states"
        assert native_state is not None, "Must give a native state"
        assert ligand_idxs is not None, "Must give ligand indices"
        assert binding_site_idxs is not None, "Must give binding site indices"

        assert type(cutoff_distance) is float

        # save attributes
        self._initial_states = initial_states
        self._native_state = native_state
        self._cutoff_distance = cutoff_distance
        self._ligand_idxs = ligand_idxs
        self._bs_idxs = binding_site_idxs

        # the distance metric to use to calculate for each
        # walker. This is dependent on the native state
        self._distance_metric = RebindingDistance(ligand_idxs,
                                                  binding_site_idxs,
                                                  native_state)

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
    def native_state(self):
        return self._native_state

    @property
    def cutoff_distance(self):
        return self._cutoff_distance

    @property
    def ligand_idxs(self):
        return self._ligand_idxs

    @property
    def binding_site_idxs(self):
        return self._bs_idxs

    @property
    def distance_metric(self):
        return self._distance_metric

    @property
    def initial_weights(self):
        return self._initial_weights

    def _check_boundaries(self, nat_rmsd):
        """

        Parameters
        ----------
        nat_rmsd :
            

        Returns
        -------

        """

        # test to see if the ligand is re-bound
        rebound = False
        if nat_rmsd <= self.cutoff_distance:
            rebound = True

        boundary_data = {'nat_rmsd' : nat_rmsd}

        return rebound, boundary_data

    def _warp(self, walker, cycle):
        """

        Parameters
        ----------
        walker :
            
        cycle :
            

        Returns
        -------

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

        native_rmsds = [self._distance_metric.distance(walker) for walker in walkers]

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
