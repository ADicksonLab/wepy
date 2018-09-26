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
from wepy.resampling.distances.openmm import OpenMMRebindingDistance

class RebindingBC(BoundaryConditions):

    WARP_INSTRUCT_DTYPE = np.dtype([('target', int)])

    WARP_AUX_DTYPES = {'cycle' : np.int, 'passage_time' :  np.float, 'warped_walker_weight' : np.float}
    WARP_AUX_SHAPES = {'cycle' : (1,), 'passage_time' : (1,), 'warped_walker_weight' : (1,)}

    def __init__(self, initial_states=None,
                 initial_weights=None,
                 cutoff_distance=0.2,
                 topology=None,
                 ligand_idxs=None,
                 binding_site_idxs=None,
                 comp_xyz=None,
                 alternative_maps=None):

        # test input
        assert initial_states is not None, "Must give a set of initial states"
        assert topology is not None, "Must give a reference topology"
        assert comp_xyz is not None, "Must give coordinates for bound state"
        assert ligand_idxs is not None
        assert binding_site_idxs is not None
        assert type(cutoff_distance) is float

        self.initial_states = initial_states
        if initial_weights is None:
            self.initial_weights = np.array([1] * len(initial_states))
        else:
            self.initial_weights = initial_weights
        self.cutoff_distance = cutoff_distance
        self.topology = topology

        self.native_distance = OpenMMRebindingDistance(topology=topology,
                                                       ligand_idxs=ligand_idxs,
                                                       binding_site_idxs=binding_site_idxs,
                                                       alt_maps=alternative_maps,
                                                       comp_xyz=comp_xyz)

    def check_boundaries(self, nat_rmsd):

        # test to see if the ligand is re-bound
        rebound = False
        if nat_rmsd <= self.cutoff_distance:
            rebound = True

        boundary_data = {'nat_rmsd' : nat_rmsd}

        return rebound, boundary_data

    def warp(self, walker, cycle):

        # choose a state randomly from the set of initial states
        warped_state = choice(self.initial_states, 1, p=self.initial_weights/np.sum(self.initial_weights))[0]

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

        new_walkers = []
        warped_walkers_records = []
        cycle_bc_records = []

        # boundary data is collected for each walker every cycle
        cycle_boundary_data = defaultdict(list)
        # warp data is collected each time a warp occurs
        cycle_warp_data = defaultdict(list)

        native_rmsds = self.native_distance.get_rmsd_native(walkers)
        for walker_idx, walker in enumerate(walkers):
            # check if it is unbound, also gives the minimum distance
            # between guest and host
            rebound, boundary_data = self.check_boundaries(native_rmsds[walker_idx])

            # add boundary data for this walker
            for key, value in boundary_data.items():
                cycle_boundary_data[key].append(value)

            # if the walker is unbound we need to warp it
            if rebound:
                # warp the walker
                warped_walker, warp_record, warp_data = self.warp(walker,cycle)

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
