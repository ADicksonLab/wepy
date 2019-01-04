"""Classes for receptor based boundary conditions."""

from collections import defaultdict
import logging

import numpy as np

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
        target_idx = choice(range(len(self.initial_states)), 1,
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
