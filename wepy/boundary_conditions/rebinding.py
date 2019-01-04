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
