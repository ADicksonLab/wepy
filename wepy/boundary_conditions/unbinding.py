import sys
import itertools as it
from collections import defaultdict

import numpy as np
import numpy.linalg as la

import mdtraj as mdj

from wepy.boundary_conditions.boundary import BoundaryConditions


class UnbindingBC(BoundaryConditions):

    WARP_INSTRUCT_DTYPE = np.dtype([('target', int)])

    WARP_AUX_DTYPES = {'passage_time' :  np.float}
    WARP_AUX_SHAPES = {'passage_time' : (1,)}

    def __init__(self, initial_state=None,
                 cutoff_distance=1.0,
                 topology=None,
                 ligand_idxs=None,
                 binding_site_idxs=None):

        # test input
        assert initial_state is not None, "Must give an initial state"
        assert topology is not None, "Must give a reference topology"
        assert ligand_idxs is not None
        assert binding_site_idxs is not None
        assert type(cutoff_distance) is float

        self.initial_state = initial_state
        self.cutoff_distance = cutoff_distance
        self.topology = topology

        self.ligand_idxs = ligand_idxs
        self.binding_site_idxs = binding_site_idxs

    def _calc_angle(self, v1, v2):
        return np.degrees(np.arccos(np.dot(v1, v2)/(la.norm(v1) * la.norm(v2))))

    def _calc_length(self, v):
        return la.norm(v)


    def _pos_to_array(self, positions):
        n_atoms = self.topology.n_atoms

        xyz = np.zeros((1, n_atoms, 3))

        for i in range(n_atoms):
            xyz[0,i,:] = ([positions[i]._value[0],
                           positions[i]._value[1],
                           positions[i]._value[2]])
        return xyz

    def _calc_min_distance(self, walker):
        # convert box_vectors to angles and lengths for mdtraj
        # calc box length
        cell_lengths = np.array([[self._calc_length(v._value) for v in walker.box_vectors]])

        # TODO order of cell angles
        # calc angles
        cell_angles = np.array([[self._calc_angle(walker.box_vectors._value[i],
                                                 walker.box_vectors._value[j])
                                 for i, j in [(0,1), (1,2), (2,0)]]])

        # make a traj out of it so we can calculate distances through
        # the periodic boundary conditions
        walker_traj = mdj.Trajectory(self._pos_to_array(walker.positions[0:self.topology.n_atoms]),
                                     topology=self.topology,
                                     unitcell_lengths=cell_lengths,
                                     unitcell_angles=cell_angles)

        # calculate the distances through periodic boundary conditions
        # and get hte minimum distance
        min_distance = np.min(mdj.compute_distances(walker_traj,
                                                    it.product(self.ligand_idxs,
                                                               self.binding_site_idxs)))
        return min_distance

    def check_boundaries(self, walker):

        min_distance = self._calc_min_distance(walker)

        # test to see if the ligand is unbound
        unbound = False
        if min_distance >= self.cutoff_distance:
            unbound = True

        boundary_data = {'min_distance' : min_distance}

        return unbound, boundary_data

    def warp(self, walker):

        # we always start at the initial state
        warped_state = self.initial_state
        # set the initial state into a new walker object with the same
        # weight
        warped_walker = type(walker)(state=warped_state, weight=walker.weight)
        # thus there is only one record
        warp_record = (0,)
        # collect the passage time
        warp_data = {'passage_time' : np.array([walker.time_value()])}

        return warped_walker, warp_record, warp_data

    def warp_walkers(self, walkers, debug_prints=False):

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
            unbound, boundary_data = self.check_boundaries(walker)

            # add boundary data for this walker
            for key, value in boundary_data.items():
                cycle_boundary_data[key].append(value)

            # if the walker is unbound we need to warp it
            if unbound:
                # warp the walker
                warped_walker, warp_record, warp_data = self.warp(walker)

                # save it in the list of new walkers to return
                new_walkers.append(self.initial_state)

                # save the record of the walker
                warped_walkers_records.append( (walker_idx, warp_record) )

                # save warp data
                for key, value in warp_data.items():
                    cycle_warp_data[key].append(value)

                # DEBUG
                if debug_prints:
                    sys.stdout.write('EXIT POINT observed at {} s\n'.format(warp_data['passage_time']))

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
