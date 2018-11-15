import json
import warnings

import numpy as np



def traj_box_vectors_to_lengths_angles(traj_box_vectors):
    """Convert box vectors for multiple 'frames' (a 'trajectory') to box lengths and angles."""

    traj_unitcell_lengths = []
    for basis in traj_box_vectors:
        traj_unitcell_lengths.append(np.array([np.linalg.norm(frame_v) for frame_v in basis]))

    traj_unitcell_lengths = np.array(traj_unitcell_lengths)

    traj_unitcell_angles = []
    for vs in traj_box_vectors:

        angles = np.array([np.degrees(
                            np.arccos(np.dot(vs[i], vs[j])/
                                      (np.linalg.norm(vs[i]) * np.linalg.norm(vs[j]))))
                           for i, j in [(0,1), (1,2), (2,0)]])

        traj_unitcell_angles.append(angles)

    traj_unitcell_angles = np.array(traj_unitcell_angles)

    return traj_unitcell_lengths, traj_unitcell_angles

def box_vectors_to_lengths_angles(box_vectors):
    """Convert box vectors for a single 'frame' to lengths and angles."""

    # calculate the lengths of the vectors through taking the norm of
    # them
    unitcell_lengths = []
    for basis in box_vectors:
        unitcell_lengths.append(np.linalg.norm(basis))
    unitcell_lengths = np.array(unitcell_lengths)

    # calculate the angles for the vectors
    unitcell_angles = np.array([np.degrees(
                        np.arccos(np.dot(box_vectors[i], box_vectors[j])/
                                  (np.linalg.norm(box_vectors[i]) * np.linalg.norm(box_vectors[j]))))
                       for i, j in [(0,1), (1,2), (2,0)]])

    return unitcell_lengths, unitcell_angles


def json_top_atom_count(json_str):
    """Count the number of atoms in a JSON topology used by wepy HDF5."""

    top_d = json.loads(json_str)
    atom_count = 0
    atom_count = 0
    for chain in top_d['chains']:
        for residue in chain['residues']:
            atom_count += len(residue['atoms'])

    return atom_count

def json_top_subset(json_str, atom_idxs):
    """Given a JSON topology and atom indices from that topology returns
    another JSON topology which is a subset of the first, preserving
    the topology between remaining atoms. The atoms will be ordered in
    the order in which the indices are given.

    """

    # cast so we can use the list index method
    atom_idxs = list(atom_idxs)

    # do checks on the atom idxs

    # no duplicates
    assert len(set(atom_idxs)) == len(atom_idxs), "duplicate atom indices"

    top = json.loads(json_str)

    # the dictionaries for each thing indexed by their old index
    atom_data_ds = {}
    residue_data_ds = {}

    # mapping of old atom indices to the residue data they belong to
    atom_res_idxs = {}
    res_chain_idxs = {}

    # go through and collect data on the atoms and convert indices to
    # the new ones
    for chain in top['chains']:

        for residue in chain['residues']:
            res_chain_idxs[residue['index']] = chain['index']

            # create the data dict for this residue
            residue_data_ds[residue['index']] = residue



            for atom in residue['atoms']:
                atom_res_idxs[atom['index']] = residue['index']

                # if the current atom's index is in the selection
                if atom['index'] in atom_idxs:

                    # we add this to the mapping by getting the index
                    # of the atom in subset
                    new_idx = atom_idxs.index(atom['index'])

                    # save the atom attributes
                    atom_data_ds[atom['index']] = atom

    # initialize the data structure for the topology subset
    top_subset = {'chains' : [],
                  'bonds' : []}

    residue_idx_map = {}
    chain_idx_map = {}

    old_to_new_atoms = {}

    # initialize the new indexing of the chains and residues
    new_res_idx_counter = 0
    new_chain_idx_counter = 0
    # now in the new order go through and create the topology
    for new_atom_idx, old_atom_idx in enumerate(atom_idxs):

        old_to_new_atoms[old_atom_idx] = new_atom_idx

        atom_data = atom_data_ds[old_atom_idx]

        # get the old residue index
        old_res_idx = atom_res_idxs[old_atom_idx]

        # since the datastrucutre is hierarchical starting with the
        # chains and residues we work our way back up craeting these
        # if neceessary for this atom, once this is taken care of we
        # will add the atom to the data structure
        if old_res_idx not in residue_idx_map:
            residue_idx_map[old_res_idx] = new_res_idx_counter
            new_res_idx_counter += 1

            # do the same but for the chain
            old_chain_idx = res_chain_idxs[old_res_idx]

            # make it if necessary
            if old_chain_idx not in chain_idx_map:
                chain_idx_map[old_chain_idx] = new_chain_idx_counter
                new_chain_idx_counter += 1

                # and add the chain to the topology
                new_chain_idx = chain_idx_map[old_chain_idx]
                top_subset['chains'].append({'index' : new_chain_idx,
                                             'residues' : []})

            # add the new index to the dats dict for the residue
            res_data = residue_data_ds[old_res_idx]
            res_data['index'] = residue_idx_map[old_res_idx]
            # clear the atoms
            res_data['atoms'] = []

            # add the reside to the chain idx
            new_chain_idx = chain_idx_map[old_chain_idx]
            top_subset['chains'][new_chain_idx]['residues'].append(res_data)

        # now that (if) we have made the necessary chains and residues
        # for this atom we replace the atom index with the new index
        # and add it to the residue
        new_res_idx = residue_idx_map[old_res_idx]
        new_chain_idx = chain_idx_map[res_chain_idxs[old_res_idx]]

        atom_data['index'] = new_atom_idx

        top_subset['chains'][new_chain_idx]['residues'][new_res_idx]['atoms'].append(atom_data)

    # then translate the atom indices in the bonds
    new_bonds = []
    for bond_atom_idxs in top['bonds']:

        if all([True if a_idx in old_to_new_atoms else False
                for a_idx in bond_atom_idxs]):
            new_bond_atom_idxs = [old_to_new_atoms[a_idx] for a_idx in bond_atom_idxs
                                  if a_idx in old_to_new_atoms]
            new_bonds.append(new_bond_atom_idxs)

    top_subset['bonds'] = new_bonds


    return json.dumps(top_subset)



# License applicable to the function 'lengths_and_angles_to_box_vectors'
##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2013 Stanford University and the Authors
#
# Authors: Robert McGibbon
# Contributors:
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################

def lengths_and_angles_to_box_vectors(a_length, b_length, c_length, alpha, beta, gamma):
    """Convert from the lengths/angles of the unit cell to the box
    vectors (Bravais vectors). The angles should be in degrees.

    Parameters
    ----------
    a_length : scalar or np.ndarray
        length of Bravais unit vector **a**
    b_length : scalar or np.ndarray
        length of Bravais unit vector **b**
    c_length : scalar or np.ndarray
        length of Bravais unit vector **c**
    alpha : scalar or np.ndarray
        angle between vectors **b** and **c**, in degrees.
    beta : scalar or np.ndarray
        angle between vectors **c** and **a**, in degrees.
    gamma : scalar or np.ndarray
        angle between vectors **a** and **b**, in degrees.

    Returns
    -------
    a : np.ndarray
        If the inputs are scalar, the vectors will one dimesninoal (length 3).
        If the inputs are one dimension, shape=(n_frames, ), then the output
        will be (n_frames, 3)
    b : np.ndarray
        If the inputs are scalar, the vectors will one dimesninoal (length 3).
        If the inputs are one dimension, shape=(n_frames, ), then the output
        will be (n_frames, 3)
    c : np.ndarray
        If the inputs are scalar, the vectors will one dimesninoal (length 3).
        If the inputs are one dimension, shape=(n_frames, ), then the output
        will be (n_frames, 3)

    Examples
    --------
    >>> import numpy as np
    >>> result = lengths_and_angles_to_box_vectors(1, 1, 1, 90.0, 90.0, 90.0)

    Notes
    -----
    This code is adapted from gyroid, which is licensed under the BSD
    http://pythonhosted.org/gyroid/_modules/gyroid/unitcell.html
    """
    if np.all(alpha < 2*np.pi) and np.all(beta < 2*np.pi) and np.all(gamma < 2*np.pi):
        warnings.warn('All your angles were less than 2*pi. Did you accidentally give me radians?')

    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180
    gamma = gamma * np.pi / 180

    a = np.array([a_length, np.zeros_like(a_length), np.zeros_like(a_length)])
    b = np.array([b_length*np.cos(gamma), b_length*np.sin(gamma), np.zeros_like(b_length)])
    cx = c_length*np.cos(beta)
    cy = c_length*(np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c_length*c_length - cx*cx - cy*cy)
    c = np.array([cx,cy,cz])

    if not a.shape == b.shape == c.shape:
        raise TypeError('Shape is messed up.')

    # Make sure that all vector components that are _almost_ 0 are set exactly
    # to 0
    tol = 1e-6
    a[np.logical_and(a>-tol, a<tol)] = 0.0
    b[np.logical_and(b>-tol, b<tol)] = 0.0
    c[np.logical_and(c>-tol, c<tol)] = 0.0

    return a.T, b.T, c.T
