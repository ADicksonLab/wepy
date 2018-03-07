import json

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

