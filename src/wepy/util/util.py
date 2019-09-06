"""Miscellaneous functions needed by wepy."""

import json
import warnings

import numpy as np



def traj_box_vectors_to_lengths_angles(traj_box_vectors):
    """Convert box vectors for multiple 'frames' (a 'trajectory') to box lengths and angles.

    Parameters
    ----------
    traj_box_vectors : arraylike of float of shape (n_frames, n_dims, n_dims)
        Box vector matrices (x, y, z) for a trajectory of frames.

    Returns
    -------
    traj_box_lengths : arraylike of float of shape (n_frames, n_dims)
        The lengths of the box for each frame

    traj_box_angles : arraylike of float of shape (n_frames, n_dims)
        The angles of the box vectors to normal for each frame in degrees.

    """

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
    """Convert box vectors for a single 'frame' to lengths and angles.

    Parameters
    ----------
    box_vectors : arraylike of float of shape (n_dims, n_dims)
        Box vector matrices (x, y, z) for a single frame.

    Returns
    -------
    box_lengths : arraylike of float of shape (n_dims,)
        The lengths of the box

    box_angles : arraylike of float of shape (n_dims,)
        The angles of the box vectors to normal in degrees.

    """

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
    Notes
    -----
    This code is adapted from gyroid, which is licensed under the BSD
    http://pythonhosted.org/gyroid/_modules/gyroid/unitcell.html
    >>> import numpy as np
    >>> result = lengths_and_angles_to_box_vectors(1, 1, 1, 90.0, 90.0, 90.0)
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


def concat_traj_fields(trajs_fields):

    # get the fields
    fields = list(trajs_fields[0].keys())

    cum_traj_fields = {}
    for field in fields:
        cum_traj_fields[field] = np.concatenate(
            [traj_fields[field] for traj_fields in trajs_fields])

    return cum_traj_fields
