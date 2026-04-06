"""Projectors into a pre-trained tICA space.
"""

import logging
import numpy as np

from geomm.grouping import group_pair
from geomm.superimpose import superimpose

from wepy.resampling.projectors.projector import Projector
from wepy.util.util import box_vectors_to_lengths_angles

logger = logging.getLogger(__name__)


def shorten_vecs(disp_vecs, box_lengths):
    """
    Apply minimum-image convention to displacement vectors in an
    cubic/tetragonal/rectangular(?)/orthorhombic periodic box (vectorized). Seems faster.

    Parameters
    ----------
    disp_vecs : array_like, shape (n, 3)
        Displacement vectors[[dx_i, dy_i, dz_i], [dx_j, dy_j, dz_j] ...].
    box_lengths : array_like, shape (3,)
        Box side lengths [Lx, Ly, Lz].

    Returns
    -------
    numpy.ndarray, shape (n, 3)
        Minimum-image displacement vectors.
    """

    X = np.asarray(disp_vecs, dtype=float)      # (n, 3)
    L = np.asarray(box_lengths, dtype=float)   # (3,)

    # Broadcasting: (n,3) / (3,) works automatically

    return X - L * np.round(X / L)
    # this should always work, np.round(0.6) = 1, np.round(-0.6) = -1 and np.round(0.2)=0.


def aligned_frame_for_coord_tica(
    coords,
    ref_coords,
    unitcell_length,
    alignment_idxs,
    pair_idx1,
    pair_idx2,
    important_idxs=None,
    return_full_aligned=False,
):
    """Single-frame version of the calc-feature alignment logic.

    This mirrors the behavior of feature_extraction.aligned_frames for a
    single coordinate frame so the tICA projector does not need to import
    aligned_frames from feature_extraction.py.
    """

    coords = np.asarray(coords)
    ref_coords = np.asarray(ref_coords)
    alignment_idxs = np.asarray(alignment_idxs, dtype=int)
    pair_idx1 = np.asarray(pair_idx1, dtype=int)
    pair_idx2 = np.asarray(pair_idx2, dtype=int)

    if important_idxs is not None:
        important_idxs = np.asarray(important_idxs, dtype=int)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")

    grouped_pos = group_pair(coords, unitcell_length, pair_idx1, pair_idx2)

    centroid = np.average(grouped_pos[alignment_idxs], axis=0)
    grouped_centered_pos = grouped_pos - centroid

    if important_idxs is not None:
        grouped_centered_pos_imp = grouped_centered_pos[important_idxs]
        align_imp_idxs = np.arange(grouped_centered_pos_imp.shape[0])

        if grouped_centered_pos_imp.shape[0] != ref_coords.shape[0]:
            raise ValueError(
                "Number of important atoms does not match between reference and provided coords"
            )

        superimposed_imp, rotation_matrix, _ = superimpose(
            ref_coords,
            grouped_centered_pos_imp,
            idxs=align_imp_idxs,
        )

        if return_full_aligned:
            full_aligned = np.dot(grouped_centered_pos, rotation_matrix)
            return full_aligned
        else:
            return superimposed_imp

    else:
        if coords.shape[0] != ref_coords.shape[0]:
            raise ValueError(
                "Number of atoms does not match between reference and provided coords"
            )

        superimposed_pos, rotation_matrix, _ = superimpose(
            ref_coords,
            grouped_centered_pos,
            idxs=alignment_idxs,
        )

        if return_full_aligned:
            return np.dot(grouped_centered_pos, rotation_matrix)
        else:
            return superimposed_pos


class DistanceTICAProjector(Projector):
    """
    Projects a state into a predefined TICA space, using a set of distances as intermediate features.
    """

    def __init__(self, dist_idxs, tica_model, periodic=True):
        """Construct a DistanceTICA projector.

        Parameters
        ----------

        dist_idxs : np.array of shape (nd,2) 
            Indices of atoms for computing distances in an image.
        
        tica_model : Deeptime or equivalent object 
            It MUST have a transform function, which 
            will be used to transform the distances into tica space.
        
        periodic : bool (default = True)
            Whether to use periodic boundary conditions to minimize pair distances
        
        """

        self.dist_idxs = np.asarray(dist_idxs, dtype=int)
        self.periodic = periodic
        self.model = tica_model
        self.ndim = self.model.dim

    def project(self, state):


        disp_vecs = state['positions'][self.dist_idxs[:, 0]] - state['positions'][self.dist_idxs[:, 1]]

        if self.periodic:
            box_lengths, _ = box_vectors_to_lengths_angles(state['box_vectors'])
            disp_vecs = shorten_vecs(disp_vecs, box_lengths)

        dists = np.linalg.norm(disp_vecs, axis=1)
        proj = self.model.transform(dists)
        
        print(f'Proj: {proj}')
		
        return proj


class CoordTICAProjector(Projector):
    """Projects a state into a predefined tICA space using the exact
    alignment and coordinate-feature construction path from
    calc_coord_features_singleref.py file (from the MD_Interpret library).
    """

    def __init__(
        self,
        alignment_idxs,
        atom_idxs,
        tica_model,
        ref_centered_pos,
        pair_idx1=None,
        pair_idx2=None,
        periodic=True,
    ):

        """
        Parameters
        ----------
        alignment_idxs: array-like of shape (align_atoms,)
            Indices of atoms whose coordinates are used as the reference to center 
	    the grouped frames. These atoms MUST match the atoms that
            were used to center the frames before aligning the coords for tica training.

        atom_idxs : array-like of shape (natoms,)
            Indices of the atoms whose coordinates to superimpose and 
	    then extracted as as features.
            The order MUST match the order used when training TICA.

        
        tica_model: Deeptime or equivalent object 
            It MUST have a transform function, which 
            will be used to transform the aligned coordinates into tica space.

        ref_centered_pos: arraylike
            MUST FOLLOW some conditions:
            1. Centered reference frame coordinates to be used in the superimpose function
            2. The exact ref pose used while making the coord features for training the tica model.
            3. Generally should be the output of the ref_centered_pose function
        
        periodic : bool (default = True)
            Whether to use periodic boundary conditions to minimize pair distances

        """



        self.alignment_idxs = np.asarray(alignment_idxs, dtype=int)
        self.atom_idxs = np.asarray(atom_idxs, dtype=int)
        self.ref_centered_pose = np.asarray(ref_centered_pos)
        self.pair_idx1 = np.asarray(pair_idx1 if pair_idx1 is not None else alignment_idxs, dtype=int)
        self.pair_idx2 = np.asarray(pair_idx2 if pair_idx2 is not None else atom_idxs, dtype=int)
        self.periodic = periodic
        self.model = tica_model
        self.ndim = self.model.dim

        if self.ref_centered_pose.shape[0] != self.atom_idxs.shape[0]:
            raise ValueError(
                "ref_centered_pos and atom_idxs must represent the same number of atoms "
                "for coordinate-tICA projection."
            )

    def project(self, state):
        if self.periodic:
            box_lengths, _ = box_vectors_to_lengths_angles(state['box_vectors'])
        else:
            box_lengths = np.array([1.0e9, 1.0e9, 1.0e9], dtype=float)

        feat_coords = aligned_frame_for_coord_tica(
            coords=state['positions'],
            ref_coords=self.ref_centered_pose,
            unitcell_length=box_lengths,
            alignment_idxs=self.alignment_idxs,
            pair_idx1=self.pair_idx1,
            pair_idx2=self.pair_idx2,
            important_idxs=self.atom_idxs,
            return_full_aligned=False,
        )

        
        feat_coord = feat_coords.reshape(1, -1)
        
        proj = self.model.transform(feat_coord)
        print(f'Proj: {proj}')
        return proj

