"""Projectors into a pre-trained tICA space.
"""

import logging
import numpy as np

from geomm.grouping import group_pair
from geomm.superimpose import superimpose

from wepy.resampling.projectors.projector import Projector
from wepy.util.util import box_vectors_to_lengths_angles

logger = logging.getLogger(__name__)


def _validate_index_array(name, indices, expected_columns=None):
    """Return a validated, non-empty array of non-negative atom indices."""
    index_array = np.asarray(indices)

    if not np.issubdtype(index_array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer atom indices")

    if expected_columns is None:
        if index_array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
    elif index_array.ndim != 2 or index_array.shape[1] != expected_columns:
        raise ValueError(
            f"{name} must have shape (n, {expected_columns}), "
            f"got {index_array.shape}"
        )

    if index_array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(index_array < 0):
        raise ValueError(f"{name} must contain only non-negative atom indices")

    return index_array.astype(int, copy=False)


def _validate_tica_model(tica_model):
    """Validate the model interface used by the projector classes."""
    if tica_model is None:
        raise TypeError("tica_model must not be None")
    if not callable(getattr(tica_model, "transform", None)):
        raise TypeError("tica_model must provide a callable transform method")
    if not hasattr(tica_model, "dim"):
        raise TypeError("tica_model must provide a dim attribute")

    ndim = tica_model.dim
    if isinstance(ndim, (bool, np.bool_)) or not isinstance(
        ndim, (int, np.integer)
    ):
        raise TypeError("tica_model.dim must be a positive integer")
    if ndim <= 0:
        raise ValueError("tica_model.dim must be a positive integer")

    return int(ndim)


def _validate_boolean(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


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

    def __init__(self, dist_idxs, tica_model, periodic=True, tica_weights=None):
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

        self.dist_idxs = _validate_index_array(
            "dist_idxs", dist_idxs, expected_columns=2
        )
        self.periodic = _validate_boolean("periodic", periodic)
        self.model = tica_model
        self.ndim = _validate_tica_model(self.model)

        if tica_weights is None:
            self.tica_weights = np.ones(self.ndim, dtype=float)
        else:
            self.tica_weights = np.asarray(tica_weights, dtype=float)
            if self.tica_weights.shape != (self.ndim,):
                raise ValueError(
                    f"tica_weights must have shape ({self.ndim},), "
                    f"got {self.tica_weights.shape}"
                )

    def project(self, state):

        disp_vecs = state['positions'][self.dist_idxs[:, 0]] - state['positions'][self.dist_idxs[:, 1]]

        if self.periodic:
            box_lengths, _ = box_vectors_to_lengths_angles(state['box_vectors'])
            disp_vecs = shorten_vecs(disp_vecs, box_lengths)

        dists = np.linalg.norm(disp_vecs, axis=1)
        print(
            "DistanceTICAProjector transform input: "
            f"ndim={dists.ndim}, shape={dists.shape}"
        )
        proj = self.model.transform(dists)
        projection_array = np.asarray(proj)
        print(
            "DistanceTICAProjector transform output: "
            f"ndim={projection_array.ndim}, shape={projection_array.shape}"
        )

        if projection_array.ndim == 0 or projection_array.shape[-1] != self.ndim:
            raise ValueError(
                "tica_model.transform returned an output whose final dimension "
                f"does not match tica_model.dim={self.ndim}: "
                f"shape={projection_array.shape}"
            )

        weighted_proj = self.tica_weights * projection_array

        return weighted_proj


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
        tica_weights=None):

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



        self.alignment_idxs = _validate_index_array("alignment_idxs", alignment_idxs)
        self.atom_idxs = _validate_index_array("atom_idxs", atom_idxs)
        missing_alignment_idxs = self.alignment_idxs[
            ~np.isin(self.alignment_idxs, self.atom_idxs)
        ]
        if missing_alignment_idxs.size:
            raise ValueError(
                "alignment_idxs must be contained in atom_idxs; missing atom "
                f"indices: {missing_alignment_idxs.tolist()}"
            )
        self.ref_centered_pose = np.asarray(ref_centered_pos, dtype=float)
        self.pair_idx1 = _validate_index_array(
            "pair_idx1", pair_idx1 if pair_idx1 is not None else alignment_idxs
        )
        self.pair_idx2 = _validate_index_array(
            "pair_idx2", pair_idx2 if pair_idx2 is not None else atom_idxs
        )
        self.periodic = _validate_boolean("periodic", periodic)
        self.model = tica_model
        self.ndim = _validate_tica_model(self.model)

        expected_ref_shape = (self.atom_idxs.size, 3)
        if self.ref_centered_pose.shape != expected_ref_shape:
            raise ValueError(
                f"ref_centered_pos must have shape {expected_ref_shape}, "
                f"got {self.ref_centered_pose.shape}"
            )

        if tica_weights is None:
            self.tica_weights = np.ones(self.ndim, dtype=float)
        else:
            self.tica_weights = np.asarray(tica_weights, dtype=float)
            if self.tica_weights.shape != (self.ndim,):
                raise ValueError(
                    f"tica_weights must have shape ({self.ndim},), "
                    f"got {self.tica_weights.shape}"
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

        print(
            "CoordTICAProjector transform input: "
            f"ndim={feat_coord.ndim}, shape={feat_coord.shape}"
        )
        proj = self.model.transform(feat_coord)
        projection_array = np.asarray(proj)
        print(
            "CoordTICAProjector transform output: "
            f"ndim={projection_array.ndim}, shape={projection_array.shape}"
        )

        if projection_array.ndim == 0 or projection_array.shape[-1] != self.ndim:
            raise ValueError(
                "tica_model.transform returned an output whose final dimension "
                f"does not match tica_model.dim={self.ndim}: "
                f"shape={projection_array.shape}"
            )

        weighted_proj = self.tica_weights * projection_array

        return weighted_proj
