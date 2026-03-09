"""Projectors into a pre-trained tICA space.
"""
# Standard Library
import logging

logger = logging.getLogger(__name__)

# Third Party Library
import numpy as np

# geomm functions
from geomm.grouping import group_pair
from geomm.superimpose import superimpose

# wepy functions
from wepy.resampling.projectors.projector import Projector
from wepy.util.util import box_vectors_to_lengths_angles


def aligned(coords, ref_coords,unitcell_length, alignment_idxs, pair_idx1,pair_idx2):

    """For a frame from a trajectory this function does a bunch of operations
    systematically:
    (i) First moves pair_idx2 coordinates to the image of the periodic unitcell that minimizes the 
    difference between the centers of geometry between the pair_idx2 and pair_idx1 (e.g. a protein and ligand).
    (ii) Then, centers the set of coordinates around a set of alingment atom indices.
    (iii) Finally superimpose all the frames on top of a reference frame. (alignment)


    This uses the geomm group pair function, followed by centering of grouped snapshots and then again
    geomm superimpose function.
    Unlike the 'aligned_frames' function, it only operates over a single frame (not the entire trajectory)

    Parameters
    ----------

    coords : arraylike
        The coordinates array of the frame of the particles you will be transforming.
        (group pairing followed by centering followed by superimpose/alignment)

    ref_coords: arraylike
        Output of the ref_centered_pose function
        Centered reference frame coordinates to be used in superimpose

    unitcell_length : arraylike 
        The lengths of the sides of a rectangular unitcell.

    alignment_idxs : arraylike 
        Collection of the indices which will be used to center the snapshot
        and align them against the ref pose.
        Please ensure that the same indices are used while centering the reference pose too.

    pair_idx1 : arraylike 
        Collection of the indices that define that member of the pair.

    pair_idxs2 : arraylike
        Collection of the indices that define that member of the pair.

    Returns
    -------

    superimposed_pos : arrays
        Transformed coordinates of the frame.
        Transformation: group pairing followed by centering followed by superimpose/alignment.

    """



    assert coords.shape[1] == 3, "coordinates are not of 3 dimensions"
    assert coords.shape[0] == ref_coords.shape[0], "Number of atoms does not match between reference and provided coords"

    grouped_pos = group_pair(coords, unitcell_length, pair_idx1, pair_idx2)
    centroid = np.average(grouped_pos[alignment_idxs], axis =0)
    grouped_centered_pos = grouped_pos - centroid

    superimposed_pos, _ , _ = superimpose(ref_coords,grouped_centered_pos, idxs=alignment_idxs)

    return superimposed_pos

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

        self.dist_idxs = np.array(dist_idxs)
        self.periodic = periodic

        self.model = tica_model
        #check if the model has .transform object/attribute 
        # hasattr(self.model, 'transform'):

        self.ndim = self.model.dim
    
    def project(self, state):

        # get all the displacement vectors
        disp_vecs = state['positions'][self.dist_idxs[:,0]] - state['positions'][self.dist_idxs[:,1]]

        if self.periodic:
            box_lengths, _ = box_vectors_to_lengths_angles(state["box_vectors"])
            disp_vecs = shorten_vecs(disp_vecs, box_lengths)

        dists = np.linalg.norm(disp_vecs, axis=1)
        
        #print(f'Dist: {dists}')
        proj = self.model.transform(dists)

        print(f'Proj: {proj}')
        return proj


class CoordTICAProjector(Projector):
    """Projects a state into a predefined TICA space, using selected
    Cartesian coordinates (e.g., CA atoms or backbone atoms) as features.

    The feature vector is constructed by taking the positions of a
    fixed set of atoms and flattening them to shape (natoms*3,).
    """

    def __init__(self, alignment_idxs, atom_idxs, tica_model, ref_centered_pos, periodic=True):
        """
        Parameters
        ----------
        alignment_idxs: array-like of shape (align_atoms,)
            Indices of atoms whose coordinates are used as the reference for 
            alignment of the frames. These atoms MUST match the atoms that
            were used to align the frames before tica model training.

        atom_idxs : array-like of shape (natoms,)
            Indices of the atoms whose coordinates are used as features.
            The order MUST match the order used when training TICA.

        
        model: Deeptime or equivalent object 
            It MUST have a transform function, which 
            will be used to transform the distances into tica space.

        ref_centered_pos: arraylike
            MUST FOLLOW some conditions:
            1. Centered reference frame coordinates to be used in the superimpose function
            2. The exact ref pose used while making the coord features for training the tica model.
            3. Generally should be the output of the ref_centered_pose function
        
        periodic : bool (default = True)
            Whether to use periodic boundary conditions to minimize pair distances

        """

        self.atom_idxs = np.asarray(atom_idxs, dtype=int)
        self.alignment_idxs = np.array(alignment_idxs, dtype=int)
        self.ref_centered_pose = ref_centered_pos

        self.periodic = periodic
        self.model = tica_model
        #check if the model has .transform object/attribute 
        # hasattr(self.model, 'transform'):
        self.ndim = self.model.dim

        #natoms = self.atom_idxs.shape[0]
        #nfeat = natoms * 3


    def project(self, state):
        """
        Project the given walker state into TICA space.

        Parameters
        ----------
        state : dict-like
            Must contain 'positions' with shape (N, 3) and, if
            periodic=True, 'box_vectors'.

        Returns
        -------
        proj : np.ndarray of shape (ntica,)
            TICA coordinates for this state.
        """
        
        
        if self.periodic:
            box_lengths, _ = box_vectors_to_lengths_angles(state["box_vectors"])

            pos_aligned = aligned(coords = state['positions'],
                                            ref_coords = self.ref_centered_pose,
                                            unitcell_length = box_lengths,
                                            alignment_idxs = self.alignment_idxs,
                                            pair_idx1 = self.alignment_idxs,
                                            pair_idx2 = self.atom_idxs)


        else: # Discuss with Alex
            centroid = np.average(state['positions'][self.alignment_idxs], axis =0)
            centered_pos = state['positions'] - centroid
            pos_aligned, _ , _ = superimpose(self.ref_centered_pose, centered_pos, idxs=self.alignment_idxs)


        feat_coord = pos_aligned[self.atom_idxs].reshape(1, -1) 
        # without reshape it is still not in the input shape for the parent tica model 
        # the parent tica model is trained on a flattened array.

        proj = self.model.transform(feat_coord)

        return proj
