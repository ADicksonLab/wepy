"""Projectors into a pre-trained tICA space.
"""
# Standard Library
import logging

logger = logging.getLogger(__name__)

# Third Party Library
import numpy as np

from wepy.resampling.projectors.projector import Projector
from wepy.util.util import box_vectors_to_lengths_angles
#from geomm.grouping import shorten_vec

def shorten_vec(x, unitcell_side_lengths):
    """
    For a given vector x between two points in a periodic box, return the
    shortest version of that vector.
    """
    pos_idxs = np.where(x > 0.5*unitcell_side_lengths)[0]

    for dim_idx in pos_idxs:
        x[dim_idx] -= unitcell_side_lengths[dim_idx]

    neg_idxs = np.where(x < -0.5*unitcell_side_lengths)[0]

    for dim_idx in neg_idxs:
        x[dim_idx] += unitcell_side_lengths[dim_idx]

    return x


def build_coord_feature(state, ref_pos, idxs,):

    
    return aligned_feat_coord
    
    
class DistanceTICAProjector(Projector):
    """Projects a state into a predefined TICA space, using a set of distances as intermediate features.
    """

    def __init__(self, dist_idxs, tica_model, periodic=True):
        """Construct a DistanceTICA projector.

        Parameters
        ----------

        dist_idxs : np.array of shape (nd,2) - indices of atoms for computing distances
        
        tica_model : Deeptime or equivalent object that has a transform function, which 
                    will be used to transform the distances into tica space.
        
        periodic :    bool (default = True) - whether to use periodic boundary conditions to
                      minimize pair distances
        """

        self.dist_idxs = np.array(dist_idxs)
        self.model = tica_model
        self.periodic = periodic

        # check for transform type functions
        if hasattr(self.model, 'transform'):

        self.ndim = self.model.dim
    
    def project(self, state):

        # get all the displacement vectors
        disp_vecs = state['positions'][self.dist_idxs[:,0]] - state['positions'][self.dist_idxs[:,1]]
            
        if self.periodic:
            # get the box lengths from the vectors
            box_lengths, box_angles = box_vectors_to_lengths_angles(state["box_vectors"])
            dists = np.array([np.sqrt(np.sum(np.square(shorten_vec(v, box_lengths)))) for v in disp_vecs])
        else:
            dists = np.array([np.sqrt(np.sum(np.square(v))) for v in disp_vecs])
            
        tranformed_dists = self.model.transform(dists)

        # calculate projections
        proj = np.zeros(self.ndim, tranformed_dists.shape[0])
        for i in range(self.ndim):
            proj[i] = np.dot(tranformed_dists[i], dists)
        
        return proj

class CoordTICAProjector(Projector):
    """Projects a state into a predefined TICA space, using selected
    Cartesian coordinates (e.g., Cα atoms) as features.

    The feature vector is constructed by taking the positions of a
    fixed set of atoms and flattening them to shape (natoms*3,).
    """

    def __init__(self, atom_idxs, tica_vectors, alignment_idxs,
                 periodic=True):
        """
        Parameters
        ----------
        atom_idxs : array-like of shape (natoms,)
            Indices of the atoms whose coordinates are used as features.
            The order MUST match the order used when training TICA.

        model: tica model to extract the vectors
        
        tica_vectors : np.ndarray of shape (ntica, natoms*3)
            TICA eigenvectors for projecting into TICA space.

        alignment_idxs: array-like of shape (align_atoms,) 
            Indices of atoms whose coordinates are used as the reference for 
            alignment of the frames. These atoms MUST match the atoms that
            were used to align the frames before tica model training.

        periodic : bool, default=True
            Whether to take periodic boundary conditions into account.

        """

        self.atom_idxs = np.asarray(atom_idxs, dtype=int)
        self.tica_vectors = np.asarray(tica_vectors, dtype=float)
        self.periodic = periodic
        self.alignment_idxs = alignment_idxs

        natoms = self.atom_idxs.shape[0]
        nfeat = natoms * 3

        assert self.tica_vectors.ndim == 2, "tica_vectors must be 2D"
        assert self.tica_vectors.shape[1] == nfeat, \
            f"TICA vectors expect {self.tica_vectors.shape[1]} features, " \
            f"but natoms*3 = {nfeat}"

        self.ndim = self.tica_vectors.shape[0]

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
        ## positions of selected atoms: (natoms, 3)
        #pos = state['positions'][self.atom_idxs].copy()
        ## 
        #ref

        ## ALWAYS ALIGN WITH A SPECIFIC SELECTION
        ## USING GEOMM AND GROUP PAIR< CENTER AROUND
        ## 
        
        if self.periodic:
            # use minimum-image convention for each displacement
            box_lengths, _ = box_vectors_to_lengths_angles(state["box_vectors"])
            for i in range(pos.shape[0]):
                disp = pos[i] - ref
                disp = shorten_vec(disp, box_lengths)
                pos[i] = disp
        else:
            pos -= ref
        #else:
        #    # optional: still can correct for PBC wrt some origin, but
        #    # usually you want anchor_first_atom=True for coord-based TICA
        #    if self.periodic:
        #        logger.warning("periodic=True but anchor_first_atom=False; "
        #                       "coordinates may be discontinuous across PBCs.")



        # flatten to (natoms*3,)
        feat_vec = pos.reshape(-1)
        # project into TICA space: (ntica, nfeat) @ (nfeat,) -> (ntica,)
        proj = self.tica_vectors.dot(feat_vec)

        return proj


