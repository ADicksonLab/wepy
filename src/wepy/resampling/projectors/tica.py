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
        x[dim_idx] += unitcell_side_lengths[dim_idx]

    neg_idxs = np.where(x < -0.5*unitcell_side_lengths)[0]

    for dim_idx in neg_idxs:
        x[dim_idx] -= unitcell_side_lengths[dim_idx]

    return x

class DistanceTICAProjector(Projector):
    """Projects a state into a predefined TICA space, using a set of distances as intermediate features.
    """

    def __init__(self, dist_idxs, tica_vectors, periodic=True):
        """Construct a DistanceTICA projector.

        Parameters
        ----------

        dist_idxs : np.array of shape (nd,2) - indices of atoms for computing distances
        tica_vectors : np.array of shape (nt,nd) - vectors for projecting into tica space
        periodic :    bool (default = True) - whether to use periodic boundary conditions to
                      minimize pair distances
        """
        self.dist_idxs = np.array(dist_idxs)
        self.tica_vectors = np.array(tica_vectors)
        self.periodic = periodic

        assert self.dist_idxs.shape[0] == self.tica_vectors.shape[1], "Number of distances needs to match dimensionality of tica vectors!"
        assert self.dist_idxs.shape[1] == 2, "Distance indices have incorrect shape"

        self.ndim = self.tica_vectors.shape[0]
      
    def project(self, state):

        # get all the displacement vectors
        disp_vecs = state['positions'][self.dist_idxs[:,0]] - state['positions'][self.dist_idxs[:,1]]
            
        if self.periodic:
            # get the box lengths from the vectors
            box_lengths, box_angles = box_vectors_to_lengths_angles(state["box_vectors"])
            dists = np.array([np.sqrt(np.sum(np.square(shorten_vec(v)))) for v in disp_vecs])
        else:
            dists = np.array([np.sqrt(np.sum(np.square(v))) for v in disp_vecs])
            
        # calculate projections
        proj = np.zeros(self.ndim)
        for i in range(self.ndim):
            proj[i] = np.dot(self.tica_vectors[i], dists)
        
        return proj

