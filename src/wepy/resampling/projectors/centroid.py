"""Projector for determining centroid distances.
"""
# Standard Library
import logging

logger = logging.getLogger(__name__)

# Third Party Library
import numpy as np

from wepy.resampling.projectors.projector import Projector
from wepy.util.util import box_vectors_to_lengths_angles
from geomm.grouping import group_pair

class CentroidProjector(Projector):
    """Projects a state onto the centroid distance between two groups.
    """

    def __init__(self, group1_idxs, group2_idxs, periodic=True):
        """Construct a centroid distance projector.

        Parameters
        ----------

        group1_idxs : list of int - indices of atoms in group1
        group2_idxs : list of int - indices of atoms in group2
        periodic :    bool (default = True) - whether to use periodic boundary conditions to
                      minimize centroid distances
        """
        self.group1_idxs = group1_idxs
        self.group2_idxs = group2_idxs
        self.periodic = periodic
      
    def project(self, state):

        # cut out only the coordinates we need
        coords = np.concatenate([state['positions'][self.group1_idxs],state['positions'][self.group2_idxs]])
        idxs1 = list(range(len(self.group1_idxs)))
        idxs2 = list(range(len(self.group1_idxs),len(self.group1_idxs) + len(self.group2_idxs)))
        
        if self.periodic:
            # get the box lengths from the vectors
            box_lengths, box_angles = box_vectors_to_lengths_angles(state["box_vectors"])
            coords = group_pair(coords,box_lengths,idxs1,idxs2)

        # determine coordinate centroids
        c1 = coords[idxs1].mean(axis=0)
        c2 = coords[idxs2].mean(axis=0)

        # return the distance between the centroids
        return np.sqrt(np.sum(np.square(c1-c2)))


