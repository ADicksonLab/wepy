import logging

import numpy as np

from wepy.util.util import box_vectors_to_lengths_angles

from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.rmsd import calc_rmsd
from geomm.centering import center_around

from wepy.resampling.distances.distance import Distance

class RMSDDistance(Distance):
    """ """
    def __init__(self, atom_idxs):

        # the idxs to align / compute RMSD with
        self._atom_idxs = atom_idxs

    def image(self, state):
        """

        Parameters
        ----------
        state :
            

        Returns
        -------

        """

        # slice these positions to get the image
        state_image = state['positions'][self._atom_idxs]

        # center selection at 0,0,0
        idxs = list(range(len(self._atom_idxs)))
        return center_around(state_image,idxs)

    def image_distance(self, image_a, image_b):
        """

        Parameters
        ----------
        image_a :
            
        image_b :
            

        Returns
        -------

        """


        # then superimpose it to the reference structure
        sup_image_b, _, _ = superimpose(image_a, image_b)

        # calculate the rmsd of entire selection
        return calc_rmsd(image_a, sup_image_b)

