import numpy as np

from geomm.superimpose import superimpose
from geomm.rmsd import calc_rmsd

from wepy.resampling.distances.distance import Distance

class UnbindingDistance(Distance):
    def __init__(self, ligand_idxs, binding_site_idxs):

        self._lig_idxs = ligand_idxs
        self._bs_idxs = binding_site_idxs

    def image(self, state):

        keep_idxs = np.concatenate( (self._lig_idxs, self._bs_idxs) )
        state_image = state[keep_idxs]

        return state_image

    def image_distance(self, image_a, image_b):

        # first we superimpose the two structures according to the
        # binding site, and by translating and rotating only one of
        # the images, image_b by convention.
        sup_image_b = superimpose(image_a, image_b, idxs=self._bs_idxs)


        # then we calculate the rmsd of only the ligands between the
        # images
        lig_rmsd = calc_rmsd(image_a, sup_image_b, idxs=self._lig_idxs)

        return lig_rmsd
