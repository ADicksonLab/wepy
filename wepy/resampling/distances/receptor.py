import logging

import numpy as np

from wepy.util.util import box_vectors_to_lengths_angles

from geomm.recentering import recenter_pair
from geomm.superimpose import superimpose
from geomm.rmsd import calc_rmsd

from wepy.resampling.distances.distance import Distance

class UnbindingDistance(Distance):
    def __init__(self, ligand_idxs, binding_site_idxs, ref_state):

        # the idxs of the ligand and binding site from the whole state
        self._lig_idxs = ligand_idxs
        self._bs_idxs = binding_site_idxs

        # number of atoms in each
        self._n_lig_atoms = len(self._lig_idxs)
        self._n_bs_atoms = len(self._bs_idxs)

        # the idxs used for the whole image
        self._image_idxs = np.concatenate( (self._lig_idxs, self._bs_idxs) )

        # the idxs of the ligand and binding site within the image
        self._image_lig_idxs = np.arange(self._n_lig_atoms)
        self._image_bs_idxs = np.arange(self._n_lig_atoms, self._n_lig_atoms + self._n_bs_atoms)

        # save the reference state's image so we can align all further
        # images to it

        self.ref_image = self._unaligned_image(ref_state)

    def _unaligned_image(self, state):

        # get the box lengths from the vectors
        box_lengths, box_angles = box_vectors_to_lengths_angles(state['box_vectors'])

        # recenter the protein-ligand complex into the center of the
        # periodic boundary conditions
        rece_positions = recenter_pair(state['positions'], box_lengths,
                                   self._bs_idxs, self._lig_idxs)


        # slice these positions to get the image
        state_image = rece_positions[self._image_idxs]

        return state_image

    def image(self, state):

        # get the unaligned image
        state_image = self._unaligned_image(state)

        # then superimpose it to the reference structure
        sup_image = superimpose(self.ref_image, state_image, idxs=self._image_bs_idxs)

        return sup_image

    def image_distance(self, image_a, image_b):

        # then we calculate the rmsd of only the ligands between the
        # images
        lig_rmsd = calc_rmsd(image_a, image_b, idxs=self._image_lig_idxs)

        return lig_rmsd
