"""Distance metrics related to common features of receptor-ligand
processes in molecular systems.

The ReceptorDistance class is an abstract class that provides some
common functionality for normalizing reference states, providing
correct indices of receptor and ligand atoms, and a common image
function.

Subclasses of ReceptorDistance need only implement the
'image_distance' function according to their needs.

The UnbindingDistance is a useful metric for enhancing ligand movement
away from the reference bound state conformation.

The RebindingDistance is a useful metric for enhancing the movement of
a ligand towards a reference state.

"""

import logging

import numpy as np

from wepy.util.util import box_vectors_to_lengths_angles

from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.rmsd import calc_rmsd
from geomm.centering import center_around

from wepy.resampling.distances.distance import Distance

class ReceptorDistance(Distance):
    """Common abstract class for receptor-ligand molecular systems.

    Any input walker state should have the attributes 'positions' and
    'box_vectors' and will first preprocess these states, by
    recentering the complex. Two component systems like
    receptor-ligands in periodic boxes can sometimes drift to
    different periodic images of the box making them unsuitable for
    RMSD calculations.

    The 'image' method will align the receptor structures of the state
    to the reference state after the recentering.

    """

    def __init__(self, ligand_idxs, binding_site_idxs, ref_state):
        """Construct a distance metric.

        Parameters
        ----------

        ligand_idxs : arraylike of int
            The indices of the atoms from the 'positions' attribute of
            states that correspond to the ligand molecule.

        binding_site_idxs : arraylike of int
            The indices of the atoms from the 'positions' attribute of
            states that correspond to the atoms of the binding site of
            the receptor. These are the atoms you want to perform
            alignment on and so can be a subset of a complete molecule.

        ref_state : object implementing WalkerState
            The reference state all walker states will be aligned to
            with 'positions' (Nx3 dims) and 'box_vectors' (3x3 array)
            attributes.

        """


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
        """The preprocessing method of states.

        First it groups the binding site and ligand into the same
        periodic box image and then centers the box around their
        mutual center of mass and returns only the positions of the
        binding site and ligand.

        Parameters
        ----------
        state : object implementing WalkerState
            State with 'positions' (Nx3 dims) and 'box_vectors' (3x3
            array) attributes.

        Returns
        -------

        """

        # get the box lengths from the vectors
        box_lengths, box_angles = box_vectors_to_lengths_angles(state['box_vectors'])

        # recenter the protein-ligand complex into the center of the
        # periodic boundary conditions

        # regroup the ligand and protein in together
        grouped_positions = group_pair(state['positions'], box_lengths,
                                    self._bs_idxs, self._lig_idxs)

        # then center them around the binding site
        centered_positions = center_around(grouped_positions, self._bs_idxs)

        # slice these positions to get the image
        state_image = centered_positions[self._image_idxs]

        return state_image

    def image(self, state):
        """Transform a state to a receptor image.

        A receptor image is one in which the binding site and ligand
        are first normalized to be within the same periodic box image
        and at the center of the box, and then only the binding site
        and ligand are retained.

        Parameters
        ----------
        state : object implementing WalkerState
            State with 'positions' (Nx3 dims) and 'box_vectors' (3x3 array)
            attributes.

        Returns
        -------

        receptor_image : array of float
            The positions of binding site and ligand after
            preprocessing.

        """

        # get the unaligned image
        state_image = self._unaligned_image(state)

        # then superimpose it to the reference structure
        sup_image, _, _ = superimpose(self.ref_image, state_image, idxs=self._image_bs_idxs)

        return sup_image


class UnbindingDistance(ReceptorDistance):
    """Distance metric for measuring differences between walker states in
    regards to the RMSDs between ligands.

    Images are produced using the ReceptorDistance.image method. The
    distance between images then is solely the RMSD of the ligand
    atoms.

    Thus walkers were ligands are divergent in position will have
    higher distances.

    """

    def image_distance(self, image_a, image_b):

        # we calculate the rmsd of only the ligands between the
        # images
        lig_rmsd = calc_rmsd(image_a, image_b, idxs=self._image_lig_idxs)

        return lig_rmsd

class RebindingDistance(ReceptorDistance):
    """Distance metric for measuring differences between walker states in
    regards to the RMSDs between ligands.

    Images are produced using the ReceptorDistance.image method. The
    distance between images then is the relative difference between
    the ligand RMSDs to the reference state.

    """

    def image_distance(self, image_a, image_b):

        # we calculate the rmsd of only the ligands between each image
        # and the reference
        state_a_rmsd = calc_rmsd(self.ref_image, image_a, idxs=self._image_lig_idxs)
        state_b_rmsd = calc_rmsd(self.ref_image, image_b, idxs=self._image_lig_idxs)

        # then we get the absolute value of the reciprocals of these rmsd
        # values
        d = abs(1./state_a_rmsd - 1./state_b_rmsd)

        return d
