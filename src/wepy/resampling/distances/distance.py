"""Modular component for defining distance metrics usable within many
different resamplers.

This module contains an abstract base class for Distance classes.

The suggested implementation method is to leave the 'distance' method
as is, and override the 'image' and 'image_distance' methods
instead. Because the default 'distance' method calls these
transparently. The resamplers will use the 'image' and
'image_distance' calls because this allows performance optimizations.

For example in WExplore the images for some walkers end up being
stored as the definitions of Voronoi regions, and if the whole walker
state was stored it would not only use much more space in memory but
require that common transformations be repeated every time a distance
is to be calculated to that image (which is many times). In REVO all
to all distances between walkers are computed which would also incur a
high cost.

So use the 'image' to do precomputations on raw walker states and use
the 'image_distance' to compute distances using only those images.

"""
# Standard Library
import logging

logger = logging.getLogger(__name__)

# Third Party Library
import numpy as np
from wepy.util.util import box_vectors_to_lengths_angles


class Distance(object):
    """Abstract Base class for Distance classes."""

    def __init__(self):
        """Constructor for Distance class."""
        pass

    def image(self, state):
        """Compute the 'image' of a walker state which should be some
        transformation of the walker state that is more
        convenient. E.g. for precomputation of expensive operations or
        for saving as resampler state.

        The abstract implementation is naive and just returns the
        state itself, thus it is the identity function.

        Parameters
        ----------
        state : object implementing WalkerState
            The state which will be transformed to an image

        Returns
        -------
        image : object implementing WalkerState
            The same state that was given as an argument.

        """

        return state

    def image_distance(self, image_a, image_b):
        """Compute the distance between two images of walker states.

        Parameters
        ----------
        image_a : object produced by Distance.image

        image_b : object produced by Distance.image

        Returns
        -------

        distance : float
            The distance between the two images

        Raises
        ------

        NotImplementedError : always because this is abstract

        """
        raise NotImplementedError

    def distance(self, state_a, state_b):
        """Compute the distance between two states.

        Parameters
        ----------
        state_a : object implementing WalkerState

        state_b : object implementing WalkerState

        Returns
        -------

        distance : float
            The distance between the two walker states


        """

        return self.image_distance(self.image(state_a), self.image(state_b))


class XYEuclideanDistance(Distance):
    """2 dimensional euclidean distance between points.

    States have the attributes 'x' and 'y'.

    """

    def image(self, state):
        return np.array([state["x"], state["y"]])

    def image_distance(self, image_a, image_b):
        return np.sqrt((image_a[0] - image_b[0]) ** 2 + (image_a[1] - image_b[1]) ** 2)

class AtomPairDistance(Distance):
    """Constructs a vector of atomic distances for each state.
    Distance is the root mean squared distance between the vectors.
    """

    def __init__(self, pair_list, periodic=True):
        """Construct a distance metric.

        Parameters
        ----------

        pair_list : arraylike of tuples
            The indices of the atom pairs between which to compute
            distances.

        """
        self.pair_list = pair_list
        self.periodic = periodic

    def _adjust_disp_vector(self, disp, box_lengths):
        edited = True
        while edited:
            edited = False
            for i in range(3):
                if disp[i] > box_lengths[i]/2:
                    disp[i] -= box_lengths[i]
                    edited = True
                elif disp[i] < -box_lengths[i]/2:
                    disp[i] += box_lengths[i]
                    edited = True
        return disp
        
    def image(self, state):

        if self.periodic:
            # get the box lengths from the vectors
            box_lengths, box_angles = box_vectors_to_lengths_angles(state["box_vectors"])

        dist_list = np.zeros((len(self.pair_list)))
        for i,p in enumerate(self.pair_list):
            disp_vector = state["positions"][p[0]] - state["positions"][p[1]]
            if self.periodic:
                dist_list[i] = np.sqrt(np.sum(np.square(self._adjust_disp_vector(disp_vector,box_lengths))))
            else:
                dist_list[i] = np.sqrt(np.sum(np.square(disp_vector)))
        
        return dist_list

    def image_distance(self, image_a, image_b):
        return np.sqrt(np.mean(np.square(image_a - image_b)))
