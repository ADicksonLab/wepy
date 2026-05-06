"""Modular component for defining "projectors" that project a 
walker state into a one- or low-dimensional subspace. These are
usable within different resamplers.

This module contains an abstract base class for Projector classes.

This is similar to the 'image' function in a Distance object
"""
# Standard Library
import logging

logger = logging.getLogger(__name__)

# Third Party Library
import numpy as np

class Projector(object):
    """Abstract Base class for Projector classes."""

    def __init__(self):
        """Constructor for Projector class."""
        pass

    def project(self, state):
        """Compute the 'projection' of a walker state onto one
        or more variables.

        The abstract implementation is naive and just returns the 
        numpy array [1].

        Parameters
        ----------
        state : object implementing WalkerState
            The state which will be transformed to an image

        Returns
        -------
        projection : numpy array
            The same state that was given as an argument.

        """

        return np.ones((1))