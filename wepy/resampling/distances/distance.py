class Distance(object):

    def __init__(self):
        raise NotImplementedError

    def preimage(self, state):
        """Return the reduced representation of a state that is the only
        necessary portion needed for calculating the distance between two
        states.

        This is useful for storing "images" of the states that are
        much smaller than the potentially very large states.

        This is the abstract implementation which just returns the
        whole state, as a default for all subclasses. Overriding this
        will customize this functionality without having to also
        override the distance method.

        """

        return state

    def preimage_distance(preimage_a, preimage_b):
        """The preimage_distance is the distance function computed between the
        exact preimages necessary for the resultant distance value.

        The `distance` function is just a wrapper around this function
        which first gets the preimages from valid states.

        This needs to be implemented in subclasses of Distance.

        """
        raise NotImplementedError

    def distance(self, state_a, state_b):
        """ Compute the distance between two states. """

        return self.preimage_distance(self.preimage(state_a),
                                      self.preimage(state_b))
