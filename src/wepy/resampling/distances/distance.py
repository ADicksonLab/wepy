import logging

class Distance(object):
    """ """

    def __init__(self):
        pass

    def image(self, state):
        """

        Parameters
        ----------
        state :
            

        Returns
        -------
        type
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

    def image_distance(self, image_a, image_b):
        """The image_distance is the distance function computed between the
        exact images necessary for the resultant distance value.
        
        The `distance` function is just a wrapper around this function
        which first gets the images from valid states.
        
        This needs to be implemented in subclasses of Distance.

        Parameters
        ----------
        image_a :
            
        image_b :
            

        Returns
        -------

        """
        raise NotImplementedError

    def distance(self, state_a, state_b):
        """Compute the distance between two states.

        Parameters
        ----------
        state_a :
            
        state_b :
            

        Returns
        -------

        """

        return self.image_distance(self.image(state_a),
                                      self.image(state_b))

