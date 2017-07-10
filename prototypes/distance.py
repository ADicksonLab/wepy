class DistanceModel(object):

    def distance(self, a, b):
        """ Computes the distance between any two states."""
        return NotImplementedError

class ZeroDistance(object):
    """Stub example of a DistanceModel subclass that always returns 0.0 for any input."""

    def distance(self, a, b):
        return 0.0

class RMSDDistance(object):

    def distance(self, a, b):
        raise NotImplementedError
