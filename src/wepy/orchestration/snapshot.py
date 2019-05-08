from copy import copy, deepcopy

class SimApparatus():
    """The simulation apparatus are the components needed for running a
    simulation without the initial conditions for starting the simulation.
    
    A runner is strictly necessary but a resampler and boundary
    conditions are not.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, filters):
        self._filters = deepcopy(filters)

    @property
    def filters(self):
        """ """
        return self._filters



class WepySimApparatus(SimApparatus):
    """ """

    def __init__(self, runner, resampler=None, boundary_conditions=None):

        # add them in the order they are done in Wepy
        filters = [runner]
        if boundary_conditions is not None:
            filters.append(boundary_conditions)
        if resampler is not None:
            filters.append(resampler)

        super().__init__(filters)

class SimSnapshot():
    """ """

    def __init__(self, walkers, apparatus):

        self._walkers = deepcopy(walkers)
        self._apparatus = deepcopy(apparatus)

    @property
    def walkers(self):
        """ """
        return self._walkers

    @property
    def apparatus(self):
        """ """
        return self._apparatus

