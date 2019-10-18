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

    RUNNER_IDX = 0
    BC_IDX = 1
    RESAMPLER_IDX = 2

    def __init__(self, runner, resampler=None, boundary_conditions=None):

        if resampler is None:
            raise ValueError("must provide a resampler")

        # add them in the order they are done in Wepy
        filters = [runner, boundary_conditions, resampler]

        super().__init__(filters)

    @property
    def runner(self):
        return self.filters[self.RUNNER_IDX]

    @property
    def boundary_conditions(self):
        return self.filters[self.BC_IDX]

    @property
    def resampler(self):
        return self.filters[self.RESAMPLER_IDX]


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

