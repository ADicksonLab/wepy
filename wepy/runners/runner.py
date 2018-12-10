import logging

class Runner(object):
    """ """

    def run_segment(self, init_walkers, segment_length):
        """

        Parameters
        ----------
        init_walkers :
            
        segment_length :
            

        Returns
        -------

        """
        raise NotImplementedError

class NoRunner(Runner):
    """Stub class that just returns the walkers back with the same state."""

    def run_segment(self, init_walkers, segment_length):
        """

        Parameters
        ----------
        init_walkers :
            
        segment_length :
            

        Returns
        -------

        """
        return init_walkers
