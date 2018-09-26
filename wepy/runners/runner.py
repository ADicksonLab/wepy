import logging

class Runner(object):

    def run_segment(self, init_walkers, segment_length):
        raise NotImplementedError

class NoRunner(Runner):
    """ Stub class that just returns the walkers back with the same state."""

    def run_segment(self, init_walkers, segment_length):
        return init_walkers
