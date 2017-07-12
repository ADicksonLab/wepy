class Resampler(object):

    def resample(self, walkers, decisions):
        raise NotImplementedError

class NoResampler(Resampler):

    def resample(self, walkers):
        return walkers, None

class WExploreResampler(Resampler):
    pass

