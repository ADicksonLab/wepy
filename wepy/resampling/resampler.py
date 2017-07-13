from collections import namedtuple
import random as rand

from wepy.walker import merge
from wepy.resampling.decision import Decision

class Resampler(object):

    def resample(self, walkers, decisions):
        raise NotImplementedError

ResamplingRecord = namedtuple("ResamplingRecord", ['decision', 'value'])


# stubs and examples of resamplers for testing purposes
class NoResampler(Resampler):

    def resample(self, walkers):

        resampling_records = [ResamplingRecord(decision=Decision.FALSE, value=i)
                             for i in len(walkers)]

        return walkers, [resampling_records]
