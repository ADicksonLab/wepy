from collections import namedtuple
import random as rand

from wepy.walker import merge
from wepy.resampling.decision import ResamplingRecord, NoDecision

class Resampler(object):

    def __init__(self, novelty, decider):
        self.novelty = novelty
        self.decider = decider

    def resample(self, walkers):

        resampled_walkers = walkers
        resampling_actions = []
        aux_data = {}
        finished = False
        step_count = 0
        while not finished:
            novelties, novelty_aux = self.novelty.novelties(walkers)
            finished, decisions, decider_aux = self.decider.decide(novelties)
            resampled_walkers = self.decider.decision.action(walkers, decisions)
            resampling_actions.append(decisions)
            aux_data.update([novelty_aux, decider_aux])

            step_count += 1

        return resampled_walkers, resampling_actions, aux_data
