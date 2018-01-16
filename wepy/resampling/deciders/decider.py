import random as rand

from wepy.resampling.decisions.decision import NoDecision
from wepy.resampling.decisions.clone_merge import CloneMergeDecision

class Decider(object):

    def decide(self):
        return NotImplementedError

class NothingDecider(Decider):
    DECISION = NoDecision

    def __init__(self):
        pass

    def decide(self, novelties):

        decisions = []
        for novelty in novelties:
            # save the only value for the enumeration for the record
            decisions.append(NoDecision.ENUM.NOTHING.value)


        return decisions, {}

