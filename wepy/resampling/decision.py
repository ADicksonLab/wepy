from enum import Enum

class Decision(Enum):
    pass

class CloneMergeDecision(Decision):
    NOTHING = 1
    CLONE = 2
    SQUASH = 3
    KEEP_MERGE = 4
