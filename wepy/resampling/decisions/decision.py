from collections import namedtuple
from enum import Enum

import numpy as np

# the record type for all resampling records
ResamplingRecord = namedtuple("ResamplingRecord", ['decision', 'instruction'])

# ABC for the Decision class
class Decision(object):
    ENUM = None
    INSTRUCTION_RECORDS = None
    INSTRUCTION_DTYPES = None


    @classmethod
    def enum_dict_by_name(cls):
        d = {}
        for enum in cls.ENUM:
            d[enum.name] = enum
        return d

    @classmethod
    def enum_dict_by_value(cls):
        d = {}
        for enum in cls.ENUM:
            d[enum.value] = enum
        return d

    @classmethod
    def enum_by_value(cls, enum_value):
        d = cls.enum_dict_by_value()
        return d[enum_value]

    @classmethod
    def enum_by_name(cls, enum_name):
        d = cls.enum_dict_by_name()
        return d[enum_name]

    @classmethod
    def instruct_record(cls, enum_value, data):
        enum = cls.enum_by_value(enum_value)
        instruct_record = dict(cls.INSTRUCTION_RECORDS)[enum](*data)
        return instruct_record

    @classmethod
    def record(cls, enum_value, data):
        instruct_record = cls.instruct_record(enum_value, data)

        resampling_record = ResamplingRecord(decision=enum_value,
                                             instruction=instruct_record)

        return resampling_record

    @classmethod
    def action(cls, walkers, decisions):
        """Perform the instructions for a set of resampling records on
        walkers."""
        raise NotImplementedError

# an example of a Decision class that has the enumeration, instruction
# record namedtuple, and the instruction dtypes
class NoDecision(Decision):

    class NothingDecisionEnum(Enum):
        NOTHING = 0

    ENUM = NothingDecisionEnum

    INSTRUCTION_RECORDS = (
        (ENUM.NOTHING, namedtuple("NothingInstructionRecord", ['slot'])),
    )

    INSTRUCTION_DTYPES = (
        (ENUM.NOTHING, [('pos', np.int)]),
    )

    @classmethod
    def action(cls, walkers, decisions):
        # list for the modified walkers
        mod_walkers = [None for i in range(len(walkers))]
        # go through each decision and perform the decision
        # instructions
        for walker_idx, decision in enumerate(decisions):
            decision_value, instruction = decision
            if decision_value == cls.ENUM.NOTHING.value:

                # check to make sure a walker doesn't already exist
                # where you are going to put it
                if mod_walkers[instruction[0]] is not None:
                    raise ValueError(
                        "Multiple walkers assigned to position {}".format(instruction[0]))

                # put the walker in the position specified by the
                # instruction
                mod_walkers[instruction[0]] = walkers[walker_idx]

        return mod_walkers
