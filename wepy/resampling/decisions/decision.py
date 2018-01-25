from collections import namedtuple
from enum import Enum
from string import ascii_lowercase

import numpy as np

# the record type for all resampling records
ResamplingRecord = namedtuple("ResamplingRecord", ['decision', 'instruction'])

def _alphabet_iterator():
    num_chars = 1
    alphabet_it = iter(ascii_lowercase)
    while True:
        try:
            letter = next(alphabet_it)
        except StopIteration:
            alphabet_it = iter(ascii_lowercase)
            letter = next(alphabet_it)
            num_chars += 1

        yield letter * num_chars

class VariableLengthRecord(object):
    def __init__(self, name, *values):
        self._fields = ()
        self._data = values
        self._name = name

        key_it = _alphabet_iterator()
        # add them to the object namespace alphanumerically
        for value in values:
            key = next(key_it)
            self.__dict__[key] = value
            self._fields += (key,)

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return "{}({})".format(self._name,
                    ', '.join(["{}={}".format(key, self.__dict__[key]) for key in self._fields]))

    def __repr__(self):
        return self.__str__()


# ABC for the Decision class
class Decision(object):
    ENUM = None
    INSTRUCTION_NAMES = None
    INSTRUCTION_FIELDS = None
    INSTRUCTION_FIELD_DTYPES = None


    @classmethod
    def enum_dict_by_name(cls):
        if cls.ENUM is None:
            raise NotImplementedError

        d = {}
        for enum in cls.ENUM:
            d[enum.name] = enum
        return d

    @classmethod
    def enum_dict_by_value(cls):
        if cls.ENUM is None:
            raise NotImplementedError

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
    def instruction_records(cls):
        if cls.INSTRUCTION_NAMES is None or cls.INSTRUCTION_FIELDS is None:
            raise NotImplementedError

        names = dict(cls.INSTRUCTION_NAMES)
        fields = dict(cls.INSTRUCTION_FIELDS)
        return {enum : (name, fields[enum]) for enum, name in names.items()}

    @classmethod
    def instruction_dtypes(cls):
        if cls.INSTRUCTION_FIELD_DTYPES is None or cls.INSTRUCTION_FIELDS is None:
            raise NotImplementedError

        fields = dict(cls.INSTRUCTION_FIELDS)
        dtypes_l = dict(cls.INSTRUCTION_FIELD_DTYPES)
        return {enum : list(zip(fields, dtypes_l[enum])) for enum, fields in fields.items()}

    @classmethod
    def instruct_record(cls, enum_value, data):
        enum = cls.enum_by_value(enum_value)
        instruct_records = cls.instruction_records()
        rec_name, fields = instruct_records[enum]

        # if fields is Ellipsis this indicates it is variable length
        if Ellipsis in fields:
            instruct_record = VariableLengthRecord(rec_name, *data)
        else:
            InstructRecord = namedtuple(rec_name, fields)
            instruct_record = InstructRecord(*data)

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

    INSTRUCTION_NAMES = (
        (ENUM.NOTHING, "NothingInstructionRecord"),
    )

    INSTRUCTION_FIELDS = (
        (ENUM.NOTHING, ('pos',)),)

    INSTRUCTION_FIELD_DTYPES = (
        (ENUM.NOTHING, (np.int,)),
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
