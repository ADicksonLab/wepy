from collections import namedtuple
from enum import Enum
from string import ascii_lowercase

import numpy as np

# the record type for all resampling records
ResamplingRecord = namedtuple("ResamplingRecord", ['decision', 'instruction'])

# for naming the fields of the VariableLengthRecord
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

# class that mimics a namedtuple (for at least how it is used in wepy)
# that doesn't specify the number of fields
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

    @classmethod
    def parent_panel(cls, resampling_panel):

        parent_panel = []
        for cycle_idx, cycle in enumerate(resampling_panel):

            # each stage in the resampling for that cycle
            # make a stage parent table
            parent_table = []

            # now iterate through the rest of the stages
            for step_idx, step in enumerate(cycle):

                # initialize a list for the parents of this stages walkers
                step_parents = [None for i in range(len(step))]

                # the rest of the stages parents are based on the previous stage
                for parent_idx, parent_rec in enumerate(step):

                    # if the decision is an ancestor then the instruction
                    # values will be the children
                    if parent_rec[0] in cls.ANCESTOR_DECISION_IDS:
                        child_idxs = parent_rec[1]
                        for child_idx in child_idxs:
                            step_parents[child_idx] = parent_idx

                # for the full stage table save all the intermediate parents
                parent_table.append(step_parents)

            # for the full parent panel
            parent_panel.append(parent_table)

        return parent_panel

    @classmethod
    def net_parent_table(cls, parent_panel):

        #net_parent_table = [[i for i in range(len(parent_panel[0][0]))]]
        net_parent_table = []

        # each cycle
        for cycle_idx, step_parent_table in enumerate(parent_panel):
            # for the net table we only want the end results,
            # we start at the last cycle and look at its parent
            step_net_parents = []
            n_steps = len(step_parent_table)
            for walker_idx, parent_idx in enumerate(step_parent_table[-1]):
                # initialize the root_parent_idx which will be updated
                root_parent_idx = parent_idx

                # if no resampling skip the loop and just return the idx
                if n_steps > 0:
                    # go back through the steps getting the parent at each step
                    for prev_step_idx in range(n_steps):
                        prev_step_parents = step_parent_table[-(prev_step_idx+1)]
                        root_parent_idx = prev_step_parents[root_parent_idx]

                # when this is done we should have the index of the root parent,
                # save this as the net parent index
                step_net_parents.append(root_parent_idx)

            # for this step save the net parents
            net_parent_table.append(step_net_parents)

        return net_parent_table


class NothingDecisionEnum(Enum):
    NOTHING = 0

# an example of a Decision class that has the enumeration, instruction
# record namedtuple, and the instruction dtypes
class NoDecision(Decision):

    ENUM = NothingDecisionEnum

    INSTRUCTION_NAMES = (
        (ENUM.NOTHING, "NothingInstructionRecord"),
    )

    INSTRUCTION_FIELDS = (
        (ENUM.NOTHING, ('pos',)),)

    INSTRUCTION_FIELD_DTYPES = (
        (ENUM.NOTHING, (np.int,)),
    )

    # the decision types that pass on their state
    ANCESTOR_DECISION_IDS = (ENUM.NOTHING.value,)

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
