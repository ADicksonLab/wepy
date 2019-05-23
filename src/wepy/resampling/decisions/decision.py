"""

"""

from collections import namedtuple
from enum import Enum
from string import ascii_lowercase
import logging

import numpy as np

# ABC for the Decision class
class Decision(object):
    """Represents and provides methods for a set of decision values.

    """
    ENUM = None

    FIELDS = ('decision_id')
    # suggestion for subclassing
    # FIELDS = super().FIELDS + ('target_idxs',)
    # etc.

    #  An Ellipsis instead of fields indicate there is a variable
    # number of fields.
    SHAPES = ((1,),)
    DTYPES = (np.int,)


    @classmethod
    def enum_dict_by_name(cls):
        """ """
        if cls.ENUM is None:
            raise NotImplementedError

        d = {}
        for enum in cls.ENUM:
            d[enum.name] = enum.value
        return d

    @classmethod
    def enum_dict_by_value(cls):
        """ """
        if cls.ENUM is None:
            raise NotImplementedError

        d = {}
        for enum in cls.ENUM:
            d[enum.value] = enum
        return d

    @classmethod
    def enum_by_value(cls, enum_value):
        """

        Parameters
        ----------
        enum_value :
            

        Returns
        -------

        """
        d = cls.enum_dict_by_value()
        return d[enum_value]

    @classmethod
    def enum_by_name(cls, enum_name):
        """

        Parameters
        ----------
        enum_name :
            

        Returns
        -------

        """
        d = cls.enum_dict_by_name()
        return d[enum_name]


    @classmethod
    def record(cls, enum_value):
        """

        Parameters
        ----------
        enum_value :
            

        Returns
        -------

        """
        # TODO check to make sure the enum value is valid
        return {'decision_id' : enum_value}

    @classmethod
    def action(cls, walkers, decisions):
        """Perform the instructions for a set of resampling records on
        walkers.

        Parameters
        ----------
        walkers :
            
        decisions :
            

        Returns
        -------

        """
        raise NotImplementedError

    @classmethod
    def parents(cls, step):
        """Given a row of resampling records (for a single resampling step)
        returns the parents of the children of this step.

        Parameters
        ----------
        step :
            

        Returns
        -------

        """

        # initialize a list for the parents of this stages walkers
        step_parents = [None for i in range(len(step))]

        # the rest of the stages parents are based on the previous stage
        for parent_idx, parent_rec in enumerate(step):

            # if the decision is an ancestor then the instruction
            # values will be the children
            if parent_rec[0] in cls.ANCESTOR_DECISION_IDS:
                # the first value of the parent record is the target
                # idxs
                child_idxs = parent_rec[1]
                for child_idx in child_idxs:
                    step_parents[child_idx] = parent_idx

        return step_parents




class NothingDecisionEnum(Enum):
    """ """
    NOTHING = 0

# an example of a Decision class that has the enumeration, instruction
# record namedtuple, and the instruction dtypes
class NoDecision(Decision):
    """ """

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
        """

        Parameters
        ----------
        walkers :
            
        decisions :
            

        Returns
        -------

        """
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
