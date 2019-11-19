"""Abstract base class for Decision classes.

See the NoDecision class and others in this module for examples.

To create your own subclass of the Decision class you must customize
the following class constants:

- ENUM
- FIELDS
- SHAPES
- DTYPE
- RECORD_FIELDS

The 'ENUM' field should be a python 'Enum' class created by
subclassing and customizing 'Enum' in the normal pythonic way. The
elements of the 'Enum' are the actual decision choices, and their
numeric value is used for serialization.

The 'FIELDS' constant is a specification of the number of fields that
a decision record will have. All decision records should contain the
'decision_id' field which is the choice of decision. This class
implements that and should be used as shown in the examples.

In order that fields be serializable to different formats, we also
require that they be a numpy array datatype.

To support this we require the data shapes and data types for each
field. Elements of SHAPES and DTYPES should be of a format
recognizable by the numpy array constructor.

To this we allow the additional option of specifying SHAPES as the
python built-in name Ellipsis (AKA '...'). This will specify the shape
as a variable length 1-dimensional array.

The RECORD_FIELDS is used as a way to specify fields which are
amenable to placement in simplified summary tables, i.e. simple
non-compound values.

The only method that needs to be implemented in the Decision is
'action'.

This function actually implements the algorithm for taking actions on
the decisions and instructions and is called from the resampler to
perform them on the collection of walkers.

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
    """The enumeration of the decision types. Maps them to integers."""

    DEFAULT_DECISION = None
    """The default decision to choose."""

    FIELDS = ('decision_id',)
    """The names of the fields that go into the decision record."""

    # suggestion for subclassing, FIELDS and others
    # FIELDS = super().FIELDS + ('target_idxs',)
    # etc.

    #  An Ellipsis instead of fields indicate there is a variable
    # number of fields.
    SHAPES = ((1,),)
    """Field data shapes."""

    DTYPES = (np.int,)
    """Field data types."""

    RECORD_FIELDS = ('decision_id',)
    """The fields that could be used in a reduced table-like representation."""

    ANCESTOR_DECISION_IDS = None
    """Specify the enum values where their walker state sample value is
    passed on in the next generation, i.e. after performing the action."""

    @classmethod
    def default_decision(cls):
        return cls.DEFAULT_DECISION

    @classmethod
    def field_names(cls):
        """Names of the decision record fields."""
        return cls.FIELDS

    @classmethod
    def field_shapes(cls):
        """Field data shapes."""
        return cls.SHAPES

    @classmethod
    def field_dtypes(cls):
        """Field data types."""
        return cls.DTYPES

    @classmethod
    def fields(cls):
        """Specs for each field.

        Returns
        -------

        fields : list of tuples
            Field specs each spec is of the form (name, shape, dtype).

        """
        return list(zip(cls.field_names(),
                   cls.field_shapes(),
                   cls.field_dtypes()))

    @classmethod
    def record_field_names(cls):
        """The fields that could be used in a reduced table-like representation."""
        return cls.RECORD_FIELDS

    @classmethod
    def enum_dict_by_name(cls):
        """Get the decision enumeration as a dict mapping name to integer."""
        if cls.ENUM is None:
            raise NotImplementedError

        d = {}
        for enum in cls.ENUM:
            d[enum.name] = enum.value
        return d

    @classmethod
    def enum_dict_by_value(cls):
        """Get the decision enumeration as a dict mapping integer to name."""

        if cls.ENUM is None:
            raise NotImplementedError

        d = {}
        for enum in cls.ENUM:
            d[enum.value] = enum
        return d

    @classmethod
    def enum_by_value(cls, enum_value):
        """Get the enum name for an enum_value.

        Parameters
        ----------
        enum_value : int

        Returns
        -------
        enum_name : enum

        """
        d = cls.enum_dict_by_value()
        return d[enum_value]

    @classmethod
    def enum_by_name(cls, enum_name):
        """Get the enum name for an enum_value.

        Parameters
        ----------
        enum_name : enum

        Returns
        -------
        enum_value : int

        """

        d = cls.enum_dict_by_name()
        return d[enum_name]


    @classmethod
    def record(cls, enum_value, **fields):
        """Generate a record for the enum_value and the other fields.

        Parameters
        ----------
        enum_value : int

        Returns
        -------
        rec : dict of str: value

        """

        assert enum_value in cls.enum_dict_by_value(), "value is not a valid Enumerated value"

        for field_key in fields.keys():
            assert field_key in cls.FIELDS, \
                "The field {} is not a field for that decision".format(field_key)
            assert field_key != 'decision_id', "'decision_id' cannot be an extra field"

        rec = {'decision_id' : enum_value}
        rec.update(fields)

        return rec

    @classmethod
    def action(cls, walkers, decisions):
        """Perform the instructions for a set of resampling records on
        walkers.

        The decisions are a collection of decision records which
        contain the decision value and the instruction values for a
        particular walker within its cohort (sample set).

        The collection is organized as a list of lists. The outer list
        corresponds to the steps of resampling within a cycle.

        The inner list is a list of decision records for a specific
        step of resampling, where the index of the decision record is
        the walker index.

        Parameters
        ----------
        walkers : list of Walker objects
            The walkers you want to perform the decision instructions on.

        decisions : list of list of decision records
            The decisions for each resampling step and their
            instructions to apply to the walkers.

        Returns
        -------

        resampled_walkers : list of Walker objects
            The resampled walkers.

        Raises
        ------
        NotImplementedError : abstract method

        """
        raise NotImplementedError

    @classmethod
    def parents(cls, step):
        """Given a step of resampling records (for a single resampling step)
        returns the parents of the children of this step.

        Parameters
        ----------
        step : list of decision records
            The decision records for a step of resampling for each walker.

        Returns
        -------
        walker_step_parents : list of int
            For each element, the index of it in the list corresponds
            to the child index and the value of the element is the
            index of it's parent before the decision action.

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
    """Enumeration of the decision values for doing nothing."""

    NOTHING = 0
    """Do nothing with the walker."""

class NoDecision(Decision):
    """Decision for a resampling process that does no resampling."""

    ENUM = NothingDecisionEnum
    DEFAULT_DECISION = ENUM.NOTHING

    FIELDS = Decision.FIELDS + ('target_idxs',)
    SHAPES = Decision.SHAPES + (Ellipsis,)
    DTYPES = Decision.DTYPES + (np.int,)

    RECORD_FIELDS = Decision.RECORD_FIELDS + ('target_idxs',)

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
