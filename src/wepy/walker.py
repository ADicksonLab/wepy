"""Reference implementations for a general walker and walker state
with utilities for cloning and merging walkers.

Wepy does not require that this or a subclass of this `Walker` or
`WalkerState` is used and only that it is something that acts like it
(duck typed).

The required attributes for the Walker interface are:

- `state` : object implementing the `WalkerState` interface
- `weight` : float

The weight attribute is simply a float and the proper normalization of
weights among a weighted ensemble of walkers should be enforced by the
resampler.

The `clone`, `squash`, and `merge` methods can be accessed by Decision
classes for implementing cloning and merging. As can the module level
`split` and `keep_merge` functions.

The WalkerState interface must have a method `dict` which returns a
dictionary with string keys and arbitrary values.

Additionally the WalkerState should provide its own `__getitem__`
magic method for the accessor syntax, i.e. walker.state['positions'].

"""

import random as rand
import logging
from copy import deepcopy

def split(walker, number=2):
    """Split (AKA make multiple clones) of a single walker.

    Creates multiple new walkers that have the same state as the given
    walker with weight evenly divided between them.

    Parameters
    ----------
    walker : object implementing the Walker interface
        The walker to split/clone
    number : int
        The number of clones to make of the walker
         (Default value = 2)

    Returns
    -------
    cloned_walkers : list of objects implementing the Walker interface

    """
    # calculate the weight of all child walkers split uniformly
    split_prob = walker.weight / (number)
    # make the clones
    clones = []
    for i in range(number):
        clones.append(type(walker)(walker.state, split_prob))

    return clones

def keep_merge(walkers, keep_idx):
    """Merge a set of walkers using the state of one of them.

    Parameters
    ----------
    walkers : list of objects implementing the Walker interface
        The walkers that will be merged together
    keep_idx : int
        The index of the walker in the walkers list that will be used
        to set the state of the new merged walker.

    Returns
    -------
    merged_walker : object implementing the Walker interface

    """

    weights = [walker.weight for walker in walkers]
    # but we add their weight to the new walker
    new_weight = sum(weights)
    # create a new walker with the keep_walker state
    new_walker = type(walkers[0])(walkers[keep_idx].state, new_weight)

    return new_walker

def merge(walkers):
    """Merge this walker with another keeping the state of one of them
    and adding the weights.

    The walker that has it's state kept is a random choice weighted by
    the walkers weights.

    Parameters
    ----------
    walkers : list of objects implementing the Walker interface
        The walkers that will be merged together

    Returns
    -------
    merged_walker : object implementing the Walker interface

    """

    weights = [walker.weight for walker in walkers]
    # choose a walker according to their weights to keep its state
    keep_walker = rand.choices(walkers, weights=weights)
    keep_idx = walkers.index(keep_walker)

    # TODO do we need this?
    # the others are "squashed" and we lose their state
    # squashed_walkers = set(walkers).difference(keep_walker)

    # but we add their weight to the new walker
    new_weight = sum(weights)
    # create a new walker with the keep_walker state
    new_walker = type(walkers[0])(keep_walker.state, new_weight)

    return new_walker, keep_idx

class Walker(object):
    """Reference implementation of the Walker interface.

    A container for:

    - state
    - weight

    """

    def __init__(self, state, weight):
        """Constructor for Walker.

        Parameters
        ----------
        state : object implementing the WalkerState interface

        weight : float

        """

        self.state = state
        self.weight = weight

    def clone(self, number=1):
        """Clone this walker by making a copy with the same state and split
        the probability uniformly between clones.

        The number is the increase in the number of walkers.

        e.g. number=1 will return 2 walkers with the same state as
        this object but with probability split 50/50 between them

        Parameters
        ----------
        number : int
            Number of extra clones to make
             (Default value = 1)

        Returns
        -------
        cloned_walkers : list of objects implementing the Walker interface

        """

        # calculate the weight of all child walkers split uniformly
        split_prob = self.weight / (number+1)
        # make the clones
        clones = []
        for i in range(number+1):
            clones.append(type(self)(self.state, split_prob))

        return clones

    def squash(self, merge_target):
        """Add the weight of this walker to another.

        Parameters
        ----------
        merge_target : object implementing the Walker interface
            The walker to add this one's weight to.

        Returns
        -------
        merged_walker : object implementing the Walker interface

        """
        new_weight = self.weight + merge_target.weight
        return type(self)(merge_target.state, new_weight)

    def merge(self, other_walkers):
        """Merge a set of other walkers into this one using the merge function.

        Parameters
        ----------
        other_walkers : list of objects implementing the Walker interface
            The walkers that will be merged together

        Returns
        -------
        merged_walker : object implementing the Walker interface

        """
        return merge([self]+other_walkers)

class WalkerState(object):
    """Reference implementation of the WalkerState interface.

    Access all key-value pairs as a dictionary with the dict() method.

    Access individual values using the accessor syntax similar to
    dictionaries:

    >>> WalkerState(my_key='value')['my_key']
    'value'

    """

    def __init__(self, **kwargs):
        """Constructor for WalkerState.

        All key-word arguments passed in will be set as the key-value
        pairs for the state.

        """
        self._data = kwargs

    def __getitem__(self, key):
        return self._data[key]

    def dict(self):
        """Return all key-value pairs as a dictionary."""
        return deepcopy(self._data)
