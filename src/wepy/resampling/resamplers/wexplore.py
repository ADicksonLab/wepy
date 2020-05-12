import math
import random as rand
import itertools as it
from collections import namedtuple, defaultdict
from copy import copy, deepcopy

import logging
from eliot import start_action, log_call

import numpy as np
import networkx as nx

from wepy.resampling.resamplers.resampler  import ResamplerError
from wepy.resampling.resamplers.clone_merge  import CloneMergeResampler
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision

class RegionTreeError(Exception):
    """Errors related to violations of constraints in RegionTree algorithms."""
    pass


## Merge methods

# algorithms for finding the number of mergeable walkers in a group
def calc_squashable_walkers_single_method(walker_weights, max_weight):
    """Calculate the maximum number of squashable walkers in collection of
    walkers, that still satisfies the max weight constraint.

    We don't guarantee or know if this is a completely optimal solver
    for this problem but it is assumed to be good enough in practice
    and no harm comes from underestimating it only a reduced potential
    performance.

    Parameters
    ----------
    walker_weights : list of float
        The weights of the walkers

    max_weight : float
        The maximum weight a walker can have.

    Returns
    -------

    n_squashable : int
        The maximum number of squashable walkers.

    """


    # to get an estimate of the number of squashable walkers we start
    # summing the weights starting from the smallest walker. When the
    # addition of the next highest weight walker would make the total
    # greater than max_weight then we quit and say that the number of
    # squashable walkers is the number of them summed up, minus one
    # for the fact that one of them won't be squashed if a merge of
    # all of them was to occur
    n_squashable = 0

    # there must be at least 2 walkers in order to be able to do a
    # merge, so if there are not enough the number of squashable
    # walkers is 0
    if len(walker_weights) < 2:
        return n_squashable


    # sort the weights smallest to biggest
    walker_weights.sort()

    idx = 0
    sum_weights = walker_weights[idx]
    merge_size = 1
    while sum_weights <= max_weight:

        # if the next index would be out of bounds break out of the
        # loop
        if idx + 1 >= len(walker_weights):
            break
        else:
            idx += 1

        # add this walker to the sum weights
        sum_weights += walker_weights[idx]

        # add one to the merge size (since we only will make our
        # estimate based on the single largest possible merge)
        merge_size += 1


    else:
        # the loop condition failed so we remove the last count of
        # merge size from the merge group. This won't run if we break
        # out of the loop because of we are out of walkers to include
        merge_size -= 1


    # then we also take one less than that as the number of
    # squashable walkers
    n_squashable = merge_size - 1


    return n_squashable


# algorithms for actually generating the merge groups
def decide_merge_groups_single_method(walker_weights, balance, max_weight):
    """Use the 'single' method for determining the merge groups.

    Determine a solution to the backpack-like problem of assigning
    squashed walkers to KEEP_MERGE walkers.

    The single method just assigns all squashed walkers in the
    collection to a single walker, thus there is a single merge_group.

    Parameters
    ----------
    walker_weights : list of float
        The weights of the walkers.

    balance : int
        The net change in the number of walkers we desire.

    max_weight : float
        The maximum weight a single walker can be.

    Returns
    -------

    merge_groups : list of list of int
        The merge group solution.

    result : bool
        Whether the merge group exceeds the max weight.

    """

    assert balance < 0, "target balance must be negative"

    # the number of walkers we need to choose in order to be
    # able to do the required amount of merges
    num_merge_walkers = abs(balance) + 1

    # select the lowest weight walkers to use for merging, these
    # are idxs on the mergeable walkers and not the walker_idxs
    chosen_idxs = np.argsort(walker_weights)[:num_merge_walkers]

    # check that this is not greater than the max weight
    if sum([walker_weights[chosen_idx] for chosen_idx in chosen_idxs]) > max_weight:
        result = False
    else:
        result = True

    # return the chosen idxs as the sole full merge group
    return [chosen_idxs], result

## Clone methods
def calc_max_num_clones(walker_weight, min_weight, max_num_walkers):
    """

    Parameters
    ----------
    walker_weight :
        
    min_weight :
        
    max_num_walkers :
        

    Returns
    -------

    """

    # initialize it to no more clones
    max_n_clones = 0

    # start with a two splitting
    n_splits = 2
    # then increase it every time it passes or until we get to the
    # max number of walkers
    while ((walker_weight / n_splits) >= min_weight) and \
          (n_splits <= max_num_walkers):

        n_splits += 1

    # the last step failed so the number of splits is one less
    # then we counted
    n_splits -= 1

    # we want the number of clones so we subtract one from the
    # number of splits to get that, and we save this for this
    # walker
    max_n_clones = n_splits - 1

    return max_n_clones


class RegionTree(nx.DiGraph):
    """Used internally in the WExploreResampler module. Not really
    intended to be used outside this module."""

    # the strings for choosing a method of solving how deciding how
    # many walkers can be merged together given a group of walkers and
    # the associated algorithm for actually choosing them
    MERGE_METHODS = ('single',)

    # Description of the methods

    # 'single' : this method simplifies the problem (likely giving
    # very suboptimal solutions especially early in sampling when
    # walkers are of similar large weights) by enforcing that within a
    # group of walkers (i.e. in a leaf region node) only one merge
    # will take place. To decide how large a given merge group can be
    # then is simply found by consecutively summing the weights of the
    # smallest walkers until the inclusion of the next highest
    # violates the maximum weight. Thus the algorithm for actually
    # finding the walkers that shall be merged is as simple as taking
    # the K lowest walkers given by the first algorithm. This is then
    # guaranteed to satisfy the potential.

    # as further methods are mathematically proven and algorithms
    # designed this will be the chosen method.

    ROOT_NODE = ()

    def __init__(self, init_state,
                 max_n_regions=None,
                 max_region_sizes=None,
                 distance=None,
                 pmin=None, pmax=None,
                 merge_method='single'):

        super().__init__()

        if (max_n_regions is None) or \
           (max_region_sizes is None) or \
           (distance is None) or \
           (pmin is None) or \
           (pmax is None):
            raise ValueError("All parameters must be defined, 1 or more are missing.")

        self._max_n_regions = max_n_regions
        self._n_levels = len(max_n_regions)
        self._max_region_sizes = max_region_sizes
        self._distance = distance
        self._pmin = pmin
        self._pmax = pmax

        # initialize the max and min number of walkers, this is a
        # dynamic thing and is manually set by the WExploreResampler
        self._max_num_walkers = False
        self._min_num_walkers = False

        assert merge_method in self.MERGE_METHODS, \
            "the merge method given, '{}', must be one of the methods available {}".format(
                merge_method, self.MERGE_METHODS)

        self._merge_method = merge_method

        self._walker_weights = []
        self._walker_assignments = []

        image_idx = 0
        # get the image using the distance object
        image = self.distance.image(init_state)
        self._images = [image]

        parent_id = self.ROOT_NODE
        self.add_node(parent_id, image_idx=0,
                      n_walkers=0,
                      n_squashable=0,
                      n_possible_clones=0,
                      balance=0,
                      walker_idxs=[])

        # make the first branch
        for level in range(len(max_n_regions)):
            child_id = parent_id + (0,)
            self.add_node(child_id, image_idx=image_idx,
                          n_walkers=0,
                          n_squashable=0,
                          n_possible_clones=0,
                          balance=0,
                          walker_idxs=[])
            self.add_edge(parent_id, child_id)
            parent_id = child_id

        # add the region for this branch to the regions list
        self._regions = [tuple([0 for i in range(self._n_levels)])]

    @property
    def merge_method(self):
        """ """
        return self._merge_method

    @property
    def distance(self):
        """ """
        return self._distance

    @property
    def images(self):
        """ """
        return self._images

    @property
    def max_n_regions(self):
        """ """
        return self._max_n_regions

    @property
    def n_levels(self):
        """ """
        return self._n_levels

    @property
    def max_region_sizes(self):
        """ """
        return self._max_region_sizes

    @property
    def pmin(self):
        """ """
        return self._pmin

    @property
    def pmax(self):
        """ """
        return self._pmax

    @property
    def walker_assignments(self):
        """ """
        return self._walker_assignments

    @property
    def walker_weights(self):
        """ """
        return self._walker_weights

    @property
    def regions(self):
        """ """
        return self._regions

    def add_child(self, parent_id, image_idx):
        """

        Parameters
        ----------
        parent_id :
            
        image_idx :
            

        Returns
        -------

        """
        # make a new child id which will be the next index of the
        # child with the parent id
        child_id = parent_id + (len(self.children(parent_id)), )

        # create the node with the image_idx
        self.add_node(child_id,
                      image_idx=image_idx,
                      n_walkers=0,
                      n_squashable=0,
                      n_possible_clones=0,
                      balance=0,
                      walker_idxs=[])

        # make the edge to the child
        self.add_edge(parent_id, child_id)

        return child_id

    def children(self, parent_id):
        """

        Parameters
        ----------
        parent_id :
            

        Returns
        -------

        """
        children_ids = list(self.adj[parent_id].keys())
        # sort them
        children_ids.sort()
        return children_ids

    def level_nodes(self, level):
        """Get the nodes/regions at the specified level.

        Parameters
        ----------
        level :
            

        Returns
        -------

        """

        if level > self.n_levels:
            raise ValueError("level is greater than the number of levels for this tree")

        return [node_id for node_id in self.nodes
                if len(node_id) == level]

    def leaf_nodes(self):
        """ """
        return self.level_nodes(self.n_levels)

    def branch_tree(self, parent_id, image):
        """

        Parameters
        ----------
        parent_id :
            
        image :
            

        Returns
        -------

        """
        # add the new image to the image index
        image_idx = len(self._images)
        self._images.append(image)

        branch_level = len(parent_id)
        # go down from there and create children
        for level in range(branch_level, self.n_levels):
            child_id = self.add_child(parent_id, image_idx)
            parent_id = child_id

        #add new assignment  to the image assignments
        self._regions.append(child_id)
        # return the leaf node id of the new branch
        return child_id


    @property
    def max_num_walkers(self):
        """ """
        return self._max_num_walkers

    @max_num_walkers.setter
    def max_num_walkers(self, max_num_walkers):
        """This must be an integer.

        Parameters
        ----------
        max_num_walkers :
            

        Returns
        -------

        """

        self._max_num_walkers = max_num_walkers

    @max_num_walkers.deleter
    def max_num_walkers(self, max_num_walkers):
        """This must be an integer.

        Parameters
        ----------
        max_num_walkers :
            

        Returns
        -------

        """

        self._max_num_walkers = None

    @property
    def min_num_walkers(self):
        """ """
        return self._min_num_walkers

    @min_num_walkers.setter
    def min_num_walkers(self, min_num_walkers):
        """This must be an integer.

        Parameters
        ----------
        min_num_walkers :
            

        Returns
        -------

        """

        self._min_num_walkers = min_num_walkers

    @min_num_walkers.deleter
    def min_num_walkers(self, min_num_walkers):
        """This must be an integer.

        Parameters
        ----------
        min_num_walkers :
            

        Returns
        -------

        """

        self._min_num_walkers = None


    def assign(self, state):
        """

        Parameters
        ----------
        state :
            

        Returns
        -------

        """

        assignment = []
        dists = []

        # a cache for the distance calculations so they need not be
        # performed more than once
        dist_cache = {}

        # perform a n-ary search through the hierarchy of regions by
        # performing a distance calculation to the images at each
        # level starting at the top
        node = self.ROOT_NODE
        for level in range(self.n_levels):
            level_nodes = self.children(node)

            # perform a distance calculation to all nodes at this
            # level
            image_dists = []
            for level_node in level_nodes:

                # get the image
                image_idx = self.node[level_node]['image_idx']
                image = self.images[image_idx]

                # if this distance is already calculated don't
                # calculate it again and just get it from the cache
                if image_idx in dist_cache:
                    dist = dist_cache[image_idx]
                # otherwise calculate it and save it in the cache
                else:
                    # image of the state
                    state_image = self.distance.image(state)

                    # there is the possibility of
                    try:
                        dist = self.distance.image_distance(state_image, image)
                    except ValueError:
                        print("state: ", state.dict())
                        print("state_image: ", state_image)
                        print("image: ", image)
                        raise ValueError("If you have triggered this error you have"
                                         " encountered a rare bug. Please attempt to"
                                         " report this using the printed outputs.")

                    # save in the dist_cache
                    dist_cache[image_idx] = dist

                # add it to the dists for this state
                image_dists.append(dist)

            # get the index of the image that is closest
            level_closest_child_idx = np.argmin(image_dists)
            # get the distance for the closest image
            level_closest_image_dist = image_dists[level_closest_child_idx]

            # save for return
            assignment.append(level_closest_child_idx)
            dists.append(level_closest_image_dist)

            # set this node as the next node
            node = level_nodes[level_closest_child_idx]

        return tuple(assignment), tuple(dists)

    def clear_walkers(self):
        """Remove all walkers from the regions."""

        # reset the walker assignments to an empty list
        self._walker_assignments = []
        self._walker_weights = []

        # set all the node attributes to their defaults
        for node_id in self.nodes:
            self.node[node_id]['n_walkers'] = 0
            self.node[node_id]['walker_idxs'] = []

            self.node[node_id]['n_squashable'] = 0
            self.node[node_id]['n_possible_clones'] = 0
            self.node[node_id]['balance'] = 0


    def place_walkers(self, walkers):
        """

        Parameters
        ----------
        walkers :
            

        Returns
        -------

        """

        # clear all the walkers and reset node attributes to defaults
        self.clear_walkers()

        # keep track of new branches made
        new_branches = []

        # place each walker
        for walker_idx, walker in enumerate(walkers):

            # assign the state of the walker to the tree and get the
            # distances to the images at each level
            assignment, distances = self.assign(walker.state)

            # check the distances going down the levels to see if a
            # branching (region creation) is necessary
            for level, distance in enumerate(distances):

                # if we are over the max region distance and we are
                # not above max number of regions we have found a new
                # region so we branch the region_tree at that level
                if distance > self.max_region_sizes[level] and \
                   len(self.children(assignment[:level])) < self.max_n_regions[level]:

                    # make an image for the region
                    image = self.distance.image(walker.state)
                    parent_id = assignment[:level]

                    # make the new branch
                    assignment = self.branch_tree(parent_id, image)

                    # save it to keep track of new branches as they occur
                    new_branches.append({'distance' : np.array([distance]),
                                         'branching_level' : np.array([level]),
                                         'new_leaf_id' : np.array(assignment),
                                         'image' : image,})

                    # we have made a new branch so we don't need to
                    # continue this loop
                    break

            # save the walker assignment
            self._walker_assignments.append(assignment)
            self._walker_weights.append(walker.weight)

            # go back through the nodes in this walker's branch
            # increase the n_walkers for each node, and save the
            # walkers (index in self.walker_assignments) it has, and
            # save increase the number above pmin if valid
            for level in range(len(assignment) + 1):
                node_id = assignment[:level]

                self.node[node_id]['n_walkers'] += 1
                self.node[node_id]['walker_idxs'].append(walker_idx)

        # We also want to find out some details about the ability of
        # the leaf nodes to clone and merge walkers. This is useful
        # for being able to balance the tree. Once this has been
        # figured out for the leaf nodes we want to aggregate these
        # numbers for the higher level regions
        for node_id in self.leaf_nodes():

            leaf_walker_idxs = self.node[node_id]['walker_idxs']
            leaf_weights = [self.walker_weights[i] for i in leaf_walker_idxs]

            # first figure out how many walkers are squashable (AKA
            # reducible)
            n_squashable = self._calc_squashable_walkers(leaf_weights)

            # get the max number of clones for each walker and sum
            # them up to get the total number of cloneable walkers
            walker_max_n_clones = [self._calc_max_num_clones(walker_weight)
                                   for walker_weight in leaf_weights]
            n_possible_clones = sum(walker_max_n_clones)

            # actually set them as attributes for the node
            self.node[node_id]['n_squashable'] = n_squashable
            self.node[node_id]['n_possible_clones'] = n_possible_clones

            # also add this amount to all of the nodes above it

            # n_squashable
            for level in reversed(range(self.n_levels)):
                branch_node_id = node_id[:level]
                self.node[branch_node_id]['n_squashable'] += n_squashable

            # n_posssible_clones
            for level in reversed(range(self.n_levels)):
                branch_node_id = node_id[:level]
                self.node[branch_node_id]['n_possible_clones'] += n_possible_clones

        return new_branches

    @classmethod
    def _max_n_merges(cls, pmax, root, weights):
        """

        Parameters
        ----------
        pmax :
            
        root :
            
        weights :
            

        Returns
        -------

        """

        # indices of the weights
        walker_idxs = [i for i, weight in enumerate(weights)]

        # remove the root from the weights
        unused_walker_idxs = list(set(walker_idxs).difference(root))

        # initialize the number of merges identified by the length of
        # the current root
        max_n_merges = len(root) - 1


        # then combine the root with the unused weights
        for root, merge_candidate in it.product([root], unused_walker_idxs):

            # get the weights for this combo
            combo_weights = [weights[i] for i in root] + [weights[merge_candidate]]

            # sum them
            sum_weight = sum(combo_weights)

            # if the sum of the weights is less than or equal than the
            # pmax then this combination beats the current record of
            # the root
            if sum_weight <= pmax:

                # then we know that the number of merges is at least
                # one more than the root
                max_n_merges += 1

                # if we still haven't reached the pmax continue making
                # merges to see if we can beat this record
                if sum_weight < pmax:

                    # make a new root for this combo and recursively call
                    # this method
                    new_combo = (*root, merge_candidate,)

                    # this will return the maximum number of merges from
                    # this subset of the walkers
                    n_merges = cls._max_n_merges(pmax, new_combo, weights)

                    # if this is greater than the current record
                    # overwrite it
                    if n_merges > max_n_merges:
                        max_n_merges = n_merges

                # if it is exactly pmax then no more merges can be
                # done so we can just end here and return this record
                elif sum_weight == pmax:
                    break


        # if no combination of this root and other candidates can make
        # any more merges than we just return the roots number of merges

        return max_n_merges

    def _calc_squashable_walkers(self, walker_weights):
        """

        Parameters
        ----------
        walker_weights :
            

        Returns
        -------

        """

        if self.merge_method == 'single':
            n_squashable = calc_squashable_walkers_single_method(walker_weights, self.pmax)
        else:
            raise ValueError("merge method {} not recognized".format(self.merge_method))

        return n_squashable

    def _calc_max_num_clones(self, walker_weight):
        """

        Parameters
        ----------
        walker_weight :
            

        Returns
        -------

        """

        return calc_max_num_clones(walker_weight, self.pmin, self.max_num_walkers)

    def _propagate_and_balance_shares(self, parental_balance, children_node_ids):
        """

        Parameters
        ----------
        parental_balance :
            
        children_node_ids :
            

        Returns
        -------

        """


        # talk about "shares" which basically are the number of
        # slots/replicas that will be allocated to this region for
        # running sampling on

        # we get the current number of shares for each child
        orig_children_shares = {child_id : len(self.node[child_id]['walker_idxs'])
                           for child_id in children_node_ids}

        # the copy to use as a tally of the shares
        children_shares = copy(orig_children_shares)

        # the donatable (squashable) walkers to start with
        children_donatable_shares = {child_id : self.node[child_id]['n_squashable']
                                     for child_id in children_node_ids}

        # the donatable (squashable) walkers to start with
        children_receivable_shares = {child_id : self.node[child_id]['n_possible_clones']
                                     for child_id in children_node_ids}

        # Our first goal in this subroutine is to dispense a parental
        # balance to it's children in a simply valid manner
        children_dispensations = self._dispense_parental_shares(
                                           parental_balance, children_shares,
                                           children_donatable_shares,
                                           children_receivable_shares)

        for child_id, dispensation in children_dispensations.items():

            # update the shares, donatables, and receivables which we
            # will then balance between regions
            children_shares[child_id] += dispensation


            # add the dispensation to the number of the donatable
            # shares
            children_donatable_shares[child_id] += dispensation

            # subtract the dispensation from the number of receivable
            # shares
            children_receivable_shares[child_id] -= dispensation

        # Now that we have dispensed the shares to the children in a
        # valid way we use an algorithm to now distribute the shares
        # between the regions as evenly as possible
        children_shares = self._balance_children_shares(children_shares,
                                                        children_donatable_shares,
                                                        children_receivable_shares)


        # calculate the net change in the balances for each region
        net_balances = {child_id : children_shares[child_id] - orig_children_shares[child_id]
                        for child_id in children_shares.keys()}

        children_balances = [balance for balance in net_balances.values()]
        if sum(children_balances) != parental_balance:

            raise RegionTreeError(
                "The balances of the child nodes ({}) do not balance to the parental balance ({})".format(
                    children_balances, parental_balance))

        # no state changes to the object have been made up until this
        # point, but now that the net change in the balances for the
        # children have been generated we set them into their nodes
        for child_node_id, child_net_balance in net_balances.items():

            self.node[child_node_id]['balance'] = child_net_balance


    def _dispense_parental_shares(self, parental_balance, children_shares,
                                  children_donatable_shares,
                                  children_receivable_shares):
        """Given a parental balance and a set of children nodes, we dispense
        the shares indicated by the balance to the children nodes in a
        VALID but not necessarily optimal or desirable way. This
        merely checks for the hard constraints on the number of shares
        a region can either give or receive based on their capacity to
        clone and merge walkers.
        
        An additional balancing step can be performed to redistribute them.

        Parameters
        ----------
        parental_balance :
            
        children_shares :
            
        children_donatable_shares :
            
        children_receivable_shares :
            

        Returns
        -------

        """

        # this will be the totaled up dispensations for each child
        # region
        children_dispensations = {child_id : 0 for child_id in children_shares.keys()}

        # if there is only one child it just inherits all of the
        # balance no matter what
        if len(children_shares.keys()) == 1:

            child_node_id = list(children_shares.keys())[0]

            # we put the shares for this only child in a dictionary
            # like the other methods would produce
            children_dispensations[child_node_id] = parental_balance

        # there are more than one child so we accredit balances
        # between them
        elif len(children_shares.keys()) > 1:

            # if the parent has a non-zero balance we either
            # increase (clone) or decrease (merge) the balance

            # these poor children are inheriting a debt and must
            # decrease the total number of their shares :(
            if parental_balance < 0:

                children_dispensations = self._dispense_debit_shares(parental_balance,
                                                                     children_shares,
                                                                     children_donatable_shares)

            # these lucky children are inheriting a positive number of
            # shares!! :)
            elif parental_balance > 0:

                children_dispensations = self._dispense_credit_shares(parental_balance,
                                                                      children_shares,
                                                                      children_receivable_shares)

        else:
            raise RegionTreeError("no children nodes to give parental balance")

        return children_dispensations

    def _dispense_debit_shares(self, parental_balance, children_shares,
                               children_donatable_shares):
        """For a negative parental balance we dispense it to the children
        nodes

        Parameters
        ----------
        parental_balance :
            
        children_shares :
            
        children_donatable_shares :
            

        Returns
        -------

        """

        children_donatable_shares = copy(children_donatable_shares)

        children_dispensations = {child_id : 0 for child_id in children_shares.keys()}

        # list of the keys so we can iterate through them
        children_node_ids = list(children_shares.keys())

        # dispense the negative shares as quickly as possible,
        # they will be balanced later
        child_iter = iter(children_node_ids)
        remaining_balance = parental_balance
        while remaining_balance < 0:
            # get the node id
            try:
                child_node_id = next(child_iter)
            except StopIteration:
                # if the parental balance is still not zero and there
                # are no more children then the children cannot
                # balance it given their constraints and there is an
                # error
                raise RegionTreeError("Children cannot pay their parent's debt")

            n_donatable = children_donatable_shares[child_node_id]

            # if this child has any squashable walkers
            if n_donatable > 0:

                # we use those for paying the parent's debt

                # the amount of the parental debt that can be
                # paid (the payment) for this child region is
                # either the number of squashable walkers or
                # the absolute value of the parental balance
                # (since it is negative for debts), whichever
                # is smaller
                payment = min(n_donatable, abs(remaining_balance))

                # take this from the remaining balance
                remaining_balance += payment

                # and take it away from the childs due balance and shares
                children_dispensations[child_node_id] -= payment

                # also account for that in its donatable shares
                children_donatable_shares[child_node_id] -= payment

        # double check the balance is precisely 0, we want to
        # dispense all the shares as well as not accidentally
        # overdispensing
        assert remaining_balance == 0, "balance is not 0"


        return children_dispensations

    def _dispense_credit_shares(self, parental_balance, children_shares,
                                children_receivable_shares):
        """

        Parameters
        ----------
        parental_balance :
            
        children_shares :
            
        children_receivable_shares :
            

        Returns
        -------

        """

        children_receivable_shares = copy(children_receivable_shares)

        children_dispensations = {child_id : 0 for child_id in children_shares.keys()}

        # list of the keys so we can iterate through them
        children_node_ids = list(children_shares.keys())

        # dispense the shares to the able children as quickly
        # as possible, they will be redistributed in the next
        # step
        child_iter = iter(children_node_ids)
        remaining_balance = parental_balance
        while remaining_balance > 0:

            # get the node id
            try:
                child_node_id = next(child_iter)
            except StopIteration:
                # if the parental balance is still not zero and there
                # are no more children then the children cannot
                # balance it given their constraints and there is an
                # error
                raise RegionTreeError("Children cannot accept their parent's credit")

            # give as much of the parental balance as we can
            # to the walkers. In the next step all this
            # balance will be shared among the children so all
            # we need to do is dispense all the shares without
            # care as to who gets them, as long as they can
            # keep it
            n_receivable = children_receivable_shares[child_node_id]

            # the amount to be disbursed to this region is
            # either the number of possible clones (the
            # maximum it can receive) or the full parental
            # balance, whichever is smaller
            disbursement = min(n_receivable, abs(remaining_balance))

            # give this disbursement by taking away from the
            # positive balance
            remaining_balance -= disbursement

            # add these shares to the net balances and share
            # totals
            children_dispensations[child_node_id] += disbursement

            # also subtract this from the number of receivable shares
            # for the child
            children_receivable_shares[child_node_id] -= disbursement

        # double check the balance is precisely 0, we want to
        # dispense all the shares as well as not accidentally
        # overdispensing
        assert remaining_balance == 0, "balance is not 0"

        return children_dispensations

    def _balance_children_shares(self, children_shares,
                                 children_donatable_shares,
                                 children_receivable_shares):
        """Given a dictionary mapping the child node_ids to the total number
        of shares they currently hold we balance between them in order
        to get an even distribution of the shares as possible.

        Parameters
        ----------
        children_shares :
            
        children_donatable_shares :
            
        children_receivable_shares :
            

        Returns
        -------

        """

        children_shares = copy(children_shares)

        # generate the actual donation pair and the amount that should
        # be donated for the best outcome
        donor_node_id, acceptor_node_id, donation_amount = \
                                    self._gen_best_donation(children_shares,
                                                            children_donatable_shares,
                                                            children_receivable_shares)

        # if the donation amount is zero we make no donation
        if donation_amount > 0:

            # account for this donation in the shares
            children_shares[donor_node_id] -= donation_amount
            children_shares[acceptor_node_id] += donation_amount

            # subtract the donation donatable_shares from the donor
            # and add the donation to the donatable_shares of the
            # acceptor
            children_donatable_shares[donor_node_id] -= donation_amount
            children_donatable_shares[acceptor_node_id] += donation_amount

            # do the opposite to the receivable shares
            children_receivable_shares[donor_node_id] += donation_amount
            children_receivable_shares[acceptor_node_id] -= donation_amount

        # we have decided the first donation, however more will be
        # performed as long as the amount of the donation is either 0
        # or that two donations of only 1 share occur twice in a
        # row. The former occurs in scenarios when there is an even
        # balance and the latter in an odd scenario and the last odd
        # share would get passed back and forth

        # we keep track of the previous donation, and initialize it to
        # None for now
        previous_donation_amount = donation_amount

        while (donation_amount > 0) and \
              not (previous_donation_amount == 1 and donation_amount == 1):

            # update the previous donation amount
            previous_donation_amount = donation_amount

            # get the next best donation
            donor_node_id, acceptor_node_id, donation_amount = \
                                            self._gen_best_donation(children_shares,
                                                                    children_donatable_shares,
                                                                    children_receivable_shares)

            # if there is a donation to be made make it
            if donation_amount > 0:

                # account for this donation in the shares
                children_shares[donor_node_id] -= donation_amount
                children_shares[acceptor_node_id] += donation_amount

                # subtract the donation donatable_shares from the donor
                # and add the donation to the donatable_shares of the
                # acceptor
                children_donatable_shares[donor_node_id] -= donation_amount
                children_donatable_shares[acceptor_node_id] += donation_amount

                # do the opposite to the receivable shares
                children_receivable_shares[donor_node_id] += donation_amount
                children_receivable_shares[acceptor_node_id] -= donation_amount



        return children_shares


    def _gen_best_donation(self, children_shares,
                                 children_donatable_shares,
                                 children_receivable_shares):
        """Given a the children shares generate the best donation. Returns the
        donor_node_id the acceptor_node_id and the donation that
        should be done between them and that will be guaranteed to be
        valid. (this is done by checking the attributes of the regions
        node however, no changes to node state are performed)
        
        returns donor_node_id, acceptor_node_id, donation_amount

        Parameters
        ----------
        children_shares :
            
        children_donatable_shares :
            
        children_receivable_shares :
            

        Returns
        -------

        """

        # to find the best possible donation we would like to give
        # shares from the region with the most to the region with the
        # least and give as many as possible that will equalize them,
        # however the size of a donation is dependent not only on the
        # amount of shares each region has but also the number of
        # squashable walkers and the number of possible clones that
        # satisfy the maximum and minimum walker weight
        # constraints. These are given by the
        # children_donatable_shares and children_receivable_shares. We
        # use arguments instead of accessing the attributes of the
        # object because so this can be done in an iterative manner
        # before modifying the node attributes.

        # default values for the results
        best_pair = (None, None)
        best_donation_amount = 0

        # if there are not enough children regions to acutally make
        # pairings between then we just return the default negative
        # result
        if len(children_shares) < 2:
            donor_node_id, acceptor_node_id = best_pair
            return donor_node_id, acceptor_node_id, best_donation_amount


        # we want to get the list of pairings where each pairing is
        # (donor, acceptor)
        pairings = []
        for a, b in it.combinations(children_shares.keys(), 2):


            # find the largest difference comparing (a,b) and (b,a),
            # this will give the donor, acceptor pair
            permutations = [(a,b), (b,a)]
            perm_idx = np.argmax([children_shares[i] - children_shares[j]
                               for i, j in permutations])

            donor_acceptor_pair = permutations[perm_idx]

            pairings.append(donor_acceptor_pair)

        # to find the best match we first calculate the differences in
        # the number of shares for each possible pairing between
        # children shares
        pairings_differences = [children_shares[donor_id] - children_shares[acceptor_id]
                                for donor_id, acceptor_id in pairings]

        pairings_donations = []
        # then we find all the non-zero pairings
        for i, difference in enumerate(pairings_differences):

            # if there is a positive difference then we calculate what
            # the largest donation would be
            if difference > 0:

                donor_node_id, acceptor_node_id = pairings[i]

                # get the total numbers of shares for each
                donor_n_shares = children_shares[donor_node_id]
                acceptor_n_shares = children_shares[acceptor_node_id]

                # as well as the donatable and receivable shares
                donor_donatable_shares = children_donatable_shares[donor_node_id]
                acceptor_receivable_shares = children_receivable_shares[acceptor_node_id]

                # actually calculate the maximum donation
                donation_amount = self._calc_share_donation(donor_n_shares, acceptor_n_shares,
                                                            donor_donatable_shares,
                                                            acceptor_receivable_shares)

                pairings_donations.append(donation_amount)

            # if there is no difference then the donation amount will also be zero
            else:
                pairings_donations.append(0)

        # now we can zip them all together and then sort them such
        # that we first sort on the size of the number of shares and
        # then on the size of the donation
        pair_values = list(zip(pairings_differences, pairings_donations, pairings))
        pair_values.sort()
        # largest to smallest
        pair_values.reverse()

        # now we take the pairing that has the highest difference and
        # has a nonzero donation size. Note there may be other
        # pairings with the same numbers that will just be ignored.
        pair_iter = iter(pair_values)


        # loop through until the first best donation is found.
        # get the first pairing
        try:
            diff, donation, pairing = next(pair_iter)
        except StopIteration:
            raise RegionTreeError("No pairings to make donations between")

        done = False
        while not done:

            # since the pair_values are sorted first by the total diff
            # and then the donation size, the first one that has a
            # positive donation is the best donation
            if donation > 0:
                best_pair = pairing
                best_donation_amount = donation
                done = True

            # try to get the next pairing if there is one
            try:
                diff, donation, pairing = next(pair_iter)
            except StopIteration:
                # we are done, use the last pairing we had as the
                # pair, its just as good as any of the others let the
                # calling method decide what to do in this situation
                best_pair = pairing
                break


        # now we have the best donation and the pair is already in the
        # donor, acceptor order from when we created it
        donor_node_id, acceptor_node_id = best_pair

        return donor_node_id, acceptor_node_id, best_donation_amount


    def _find_best_donation_pair(self, children_donatable_shares,
                                       children_receivable_shares):
        """This method just returns which children have the most and least
        number of 'shares' which are the effective number of walker
        slots it will be granted in the next segment of dynamics in
        the simulation. This is essentially the amount of sampling
        effort that will be allocated to this region.
        
        This method is give the dictionary of the childrens

        Parameters
        ----------
        children_donatable_shares :
            
        children_receivable_shares :
            

        Returns
        -------

        """

        # one region will be the donor of shares
        donor_child_node_id = None

        # the other will accept them
        acceptor_child_node_id = None

        # record for maximum number of donateable shares
        max_donatable_shares = None

        # records for the max and min number of shares of the acceptor
        # and donor regions
        donor_n_shares = None
        acceptor_n_shares = None

        # go through every walker and test it to see if it is either
        # the highest or lowest, record it if it is
        for child_node_id in children_donatable_shares.keys():

            # the number of donatable shares is equal to the number
            # of squashable walkers
            n_donatable_shares = children_donatable_shares[child_node_id]

            # the number of possible shares this node can receive is
            # equal to the number of possible clones it can make
            n_receivable_shares = children_receivable_shares[child_node_id]

            # we see if this node region is the max region by testing
            # if it is the new highest in shares. It must also be able
            # to donate a share by having at least 1 squashable walker
            if ((donor_child_node_id is None) or (n_shares >  donor_n_shares)) and \
               (n_donatable_shares > 0):

                # this is a new record
                max_donatable_shares = n_donatable_shares

                # save how many shares this region has in total
                donor_n_shares = n_shares
                donor_child_node_id = child_node_id


            # test if this is the region with the lowest number of
            # shares that is still able to receive at least one share
            if ((acceptor_child_node_id is None) or (n_shares < acceptor_n_shares)) and \
               (n_receivable_shares > 0):

                acceptor_n_shares = n_shares
                acceptor_child_node_id = child_node_id

        # check that both a donor and acceptor were identified and
        # that values for there shares were given
        assert all([True if val is not None else False
                    for val in [donor_n_shares, acceptor_n_shares,
                                donor_child_node_id, acceptor_child_node_id]]), \
                "A donor or acceptor was not found"

        # if the acceptor's number of shares is not less then the
        # donor then there is not possible donation
        if acceptor_n_shares >= donor_n_shares:
            return False
        # if there is a net donation we return the donor and acceptor
        else:
            return donor_child_node_id, acceptor_child_node_id

    def _calc_share_donation(self, donor_n_shares, acceptor_n_shares,
                                   donor_donatable_shares, acceptor_receivable_shares):
        """

        Parameters
        ----------
        donor_n_shares :
            
        acceptor_n_shares :
            
        donor_donatable_shares :
            
        acceptor_receivable_shares :
            

        Returns
        -------

        """

        # the sibling with the greater number of shares (from both
        # previous resamplings and inherited from the parent) will
        # give shares to the sibling with the least.

        # To decide how many it shall give we first propose a desired
        # donation that will make them the most similar, rounding down
        # (i.e. midpoint)
        desired_donation = math.floor((donor_n_shares - acceptor_n_shares)/2)

        # however, the donor only has a certain capability of donation
        # and the acceptor has a certain capacity of receiving. Out of
        # the three we can only actually donate the smallest amount
        actual_donation = min(desired_donation,
                              donor_donatable_shares,
                              acceptor_receivable_shares)

        return actual_donation

    def _decide_merge_leaf(self, leaf, merge_groups):
        """

        Parameters
        ----------
        leaf :
            
        merge_groups :
            

        Returns
        -------

        """

        # this method assumes no cloning has been performed before this

        # TODO: potentially unneeded
        # all the walker idxs
        walker_idxs = list(range(len(merge_groups)))

        # the balance of this leaf
        leaf_balance = self.node[leaf]['balance']

        # there should not be any taken walkers in this leaf since a
        # leaf should only have this method run for it once during
        # decision making, so the mergeable walkers are just all the
        # walkers in this leaf
        leaf_walker_idxs = self.node[leaf]['walker_idxs']
        leaf_walker_weights = [self.walker_weights[walker_idx] for walker_idx in leaf_walker_idxs]


        # now that we have the walkers that may potentially be merged
        # we need to actually find a set of groupings that satisfies
        # the reduction in balance without any individual walker
        # exceeding the maximum weight. In general this is a difficult
        # problem both here and in deciding how balances are
        # distributed (because the potential merges determine a leafs
        # ability to pay a portion of a debt from a higher level in
        # the region tree).

        # currently we avoid this general problem (potentially of the
        # backpack kind if you want to help solve this issue) and
        # simply assume that we will perform a single merge of the
        # walkers of the lowest weights to achieve our balance
        # reduction goal. As long as this assumption holds in how the
        # balances are determined this will succeed, if not this will
        # fail

        # to allow for easy improvements later on pending this problem
        # becoming solved it is functionalized here to make a set of
        # pairings that satisfy the balance reduction goal, these are
        # "merge groups" except that at this point we haven't chosen
        # one to be the KEEP_MERGE walker and have its state
        # retained. This will be decided further on. So these "full
        # merge groups" include all the walkers that will be merged
        # and the sum of their weights will be the weight of the final
        # merged walker and should satisfy the maximum weight
        # requirement, i.e. it will not be checked here.
        full_merge_groups_leaf_walker_idxs = \
                                self._solve_merge_groupings(leaf_walker_weights, leaf_balance)

        # now we go through each of these "full merge groups" and make
        # the "merge groups". Pardon the terminology, but the
        # distinction is trivial and is only relevant to the
        # implementation. The "merge groups" are what is returned. To
        # make them we just choose which walker to keep and which
        # walkers to squash in each full merge group
        for full_merge_group_leaf_walker_idxs in full_merge_groups_leaf_walker_idxs:

            # the indices from this are in terms of the list of weights
            # given to the method so we translate them back to the actual
            # walker indices
            chosen_walker_idxs = [leaf_walker_idxs[leaf_walker_idx]
                                  for leaf_walker_idx in full_merge_group_leaf_walker_idxs]

            # get the weights of these chosen_walker_idxs
            chosen_weights = [self.walker_weights[walker_idx] for walker_idx in chosen_walker_idxs]

            # choose the one to keep the state of (e.g. KEEP_MERGE
            # in the Decision) based on their weights

            # normalize weights to the sum of all the chosen weights
            chosen_pdist = np.array(chosen_weights) / sum(chosen_weights)

            # then choose one of the the walker idxs to keep according to
            # their normalized weights
            keep_walker_idx = np.random.choice(chosen_walker_idxs, 1, p=chosen_pdist)[0]

            # pop the keep idx from the walkers so we can use the rest of
            # them as the squash idxs
            chosen_walker_idxs.pop(chosen_walker_idxs.index(keep_walker_idx))

            # the rest are squash_idxs
            squash_walker_idxs = chosen_walker_idxs

            # update the merge group based on this decision
            merge_groups[keep_walker_idx].extend(squash_walker_idxs)

        return merge_groups

    def _solve_merge_groupings(self, walker_weights, balance):
        """

        Parameters
        ----------
        walker_weights :
            
        balance :
            

        Returns
        -------

        """

        # this method chooses between the methods for solving the
        # backpack problem of how to merge walkers together to satisfy
        # a goal

        # as a method for easy transition between potential methods (I
        # expect there are multiple solutions to the problem with
        # different tradeoffs that will want to be tested) a method
        # can be chosen when the region tree is created and a constant
        # string identifier will be set indicating which method is in use

        # so we use that string to find which method to use
        if self.merge_method == 'single':

            full_merge_groups, result = decide_merge_groups_single_method(
                walker_weights, balance, self.pmax)

        else:
            raise ValueError("merge method {} not recognized".format(self.merge_method))

        # if the result came out false then a solution could not be
        # found
        if not result:

            raise RegionTreeError(
                "A solution to the merging problem could not be found given the constraints")

        else:
            return full_merge_groups

    def _decide_clone_leaf(self, leaf, merge_groups, walkers_num_clones):
        """

        Parameters
        ----------
        leaf :
            
        merge_groups :
            
        walkers_num_clones :
            

        Returns
        -------

        """

        # just follow the instructions in the walkers_num_clones and
        # find them slots

        # this assumes that the all squashes have already been
        # specified in the merge group, this is so that we can use
        # unused walker slots.

        # if this leaf node was assigned a debt we need to merge
        # walkers
        leaf_balance = self.node[leaf]['balance']
        leaf_walker_idxs = self.node[leaf]['walker_idxs']
        leaf_walker_weights = {walker_idx : self.walker_weights[walker_idx]
                               for walker_idx in self.node[leaf]['walker_idxs']}

        # calculate the maximum possible number of clones each free walker
        # could produce
        walker_n_possible_clones = {walker_idx : self._calc_max_num_clones(leaf_weight)
                                    for walker_idx, leaf_weight in leaf_walker_weights.items()}

        # the sum of the possible clones needs to be greater than or
        # equal to the balance
        if not sum(walker_n_possible_clones.values()) >= leaf_balance:
            raise RegionTreeError("there isn't enough clones possible to pay the balance")

        # go through the list of free walkers and see which ones have
        # any possible clones and make a list of them
        cloneable_walker_idxs = [walker_idx for walker_idx in leaf_walker_idxs
                                 if walker_n_possible_clones[walker_idx] > 0]

        cloneable_walker_weights = [leaf_walker_weights[walker_idx]
                                    for walker_idx in cloneable_walker_idxs]

        # if the sum of them is equal to the leaf balance then we
        # don't have to optimally distribute them and we just give them out
        if sum(walker_n_possible_clones.values()) == leaf_balance:

            for walker_idx in cloneable_walker_idxs:
                walkers_num_clones[walker_idx] = walker_n_possible_clones[walker_idx]

            # return this without doing all the prioritization
            return walkers_num_clones

        # otherwise we want to optimally distribute the clones to the
        # walkers such that we split the largest walkers first

        # to distribute the clones we iteratively choose the walker
        # with the highest weight after a single clone
        # weight/(n_clones +1) where n_clones is the current number of
        # clones assigned to it (plus itself), and then add another
        # clone to it as long as it is within the range of the number
        # of clones it can make

        # go until the balance is paid off
        clones_left = leaf_balance
        while clones_left > 0:

            # determine which walkers are still in the running for
            # receiving clones
            still_cloneable_walker_idxs = []
            still_cloneable_walker_weights = []
            for walker_idx, weight in zip(cloneable_walker_idxs, cloneable_walker_weights):

                # if the number of clones is less than its maximum
                # possible add it to the still applicable ones
                if (walkers_num_clones[walker_idx] < walker_n_possible_clones[walker_idx]):

                    still_cloneable_walker_idxs.append(walker_idx)
                    still_cloneable_walker_weights.append(weight)

            # if there is only one applicable walker left give it the
            # rest of the balance and return
            if len(still_cloneable_walker_idxs) == 1:
                walkers_num_clones[still_cloneable_walker_idxs[0]] += clones_left
                clones_left -= clones_left

                # end this loop iteration, skipping the decision part
                continue

            # if there are multiple walkers left we decide between them

            # calculate the weights of the walker's children given the
            # current number of clones, if num_clones is 0 then it is
            # just it's own weight
            child_weights = []
            for walker_idx, weight in zip(still_cloneable_walker_idxs, still_cloneable_walker_weights):

                # the weight of its children given the number of
                # clones already assigned to it
                child_weight = weight / (walkers_num_clones[walker_idx]+1)

                child_weights.append(child_weight)

            # get the walker_idx with the highest would-be child weight
            chosen_walker_idx = still_cloneable_walker_idxs[np.argsort(child_weights)[-1]]

            # add a clone to it
            walkers_num_clones[chosen_walker_idx] += 1

            # we are one step closer to satisfying the cloning
            # requirement
            clones_left -= 1

        return walkers_num_clones

    def _decide_settle_balance(self):
        """Given the balances of all the leaves figure out actually how to
        settle all the balances. Returns the merge_groups and
        walkers_num_clones

        Parameters
        ----------

        Returns
        -------

        """

        # initialize the main data structures for specifying how to
        # merge and clone a set of walkers. These will be modified for
        # clones and merges but in the initialized state they will
        # generate all NOTHING records.

        # the merge groups, a list of lists where the elements of the
        # outer list are individual "merge groups" that themselves
        # contain the elements of the walkers that will be squashed in
        # a merge and the index of the merge group in the outer list
        # is the index of the walker that will be kept
        # (i.e. KEEP_MERGE and have its state persist to the next
        # step). Indices appearing in any merge group can themselves
        # not have a merge group
        merge_groups = [[] for i in self.walker_weights]

        # the number of clones to make for each walker. Simply a list
        # of 0 or positive integers that specify how many EXTRA clones
        # will be made. E.g. if a cloned walker is to have 3 children
        # then the number of clones is 2. We consider clones copies
        # from the original walker which is given by the index in the
        # list. This number then gives the number of new slots needed
        # for a cloning event
        walkers_num_clones = [0 for i in self.walker_weights]

        # get all the leaf balances
        leaf_nodes = self.leaf_nodes()
        leaf_balances = [self.node[leaf]['balance'] for leaf in leaf_nodes]

        # get the negative and positive balanced leaves
        neg_leaves = [leaf_nodes[leaf_idx[0]] for leaf_idx in
                      np.argwhere(np.array(leaf_balances) < 0)]
        pos_leaves = [leaf_nodes[leaf_idx[0]] for leaf_idx in
                      np.argwhere(np.array(leaf_balances) > 0)]

        # we decide on how the walkers will be cloned and
        # merged. These steps are purely functional and do not modify
        # any attributes on the RegionTree. The merge_groups and
        # walkers_num_clones can be used to commit these changes
        # elsewhere if desired.

        # first do all leaves with negative balances, so that after we
        # have freed up slots we can fill them with clones since in
        # WEXplore we want to have an economy of slots and not create
        # them if we don't have to
        for leaf in neg_leaves:
            merge_groups = self._decide_merge_leaf(leaf, merge_groups)

        # then do all the leaves with positive balances to fill the
        # slots left from squashing walkers
        for leaf in pos_leaves:
            walkers_num_clones = self._decide_clone_leaf(leaf, merge_groups, walkers_num_clones)


        return merge_groups, walkers_num_clones

    def _check_clone_merge_specs(self, merge_groups, walkers_num_clones):
        """This will perform the computations to get the weights of the clones
        and merges but does not actually assign them to slots. This is
        mainly for checking that we have not violated any rules.

        Parameters
        ----------
        merge_groups :
            
        walkers_num_clones :
            

        Returns
        -------

        """

        # keep a dictionary of all the walkers that will be parents to
        # at least one child walker and make a list of the weights
        # what each of the children will be
        walker_children_weights = defaultdict(list)

        # walkers that will be keep merges
        keep_merge_walker_idxs = []

        # walkers that will parents of clones
        clone_parent_walker_idxs = []

        # walkers that do nothing and keep their state and weight
        nothing_walker_idxs = []

        # if that passes then we can check whether or not the weights make sense
        new_walker_weights = []

        # get all the squash idxs so we can pass over them
        all_squash_idxs = list(it.chain(*merge_groups))

        # go through each walker and see what the results of it would
        # be without assigning it to anywhere in particular
        for walker_idx, num_clones in enumerate(walkers_num_clones):

            # check that clones are not performed on KEEP_MERGE and SQUASH
            # walkers
            if num_clones > 0:
                if len(merge_groups[walker_idx]) > 0:
                    raise ResamplerError("trying to clone a KEEP_MERGE walker")

                squash_idxs = list(it.chain(merge_groups))
                if walker_idx in squash_idxs:
                    raise ResamplerError("trying to clone a SQUASH walker")

            squash_walker_idxs = merge_groups[walker_idx]

            # if it is a squashed walker ignore it
            if walker_idx in all_squash_idxs:
                pass

            # split the weight up evenly, the numbers in the list is
            # the extra number of walkers that should exist so that we
            # should add 1 to get the total number of child walkers
            # after the split
            elif num_clones > 0 and len(squash_walker_idxs) == 0:

                # add this to the list of clone parents
                clone_parent_walker_idxs.append(walker_idx)

                # get the weight each of the children will have
                clone_weights = self._walker_weights[walker_idx] / (num_clones + 1)

                # add them to the new_walker_weights and as the
                # children of this walkers weights
                for i in range(num_clones + 1):
                    # weights of all walkers
                    new_walker_weights.append(clone_weights)

                    # weights of the children
                    walker_children_weights[walker_idx].append(clone_weights)

            # if this is a merge group keep idx then we add the
            # weights of the merge group together
            elif len(squash_walker_idxs) > 0:

                # add this to the list of keep merge parents
                keep_merge_walker_idxs.append(walker_idx)

                # add up the weights of the squashed walkers
                squashed_weight = sum([self.walker_weights[i] for i in squash_walker_idxs])
                # add them to the weight for the keep walker
                walker_weight = self._walker_weights[walker_idx] + squashed_weight

                new_walker_weights.append(walker_weight)
                walker_children_weights[walker_idx].append(walker_weight)

            # this is then a nothing instruction so we just add its
            # weight to the list as is
            else:
                nothing_walker_idxs.append(walker_idx)
                new_walker_weights.append(self._walker_weights[walker_idx])
                walker_children_weights[walker_idx].append(self._walker_weights[walker_idx])

        # check that we have the same number of walkers as when we started
        if not len(new_walker_weights) == len(self._walker_weights):
            raise ResamplerError("There is not the same number of walkers as before the clone-merges")

        # then we check that the total weight before and after is the
        # same or close to the same
        if not np.isclose(sum(self._walker_weights), sum(new_walker_weights)):
            raise ResamplerError("There has been a change in total amount of weight")

        # check that none of the walkers are outside the range of
        # probabilities
        new_walker_weights = np.array(new_walker_weights)

        overweight_walker_idxs = np.where(new_walker_weights > self.pmax)[0]
        # check that all of the weights are less than or equal to the pmax
        if len(overweight_walker_idxs > 0):

            # list of parents that produce overweight children
            overweight_producer_idxs = []

            # figure out which parent created them, this will have
            # come from a merge so we just go through the parents that
            # are keep merges
            for keep_merge_walker_idx in keep_merge_walker_idxs:
                child_weight = walker_children_weights[keep_merge_walker_idx][0]
                if child_weight >= self.pmax:
                    overweight_producer_idxs.append(keep_merge_walker_idx)

            raise ResamplerError(
                "Merge specs produce overweight walkers for merge groups {}".format(
                    [str(i) for i in overweight_producer_idxs]))

        # check that all of the weights are less than or equal to the pmin
        underweight_walker_idxs = np.where(new_walker_weights < self.pmin)[0]
        if len(underweight_walker_idxs > 0):

            # list of clone parents that will produce underweight
            # walkers
            underweight_producer_idxs = []

            # figure out which parents create underweight walkers,
            # only clones will do this so we just look through them
            for clone_parent_walker_idx in clone_parent_walker_idxs:
                # all children will be the same weight so we just get
                # one of the weights
                child_weight = walker_children_weights[clone_parent_walker_idx][0]
                if child_weight <= self.pmin:
                    underweight_producer_idxs.append(clone_parent_walker_idx)

            raise ResamplerError(
                "Clone specs produce underweight walkers for clone walkers {}".format(
                    [str(i) for i in underweight_producer_idxs]))

    def balance_tree(self, delta_walkers=0):
        """Do balancing between the branches of the tree. the `delta_walkers`
        kwarg can be used to increase or decrease the total number of
        walkers, but defaults to zero which will cause no net change
        in the number of walkers.

        Parameters
        ----------
        delta_walkers :
             (Default value = 0)

        Returns
        -------

        """

        # set the delta walkers to the balance of the root node
        self.node[self.ROOT_NODE]['balance'] = delta_walkers

        # do a breadth first traversal and balance at each level
        for parent, children in nx.bfs_successors(self, self.ROOT_NODE):

            # pass on the balance of this parent to the children from the
            # parents, distribute walkers between
            parental_balance = self.node[parent]['balance']

            # this will both propagate the balance set for the root
            # walker down the tree and balance between the children
            self._propagate_and_balance_shares(parental_balance, children)

        # check that the sum of the balances of the leaf nodes
        # balances to delta_walkers
        leaf_balances = [self.node[leaf]['balance'] for leaf in self.leaf_nodes()]
        if sum(leaf_balances) != delta_walkers:

            raise RegionTreeError(
                "The balances of the leaf nodes ({}) do not balance to delta_walkers ({})".format(
                    leaf_balances, delta_walkers))

        # decide on how to settle all the balances between leaves
        merge_groups, walkers_num_clones = self._decide_settle_balance()

        # count up the number of clones and merges in the merge_groups
        # and the walkers_num_clones
        num_clones = sum(walkers_num_clones)
        num_squashed = sum([len(merge_group) for merge_group in merge_groups])

        # check that the number of clones and number of squashed
        # walkers balance to the delta_walkers amount
        if num_clones - num_squashed != delta_walkers:

            raise RegionTreeError("The number of new clones ({}) is not balanced by the number of"
                                  "squashed walkers ({}) to the delta_walkers specified ({})".format(
                                      num_clones, num_squashed, delta_walkers))

        # DEBUG
        # check the merge groups and walkers_num_clones to make sure
        # they are valid
        try:
            self._check_clone_merge_specs(merge_groups, walkers_num_clones)
        except ResamplerError as resampler_err:
            print(resampler_err)
            import ipdb; ipdb.set_trace()

        return merge_groups, walkers_num_clones

class WExploreResampler(CloneMergeResampler):
    """Resampler implementing the WExplore algorithm.

    See the paper for a full description of the algorithm, but
    briefly:

    WExplore defines a hierarchical Voronoi tesselation on a subspace
    of the full walker state. Regions in a Voronoi cell are defined by
    a point in this subspace called an 'image', and are scoped by
    their enclosing region in the hierarchy.

    The hierarchy is defined initially by:

    - number of levels of the hierarchy, AKA depth
    - number of regions allowed at a level of the hierarchy
    - a cutoff 'distance' at each level of the hierarchy

    All regions have a unique specification (called a leaf_id) which
    is a k-tuple of region indices, where k is the depth of the
    hierarchy.

    At first the hierarchy only has a single region at all levels
    which is given by the image of the 'init_state' constructor
    argument.

    For a hierarchy of depth 4 the leaf_id for this first region is
    (0, 0, 0, 0) which indicates that at level 0 (the highest level)
    we are selecting the first region (region 0), and at level 1 we
    select the first region at that level (region 0 again), and so on.

    For this region all images at each level are identical.

    During resampling walkers first are binned into the region they
    fall. This is achieved using a breadth first search where the
    distance (according to the distance metric) between the walker
    state and all region images are computed. The image that is
    closest to the walker is selected and in the next iteration of
    comparisons we restrict the search to only regions within this
    super-region.

    For example for a region tree with the following leaf ids: (0,0,0)
    and (0,1,0) we skip distance computations at level 0, since there
    is only one region. At level 1, we compute the distance to images
    0 and 1 and choose the one that is closest. Since each of these
    regions only has 1 sub-region we do not need to recalculate the
    distance and can assign the walker.

    Indeed every super-region will have exactly one sub-region that
    has the same image as it, and these distance calculations are
    never repeated for performance.

    This resampler adds the additional 'region_assignment' field to
    the resampling records which indicates the region a walker was
    assigned to during resampling.


    While walkers can always be assigned uniquely to a region the
    specification of the cutoff distances at each level indicate when
    sufficient novelty in a walker motivates creation of a new region.

    The distances of the walker to each of the closest region images
    is saved for each level, e.g. (0.1, 0.3, 0.5) for the example
    above. This is compared to the cutoff distance specification,
    e.g. (0.2, 0.2, 0.2). The highest level at which a distance
    exceeds the cutoff will trigger region creation.

    Following the example above the walker above exceeded the cutoff
    distance at level 1 and level 2 of the hierarchy, however level 1
    takes precedence over the lower level.

    This event of region creation can be thought of as a branching
    event of the tree, where the branching level is the level at which
    the branch occurred.

    All branches of a region tree must extend the full depth and so
    the specification of the new branch can be given by the leaf_id of
    the region created, which would be (0, 2, 0) for this example.

    The new image of this region is the image of the walker that
    triggered the branching.

    Note that the 'boundaries' of the Voronoi cells are subject to
    change during branching.

    The only limitation to this process is the allowed number of
    sub-regions for a super-region which is given for each level. For
    example, (5, 10, 10) indicates that regions at the top-most level
    can have 5 sub-regions, and in turn those sub-regions can have 10
    sub-regions, and so on.

    The resampler records give updates on the definitions of new
    regions and includes:

    - branching_level : the level of the tree the branching occured
          at, which relates it to the number of allowed regions and the
          cutoff distance new regions are made at.

    - distance : the distance of the walker to the nearest region
          image that triggered the creation of a new region.

    - new_leaf_id : the leaf id of the new region, which is a tuple of
          the index of each region index at each branching level.

    - image : a datatype that stores the actual value of the newly
          created region.



    That covers how regions are initialized, adaptively created, and
    recorded, but doesn't explain how these regions are used inform
    the actual resampling process.

    Essentially, the hierarchical structure allows for balanced
    resource trading between regions. The resource in the case of
    weighted ensemble is the allocation of a walker which will be
    scheduled for sampling in the next cycle. The more walkers a
    region has the more sampling of the region will occur.

    After the sampling step walkers move around in regions and between
    regions. Some regions will end up collecting more walkers than
    other regions during this. The goal is to redistribute those
    walkers to other regions so that each level of the hierarchy is as
    balanced as possible.

    So after sampling the number of walkers is added up for each
    region. At the top of the hierarchy a collection of trades between
    the super-regions is negotiated based on which regions are able to
    give up walkers and those that need them such that each region has
    as close to the same number of walkers as possible.

    The 'payment' of a tax (or donation) is performed by merging
    walkers and the reception of the donation (or welfare
    dispensation) is achieved by cloning walkers.

    Because there are constraints on how many walkers can be merged
    and cloned due to minimum and maximum walker probabilities and
    total number of walkers, sometimes a region may have an excess of
    walkers but none of them are taxable (donatable) because any merge
    would create a walker above the maximum probability. Conversely, a
    region may not be able to receive walkers because any clone of a
    walker would make walkers with weight lower than the minimum.

    However, super-regions cannot actually 'pay' for these trades
    themselves and simply make a request to their sub-regions to
    provide the requested walkers (or to find room for accepting
    them). The trade negotiation process then repeats within each
    sub-region. When the request for debits and credits finally
    reaches leaf node regions the process stops and the actual
    identities of the walkers that will be cloned or merged are
    determined.

    Then the cloning and merging of walkers is performed.

    After this process if we were to recount the number of walkers in
    all regions then they should be pretty well balanced at each
    level.

    Its worth noting that some regions will have no walkers because
    once a region contains no walkers (all walkers leave the region
    during sampling) it can no longer receive any walkers at all
    because it cannot clone. Of course walkers may re-enter the region
    and repopulate it but until that happens these regions are
    excluded from negotiations.

    Only the net clones and merges are recorded in the records.

    """

    # TODO refactor step_idx and walker_idx to superclass

    # fields for resampling data
    RESAMPLING_FIELDS = CloneMergeResampler.RESAMPLING_FIELDS + ('region_assignment',)
    RESAMPLING_SHAPES = CloneMergeResampler.RESAMPLING_SHAPES + (Ellipsis,)
    RESAMPLING_DTYPES = CloneMergeResampler.RESAMPLING_DTYPES + (np.int,)

    # fields that can be used for a table like representation
    RESAMPLING_RECORD_FIELDS = CloneMergeResampler.RESAMPLING_RECORD_FIELDS + ('region_assignment',)

    # datatype for the state change records of the resampler, here
    # that is the defnition of a new branch of the region tree, the
    # value is the level of the tree that is branched. Most of the
    # useful information will be in the auxiliary data, like the
    # image, distance the walker was away from the image at that
    # level, and the id of the leaf node
    RESAMPLER_FIELDS = CloneMergeResampler.RESAMPLER_FIELDS + \
                       ('branching_level', 'distance', 'new_leaf_id', 'image')
    RESAMPLER_SHAPES = CloneMergeResampler.RESAMPLER_SHAPES + \
                       ((1,), (1,), Ellipsis, Ellipsis)
    RESAMPLER_DTYPES = CloneMergeResampler.RESAMPLER_DTYPES + \
                       (np.int, np.float, np.int, None)

    # fields that can be used for a table like representation
    RESAMPLER_RECORD_FIELDS = CloneMergeResampler.RESAMPLER_RECORD_FIELDS + \
                              ('branching_level', 'distance', 'new_leaf_id')



    def __init__(self, seed=None,
                 distance=None,
                 max_region_sizes=None,
                 init_state=None,
                 pmin=1e-12,
                 pmax=0.1,
                 max_n_regions=(10, 10, 10, 10),
                 **kwargs
                ):
        """Constructor for the WExploreResampler.

        Parameters
        ----------

        seed : None or int
            The random seed. If None the system (random) one will be used.

        distance : object implementing Distance
            The distance metric to compare walkers to region images with.

        init_state : WalkerState object
            The state that seeds the first region in the region hierarchy.

        max_n_regions : tuple of int
            The number of allowed sub-regions for a region at each
            level of the region hierarchy.

        max_region_sizes : tuple of float
            The cutoff distances that trigger the creation of new
            regions at each level of the hierarchy. Numbers dependent
            on the units of your distance metric.
            For example: (1, 0.5, 0.35, 0.25).

        """

        # we call the common methods in the CloneMergeResampler
        # superclass. We set the min and max number of walkers to be
        # constant
        super().__init__(pmin=pmin, pmax=pmax,
                         min_num_walkers=Ellipsis,
                         max_num_walkers=Ellipsis,
                         **kwargs)

        assert distance is not None, "Distance object must be given."
        assert max_region_sizes is not None, "Max region sizes must be given"

        self.distance = distance

        assert init_state is not None, "An initial state must be given."

        # the region tree which keeps track of the regions and can be
        # balanced for cloning and merging between them, is
        # initialized the first time resample is called because it
        # needs an initial walker
        self._region_tree = None

        # parameters
        self.seed = seed
        if self.seed is not None:
            rand.seed(self.seed)

        self.max_n_regions = max_n_regions
        self.n_levels = len(max_n_regions)
        self.max_region_sizes = max_region_sizes


        # we do not know the shape and dtype of the images until
        # runtime so we determine them here
        image = self.distance.image(init_state)
        self.image_shape = image.shape
        self.image_dtype = image.dtype


        # initialize the region tree with the first state
        self._region_tree = RegionTree(init_state,
                                       max_n_regions=self.max_n_regions,
                                       max_region_sizes=self.max_region_sizes,
                                       distance=self.distance,
                                       pmin=self.pmin,
                                       pmax=self.pmax)

    def resampler_field_shapes(self):

        # index of the image idx
        image_idx = self.resampler_field_names().index('image')

        # shapes adding the image shape
        shapes = list(super().resampler_field_shapes())
        shapes[image_idx] = self.image_shape

        return tuple(shapes)

    def resampler_field_dtypes(self):

        # index of the image idx
        image_idx = self.resampler_field_names().index('image')

        # dtypes adding the image dtype
        dtypes = list(super().resampler_field_dtypes())
        dtypes[image_idx] = self.image_dtype

        return tuple(dtypes)

    @property
    def region_tree(self):
        """The RegionTree instance used to manage the region hierachy.

        This is really only used internally to this class.

        """
        return self._region_tree


    def assign(self, walkers):
        """Assign walkers to regions in the tree, with region creation.

        Parameters
        ----------
        walkers : list of Walker objects

        Returns
        -------

        assignments : list of tuple of int
            The leaf_id for each walker that it was assigned to.

        resampler_data : list of dict of str: value
            The list of resampler records recording each branching event.

        """
        ## Assign the walkers based on the current defined Voronoi
        ## images which assign them to bins/leaf-nodes, possibly
        ## creating new regions, do this by calling the method to
        ## "place_walkers"  on the tree which changes the tree's state
        new_branches = self.region_tree.place_walkers(walkers)

        # data records about changes to the resampler, here is just
        # the new branches data
        resampler_data = new_branches

        # the assignments
        assignments = np.array(self.region_tree.walker_assignments)

        # return the assignments and the resampler records of changed
        # resampler state, which is addition of new regions
        return assignments, resampler_data

    def decide(self, delta_walkers=0):
        """Make decisions for resampling for a single step.

        Parameters
        ----------
        delta_walkers : int
            The net change in the number of walkers to make.
             (Default value = 0)

        Returns
        -------

        resampling_data : list of dict of str: value
            The resampling records resulting from the decisions.

        """

        ## Given the assignments (which are on the tree nodes) decide
        ## on which to merge and clone

        # do this by "balancing" the tree. delta_walkers can be
        # specified to increase or decrease the total number of
        # walkers
        merge_groups, walkers_num_clones = \
                        self.region_tree.balance_tree(delta_walkers=delta_walkers)

        logging.info("merge_groups\n{}".format(merge_groups))
        logging.info("Walker number of clones\n{}".format(walkers_num_clones))
        logging.info("Walker assignments\n{}".format(self.region_tree.walker_assignments))
        logging.info("Walker weights\n{}".format(self.region_tree.walker_weights))

        # check to make sure we have selected appropriate walkers to clone
        logging.info("images_assignments\n{}".format(self.region_tree.regions))

        # take the specs for cloning and merging and generate the
        # actual resampling actions (instructions) for each walker,
        # this does not change the state of the resampler or region
        # tree
        resampling_actions = self.assign_clones(merge_groups, walkers_num_clones)

        if self.is_debug_on:
            # check that the actions were performed correctly
            try:
                self._check_resampling_data(resampling_actions)
            except ResamplerError as resampler_err:
                print(resampler_err)
                import ipdb; ipdb.set_trace()


        return resampling_actions


    @staticmethod
    def _check_resampling_data(resampling_data):
        """

        Parameters
        ----------
        resampling_data :
            

        Returns
        -------

        """

        # in WExplore we don't increase or decrease the number of
        # walkers and thus all slots must be filled so we go through
        # each decision that targets slots in the next stop and
        # collect all of those

        n_slots = len(resampling_data)

        taken_slot_idxs = []
        squash_slot_idxs = []
        keep_merge_slot_idxs = []
        for rec_d in resampling_data:
            if rec_d['decision_id'] in (1, 2, 4):
                taken_slot_idxs.extend(rec_d['target_idxs'])

            if rec_d['decision_id'] == 3:
                squash_slot_idxs.extend(rec_d['target_idxs'])

            if rec_d['decision_id'] == 4:
                keep_merge_slot_idxs.extend(rec_d['target_idxs'])


        # see if there are any repeated targets
        if len(set(taken_slot_idxs)) < len(taken_slot_idxs):
            raise ResamplerError("repeated slots to be used")

        # check that the number of targets is exactly the number of slots available
        if len(taken_slot_idxs) < n_slots:
            raise ResamplerError("Number of slots used is less than the number of slots")
        elif len(taken_slot_idxs) > n_slots:
            raise ResamplerError("Number of slots used is greater than the number of slots")

        # check that all squashes are going to a merge slot
        if not all([False if squash_slot_idx not in keep_merge_slot_idxs else True
         for squash_slot_idx in set(squash_slot_idxs)]):
            raise ResamplerError("Not all squashes are assigned to keep_merge slots")

    def _resample_init(self, walkers=None):
        """

        Parameters
        ----------
        walkers :
            

        Returns
        -------

        """

        super()._resample_init(walkers=walkers)

        # then get the walker nums using our methods to get it for
        # this resampling and just give that to the region tree
        self.region_tree.max_num_walkers = self.max_num_walkers()
        self.region_tree.min_num_walkers = self.min_num_walkers()

        if self.is_debug_on:

            # cache a copy of the region_tree in its state before putting
            # these walkers through it so we can replay steps if necessary
            self._cached_region_tree = deepcopy(self._region_tree)

            # and keep the walkers too
            self._input_walkers = deepcopy(walkers)

    def _resample_cleanup(self, resampling_data=None,
                          resampler_data=None,
                          resampled_walkers=None):
        """

        Parameters
        ----------
        resampling_data :
            
        resampler_data :
            
        resampled_walkers :
            

        Returns
        -------

        """

        if self.is_debug_on:

            # check that the weights of the resampled walkers are not
            # beyond the bounds of what they are supposed to be
            try:
                self._check_resampled_walkers(resampled_walkers)
            except ResamplerError as resampler_err:
                print(resampler_err)
                import ipdb; ipdb.set_trace()

                # keep the tree we just used
                curr_region_tree = self._region_tree

                # replace the region tree with the cached region_tree
                self._region_tree = self._cached_region_tree

                # then run resample again with the original walkers
                self.resample(self._input_walkers)

                # then reset the old region tree
                self._region_tree = curr_region_tree

                # and clean out the debug variables
                del self._cached_region_tree
                del self._input_walkers

        # clear the tree of walker information for the next resampling
        self.region_tree.clear_walkers()

        # just use the superclass method
        super()._resample_cleanup()

        # then get the walker nums using our methods to get it for
        # this resampling and just give that to the region tree
        self.region_tree.max_num_walkers = False
        self.region_tree.min_num_walkers = False

    @log_call(include_args=[],
              include_result=False)
    def resample(self, walkers):

        # do some initialiation routines and debugging preparations if
        # necessary
        self._resample_init(walkers=walkers)

        ## assign/score the walkers, also getting changes in the
        ## resampler state
        assignments, resampler_data = self.assign(walkers)


        # make the decisions for the the walkers for only a single
        # step
        resampling_data = self.decide(delta_walkers=0)

        # add the walker idxs
        for walker_idx, walker_record in enumerate(resampling_data):
            walker_record['walker_idx'] = walker_idx

        # perform the cloning and merging, the action function expects
        # records a lists of lists for steps and walkers
        resampled_walkers = self.DECISION.action(walkers, [resampling_data])

        # normally decide is only for a single step and so does not
        # include the step_idx, so we add this to the records
        for walker_idx, walker_record in enumerate(resampling_data):

            walker_record['step_idx'] = np.array([0])

        # convert the target idxs and decision_id to feature vector arrays
        for record in resampling_data:
            record['target_idxs'] = np.array(record['target_idxs'])
            record['decision_id'] = np.array([record['decision_id']])
            record['walker_idx'] = np.array([record['walker_idx']])


        # then add the assignments and distance to image for each walker
        for walker_idx, assignment in enumerate(assignments):
            resampling_data[walker_idx]['region_assignment'] = assignment



        self._resample_cleanup(resampling_data=resampling_data,
                               resampler_data=resampler_data,
                               resampled_walkers=resampled_walkers)

        return resampled_walkers, resampling_data, resampler_data


