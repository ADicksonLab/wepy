import math
import random as rand
import itertools as it
from collections import namedtuple
from copy import copy
from copy import deepcopy

import numpy as np
import networkx as nx

from wepy.resampling.resamplers.resampler  import Resampler
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision

class RegionTreeError(Exception):
    pass

class RegionTree(nx.DiGraph):

    ROOT_NODE = ()

    def __init__(self, init_state,
                 max_n_regions=None,
                 max_region_sizes=None,
                 distance=None,
                 pmin=None, pmax=None):

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

        self._walker_weights = []
        self._walker_assignments = []

        image_idx = 0
        # get the image using the distance object
        image = self.distance.image(init_state)
        self._images = [image]

        parent_id = self.ROOT_NODE
        self.add_node(parent_id, image_idx=0,
                      n_walkers=0,
                      n_mergeable=0,
                      n_cloneable=0,
                      balance=0,
                      walker_idxs=[])

        # make the first branch
        for level in range(len(max_n_regions)):
            child_id = parent_id + (0,)
            self.add_node(child_id, image_idx=image_idx,
                          n_walkers=0,
                          n_mergeable=0,
                          n_cloneable=0,
                          balance=0,
                          walker_idxs=[])
            self.add_edge(parent_id, child_id)
            parent_id = child_id

        # add the region for this branch to the regions list
        self._regions = [tuple([0 for i in range(self._n_levels)])]

    @property
    def distance(self):
        return self._distance

    @property
    def images(self):
        return self._images

    @property
    def max_n_regions(self):
        return self._max_n_regions

    @property
    def n_levels(self):
        return self._n_levels

    @property
    def max_region_sizes(self):
        return self._max_region_sizes

    @property
    def pmin(self):
        return self._pmin

    @property
    def pmax(self):
        return self._pmax

    @property
    def walker_assignments(self):
        return self._walker_assignments

    @property
    def walker_weights(self):
        return self._walker_weights

    @property
    def regions(self):
        return self._regions

    def add_child(self, parent_id, image_idx):
        # make a new child id which will be the next index of the
        # child with the parent id
        child_id = parent_id + (len(self.children(parent_id)), )

        # create the node with the image_idx
        self.add_node(child_id,
                      image_idx=image_idx,
                      n_walkers=0,
                      n_mergeable=0,
                      n_cloneable=0,
                      balance=0,
                      walker_idxs=[])

        # make the edge to the child
        self.add_edge(parent_id, child_id)

        return child_id

    def children(self, parent_id):
        children_ids = list(self.adj[parent_id].keys())
        # sort them
        children_ids.sort()
        return children_ids

    def level_nodes(self, level):
        """Get the nodes/regions at the specified level."""

        if level > self.n_levels:
            raise ValueError("level is greater than the number of levels for this tree")

        return [node_id for node_id in self.nodes
                if len(node_id) == level]

    def leaf_nodes(self):
        return self.level_nodes(self.n_levels)

    def branch_tree(self, parent_id, image):
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

    def assign(self, state):

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
                    dist = self.distance.image_distance(state_image, image)

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
            self.node[node_id]['n_mergeable'] = 0
            self.node[node_id]['n_cloneable'] = 0
            self.node[node_id]['balance'] = 0
            self.node[node_id]['walker_idxs'] = []

    def place_walkers(self, walkers):

        # clear all the walkers and reset node attributes to defaults
        self.clear_walkers()

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
                    image = self.distance.image(walker.state)
                    parent_id = assignment[:level]
                    assignment = self.branch_tree(parent_id, image)
                    # we have made a new branch so we don't need to
                    # continue this loop
                    break

            # save the walker assignment
            self._walker_assignments.append(assignment)
            self._walker_weights.append(walker.weight)

            # test to see if this walker has a weight greater than the
            # minimum, and raise a flag to tally it in the node
            # attributes if it does (in the next loop)
            above_pmin = False
            if walker.weight > self.pmin:
                above_pmin = True

            # go back through the nodes in this walker's branch
            # increase the n_walkers for each node, and save the
            # walkers (index in self.walker_assignments) it has, and
            # save increase the number above pmin if valid
            for level in range(len(assignment) + 1):
                node_id = assignment[:level]

                self.node[node_id]['n_walkers'] += 1
                self.node[node_id]['walker_idxs'].append(walker_idx)
                if above_pmin:
                    self.node[node_id]['n_cloneable'] += 1

        # after placing all the walkers we calculate the number of
        # reducible walkers for each node

        # for each leaf node calculate the number of reducible walkers
        # (i.e. the largest possible number of merges that could occur
        # taking in to account the pmax (max weight) constraint)
        for node_id in self.leaf_nodes():
            if self.node[node_id]['n_walkers'] > 1:

                # the weights of the walkers in this node
                weights = np.sort([self.walker_weights[i]
                                   for i in self.node[node_id]['walker_idxs']])

                # figure out the most possible mergeable walkers
                # assuming they cannot ever be larger than pmax
                sum_weights = 0
                for i in range(len(weights)):
                    sum_weights += weights[i]
                    # if we still haven't gone past pmax set the n_red
                    # to the current index
                    if sum_weights < self.pmax and i+1 < len(weights):
                        self.node[node_id]['n_mergeable'] = i + 1

                # increase the reducible walkers for the higher nodes
                # in this leaf's branch
                for level in reversed(range(self.n_levels)):
                    branch_node_id = node_id[:level]
                    self.node[branch_node_id]['n_mergeable'] += self.node[node_id]['n_mergeable']

    def minmax_beneficiaries(self, children):

        #min_n_walkers = None
        min_n_shares = None
        min_child_idx = None

        #max_n_walkers = None
        max_n_shares = None
        max_child_idx = None

        # test each walker sequentially
        for i, child in enumerate(children):

            n_mergeable = self.node[child]['n_mergeable']
            n_cloneable = self.node[child]['n_cloneable']

            # we need to take into account the balance inherited
            # from the parent when calculating the total number of
            # walkers that can be given to other node/regions
            #total_n_walkers = n_walkers + self.node[child]['balance']
            #total_n_mergeable = n_mergeable + self.node[child]['balance']

            # the number of shares the walker currently has which is
            # the sum of the number of mergeable (squashable) plus the
            # balance
            n_shares =  n_mergeable + self.node[child]['balance']

            # the maximum number of walkers that will exist after
            # cloning, if the number of reducible walkers is 1 or more
            if ((max_child_idx is None) or (n_shares >  max_n_shares)) and \
               (n_shares >= 1):

                max_n_shares = n_shares
                max_child_idx = i

            # the minimum number of walkers that will exist after
            # cloning, if the number of number of walkers above the
            # minimum weight is 1 or more
            if ((min_child_idx is None) or (n_shares < min_n_shares)) and \
               (n_cloneable >= 1):

                min_n_shares = n_shares
                min_child_idx = i

        return min_child_idx, max_child_idx

    def calc_share_donation(self, donor, recipient):

        total_donor_n_walkers = self.node[donor]['n_mergeable'] + \
                                self.node[donor]['balance']
        total_recipient_n_walkers = self.node[recipient]['n_mergeable'] + \
                                    self.node[recipient]['balance']

        # the sibling with the greater number of shares
        # (from both previous resamplings and inherited
        # from the parent) will give shares to the sibling
        # with the least. The number it will share is the
        # number that will make them most similar rounding
        # down (i.e. midpoint)


        n_shares = math.floor((total_donor_n_walkers - total_recipient_n_walkers)/2)

        return n_shares

    def resample_regions(self, parental_balance, children):
        # there are more than one child so we accredit balances
        # between them
        if len(children) > 1:

            # if the node had a balance assigned to previously,
            # apply this to the number of walkers it has
            #self.node[parent]['n_mergeable'] += parental_balance
            #self.node[parent]['n_walkers'] += parental_balance

            # if the parent has a non-zero balance we either
            # increase (clone) or decrease (merge) the balance

            # these poor children are inheriting a debt and must
            # decrease the total number of their credits :(
            if parental_balance < 0:

                # find children with mergeable walkers and account
                # for them in their balance
                for child in children:

                    # if this child has any mergeable walkers
                    if self.node[child]['n_mergeable'] >= 1:
                        # we use those for paying the parent's debt
                        diff = min(self.node[child]['n_mergeable'],
                                       abs(parental_balance))
                        parental_balance += diff
                        self.node[child]['balance'] -= diff

                if parental_balance < 0:
                    raise ValueError("Children cannot pay their parent's debt")


            # these lucky children are inheriting a positive number of
            # credits!! :)
            elif parental_balance > 0:

                for child in children:
                    if self.node[child]['n_cloneable']:
                        self.node[child]['balance'] += parental_balance
                        parental_balance = 0

                if parental_balance > 0:
                    raise ValueError("Children cannot perform any clones.")

            ## Balance the walkers between the children. To do
            ## this we iteratively identify nodes/regions to trade
            ## walkers between. We choose the two regions with the
            ## highest and the lowest number of walkers (that have
            ## at least one reducible walker and have at least one
            ## walker that is cloneable (above pmin)
            ## respectively). We then tally these trades and
            ## repeat until no further trades can be made.

            # get the children with the max and min numbers of walkers
            min_child_idx, max_child_idx = self.minmax_beneficiaries(children)

            # get the actual node_ids for the children from their
            # index among children
            if max_child_idx is not None:
                max_child = children[max_child_idx]
            else:
                max_child = None
            if min_child_idx is not None:
                min_child = children[min_child_idx]
            else:
                min_child = None

            # The recipient needs to have at least one clonable
            # walker and the donor needs to have at least one
            # mergeable walker. If they do not (None from the
            # minmax function) then we cannot assign a number of
            # shares for them to give each other
            if None in (max_child, min_child):
                n_shares = 0
            else:
                # calculate the number of share donations to take from
                # the max child and give to the min child
                n_shares = self.calc_share_donation(max_child,
                                                    min_child)

                # account for these in the nodes
                self.node[max_child]['balance'] -= n_shares
                self.node[min_child]['balance'] += n_shares

            # iteratively repeat this process until the number of
            # shares being donated is 1 or less which means the
            # distribution is as uniform as possible
            while(n_shares >= 1):
                # repeat above steps
                min_child_idx, max_child_idx = self.minmax_beneficiaries(children)

                # get the actual node_ids for the children from their
                # index among children
                if max_child_idx is not None:
                    max_child = children[max_child_idx]
                else:
                    max_child = None
                if min_child_idx is not None:
                    min_child = children[min_child_idx]
                else:
                    min_child = None


                n_shares = self.calc_share_donation(max_child,
                                                    min_child)
                self.node[max_child]['balance'] -= n_shares
                self.node[min_child]['balance'] += n_shares

        # only one child so it just inherits balance
        elif len(children) == 1:

            # increase the balance of the only child
            self.node[children[0]]['balance'] = parental_balance

    def merge_leaf(self, leaf, merge_groups, walkers_num_clones):

        leaf_balance = self.node[leaf]['balance']
        # we need to check that we aren't squashing or merging an
        # already cloned walker
        mergeable_walkers = [walker_idx for walker_idx, num_clones
                             in enumerate(walkers_num_clones)
                             if (num_clones == 0) and
                                (walker_idx in self.node[leaf]['walker_idxs'])]

        mergeable_weights = [self.walker_weights[walker_idx] for walker_idx in mergeable_walkers]

        # the number of walkers we need to choose in order to be
        # able to do the required amount of merges
        num_merge_walkers = abs(leaf_balance) + 1
        # select the lowest weight walkers to use for merging
        selected_walkers = np.argsort(mergeable_weights)[:num_merge_walkers]

        # get the walker_idx of the selected walkers
        walker_idxs = [mergeable_idx for i, mergeable_idx in enumerate(mergeable_walkers)
                          if i in selected_walkers]

        chosen_weights = [weight for i, weight in enumerate(self.walker_weights)
                          if i in walker_idxs]

        # choose the one to keep the state of (e.g. KEEP_MERGE
        # in the Decision) based on their weights
        #keep_idx = rand.choices(walker_idxs, k=1, weights=chosen_weights)
        # normalize weights to a distribution
        chosen_pdist = np.array(chosen_weights) / sum(chosen_weights)
        keep_idx = np.random.choice(walker_idxs, 1, p=chosen_pdist)[0]

        # pop the keep idx from the walkers so we can use them as the squash idxs
        walker_idxs.pop(walker_idxs.index(keep_idx))
        # the rest are squash_idxs
        squash_idxs = walker_idxs

        # account for the weight from the squashed walker to
        # the keep walker
        squashed_weight = sum([self.walker_weights[i] for i in squash_idxs])
        self._walker_weights[keep_idx] += squashed_weight
        for squash_idx in squash_idxs:
            self._walker_weights[squash_idx] = 0.0

        # update the merge group
        merge_groups[keep_idx].extend(squash_idxs)

        return merge_groups, walkers_num_clones

    def clone_leaf(self, leaf, merge_groups, walkers_num_clones):

        # if this leaf node was assigned a debt we need to merge
        # walkers
        leaf_balance = self.node[leaf]['balance']

        # all of the squashed walkers
        squashed_walkers = list(it.chain(merge_groups))
        keep_walkers = [i for i, merge_group in enumerate(merge_groups)
                        if len(merge_group) > 0]
        merged_walkers = squashed_walkers + keep_walkers

        # get the weights of just the walkers in this leaf
        leaf_weights = [self.walker_weights[walker_idx]
                        for walker_idx in self.node[leaf]['walker_idxs']]

        # get the idxs of the cloneable walkers that are above the
        # pmin and are not being squashed that are in this leaf
        cloneable_walker_idxs = [walker_idx for walker_idx, weight
                                 in zip(self.node[leaf]['walker_idxs'], leaf_weights)
                                 if (weight >= self.pmin) and (walker_idx not in merged_walkers)]

        cloneable_walker_weights = [self.walker_weights[walker_idx]
                                    for walker_idx in cloneable_walker_idxs]

        assert len(cloneable_walker_idxs) >= 1, "there must be at least one walker to clone"


        # to distribute the clones we iteratively choose the walker
        # with the highest weight after amplification weight/(n_clones
        # +1) where n_clones is the current number of clones assigned
        # to it (plus itself)
        clones_left = leaf_balance
        while clones_left > 0:

            # calculate the weights of the walkers children given the
            # current number of clones
            child_weights = [weight/(walkers_num_clones[walker_idx]+1)
                             for walker_idx, weight in
                             zip(cloneable_walker_idxs, cloneable_walker_weights)]

            # get the walker_idx with the highest child weight
            chosen_walker_idx = cloneable_walker_idxs[np.argsort(child_weights)[-1]]
            walkers_num_clones[chosen_walker_idx] += 1

            clones_left -= 1

        return merge_groups, walkers_num_clones

    def settle_balance(self, leaf, merge_groups, walkers_num_clones):

        # if this leaf node was assigned a debt we need to merge
        # walkers
        leaf_balance = self.node[leaf]['balance']
        if leaf_balance < 0:

            merge_groups, walkers_num_clones = self.merge_leaf(leaf, merge_groups, walkers_num_clones)

        # if this leaf node was assigned a credit then it can spend
        # them on cloning walkers
        elif leaf_balance > 0:

            merge_groups, walkers_num_clones = self.clone_leaf(leaf, merge_groups, walkers_num_clones)

        return merge_groups, walkers_num_clones


    def balance_tree(self, delta_walkers=0):
        """Do balancing between the branches of the tree. the `delta_walkers`
        kwarg can be used to increase or decrease the total number of
        walkers, but defaults to zero which will cause no net change
        in the number of walkers.

        """

        # set the delta walkers to the balance of the root node
        self.node[self.ROOT_NODE]['balance'] = delta_walkers

        # do a breadth first traversal and balance at each level
        for parent, children in nx.bfs_successors(self, self.ROOT_NODE):

            # pass on the balances to the children from the
            # parents, distribute walkers between
            self.resample_regions(self.node[parent]['balance'], children)

        # check the balance of all the leaves and make sure they add
        # up to 0
        balance = sum([self.node[leaf]['balance'] for leaf in self.leaf_nodes()])
        if balance != delta_walkers:
            raise RegionTreeError("Leaf node balances do sum to delta_walkers")

        # for each leaf-group decide how to actually settle (pay) the
        # balances through cloning and merging
        merge_groups = [[] for i in self.walker_weights]
        walkers_num_clones = [0 for i in self.walker_weights]

        # The buck ends here! Iterate over the leaves and settle the
        # balances for each child according to how they were
        # distributed in balancing
        for leaf in self.leaf_nodes():

            # clone and merge groups for this leaf
            leaf_node_balance = self.node[leaf]['balance']
            if leaf_node_balance != 0:
                merge_groups, walkers_num_clones = \
                                                self.settle_balance(leaf, merge_groups, walkers_num_clones)

        # count up the number of clones and merges in the merge_groups
        # and the walkers_num_clones
        num_clones = sum(walkers_num_clones)
        num_squashed = sum([len(merge_group) for merge_group in merge_groups])

        if num_clones != num_squashed:
            raise RegionTreeError("The number of squashed walkers in the merge group"
                                  "is not the same as the number of clones planned.")


        return merge_groups, walkers_num_clones

class WExplore1Resampler(Resampler):

    DECISION = MultiCloneMergeDecision

    def __init__(self, seed=None, pmin=1e-12, pmax=0.1,
                 distance=None,
                 max_n_regions=(10, 10, 10, 10),
                 max_region_sizes=(1, 0.5, 0.35, 0.25),
    ):

        self.decision = self.DECISION

        # the region tree which keeps track of the regions and can be
        # balanced for cloning and merging between them, is
        # initialized the first time resample is called because it
        # needs an initial walker
        self._region_tree = None

        # parameters
        self.pmin=pmin
        self.pmax=pmax
        self.seed = seed
        if self.seed is not None:
            rand.seed(self.seed)

        self.max_n_regions = max_n_regions
        self.n_levels = len(max_n_regions)
        self.max_region_sizes = max_region_sizes # in nanometers!

        # distance metric
        self.distance = distance

    @property
    def region_tree(self):
        return self._region_tree

    def resample(self, walkers, delta_walkers=0, debug_prints=False):

        # if the region tree has not been initialized, do so
        if self.region_tree is None:
            self._region_tree = RegionTree(walkers[0].state,
                                          max_n_regions=self.max_n_regions,
                                          max_region_sizes=self.max_region_sizes,
                                           distance=self.distance,
                                           pmin=self.pmin,
                                           pmax=self.pmax)

        ## "Score" the walkers based on the current defined Voronoi
        ## images which assign them to bins/leaf-nodes, possibly
        ## creating new regions, do this by calling the method to
        ## "place_walkers"  on the tree which changes the tree's state
        self.region_tree.place_walkers(walkers)
        if debug_prints:
            print("Assigned regions=\n{}".format(self.region_tree.walker_assignments))

        ## Given the assignments ("scores") (which are on the tree
        ## nodes) decide on which to merge and clone

        # do this by "balancing" the tree. delta_walkers can be
        # specified to increase or decrease the total number of
        # walkers
        merge_groups, walkers_num_clones = self.region_tree.balance_tree(delta_walkers=delta_walkers)
        if debug_prints:
            print("merge_groups\n{}".format(merge_groups))
            print("Walker number of clones\n{}".format(walkers_num_clones))

        # check to make sure we have selected appropriate walkers to clone
        #print images
        if debug_prints:
            print("images\n{}".format(self.region_tree.images))
            print("images_assignments\n{}".format(self.region_tree.regions))

        for walker_idx, n_clones in enumerate(walkers_num_clones):

            if n_clones > 0:
                if len(merge_groups[walker_idx]) > 0:
                    raise ValueError("trying to clone a KEEP_MERGE walker")

                squash_idxs = list(it.chain(merge_groups))
                if walker_idx in squash_idxs:
                    raise ValueError("trying to clone a SQUASH walker")

        # clear the tree of walker information
        self.region_tree.clear_walkers()

        # using the merge groups and the non-specific number of clones
        # for walkers create resampling actions for them (from the
        # Resampler superclass)
        resampling_actions = [self.assign_clones(merge_groups, walkers_num_clones)]

        for step in resampling_actions:
            taken_slots = []
            for decision, instruction in step:

                # unless it is a squash add it to the taken slots
                if decision != 3:
                    try:
                        taken_slots.extend(instruction)
                    except TypeError:
                        taken_slots.append(instruction)

            if len(set(taken_slots)) < len(taken_slots):
                raise ValueError("Multiple assignments to the same slot")

        # perform the cloning and merging
        resampled_walkers = self.DECISION.action(walkers, resampling_actions)

        # Auxiliary data
        aux_data = {"walker_assignments" : np.array(self.region_tree.walker_assignments)}


        return resampled_walkers, resampling_actions, aux_data
