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

class RegionTree(nx.DiGraph):

    ROOT_NODE = ()
    def __init__(self, init_image, max_n_regions, max_region_size, distance,
                 pmin=1e-12, pmax=0.5):
        super().__init__()
        self._max_n_regions = max_n_regions
        self._n_levels = len(max_n_regions)
        self._max_region_size = max_region_size
        self._distance = distance
        self._pmin = pmin
        self._pmax = pmax

        self._walkers = []
        self._walker_assignments = []

        image_idx = 0
        # get the preimage using the distance object
        preimage = self.distance.preimage(init_image)
        self._images = [preimage]

        parent_id = self.ROOT_NODE
        self.add_node(parent_id, image_idx=0,
                      n_walkers=0,
                      n_reduc=0,
                      n_above_pmin=0,
                      balance=0,
                      walkers=[])

        # make the first branch
        for level in range(len(max_n_regions)):
            child_id = parent_id + (0,)
            self.add_node(child_id, image_idx=image_idx,
                          n_walkers=0,
                          n_reduc=0,
                          n_above_pmin=0,
                          balance=0,
                          walkers=[])
            self.add_edge(parent_id, child_id)
            parent_id = child_id

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
    def max_region_size(self):
        return self._max_region_size

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
    def walkers(self):
        return self._walkers

    def add_child(self, parent_id, image_idx):
        # make a new child id which will be the next index of the
        # child with the parent id
        child_id = parent_id + (len(self.children(parent_id)), )

        # create the node with the image_idx
        self.add_node(child_id,
                      image_idx=image_idx,
                      n_walkers=0,
                      n_reduc=0,
                      n_above_pmin=0,
                      balance=0,
                      walkers=[])

        # make the edge to the child
        self.add_edge(parent_id, child_id)

        return child_id

    def children(self, parent_id):
        return list(self.adj[parent_id].keys())

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
        image_idx = len(self.images)
        # get the true preimage of the distance function to save as
        # the image
        preimage = self.distance.preimage(image)
        self.images.append(preimage)

        branch_level = len(parent_id)
        # go down from there and create children
        for level in range(branch_level, self.n_levels):
            child_id = self.add_child(parent_id, image_idx)
            parent_id = child_id

        # return the leaf node id of the new branch
        return child_id

    def assign(self, state):

        assignment = []
        dists = []
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
                image = self.images[self.node[level_node]['image_idx']]

                # preimage of the state
                state_preimage = self.distance.preimage(state)
                dist = self.distance.preimage_distance(state_preimage, image)
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

        # set all the node attributes to their defaults
        for node_id in self.nodes:
            self.node[node_id]['n_walkers'] = 0
            self.node[node_id]['n_reduc'] = 0
            self.node[node_id]['walkers'] = []

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

                # if we are over the max region distance we have found a
                # new region so we branch the region_tree at that level
                if distance > self.max_region_size[level]:
                    image = self.distance.preimage(walker.state)
                    parent_id = assignment[:level]
                    assignment = self.branch_tree(parent_id, image)
                    # we have made a new branch so we don't need to
                    # continue this loop
                    break

            # save the walker assignment
            self._walker_assignments.append(assignment)
            self._walkers.append(walker)

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
                self.node[node_id]['walkers'].append(walker_idx)
                if above_pmin:
                    self.node[node_id]['n_above_pmin'] += 1

        # after placing all the walkers we calculate the number of
        # reducible walkers for each node

        # for each leaf node calculate the number of reducible walkers
        # (i.e. the largest possible number of merges that could occur
        # taking in to account the pmax (max weight) constraint)
        for node_id in self.leaf_nodes():
            if self.node[node_id]['n_walkers'] > 1:

                # the weights of the walkers in this node
                weights = np.sort([self.walkers[i].weight
                                   for i in self.node[node_id]['walkers']])

                # figure out the most possible mergeable walkers
                # assuming they cannot ever be larger than pmax
                sum_weights = weights[0]
                for i in range(1, len(weights)):
                    sum_weights += weights[i]
                    # if we still haven't gone past pmax set the n_red
                    # to the current index
                    if sum_weights < self.pmax:
                        self.node[node_id]['n_reduc'] = i

                # increase the reducible walkers for the higher nodes
                # in this leaf's branch
                for level in reversed(range(self.n_levels-1)):
                    branch_node_id = node_id[:level]
                    self.node[branch_node_id]['n_reduc'] += self.node[node_id]['n_reduc']

class ImageNode(object):

    def __init__(self, image_idx=None, children=None):
        self.image_idx = image_idx
        if children is None:
            self.children = []
        else:
            self.children = children

    def is_leaf_node(self):
        if len(self.children) <= 0:
            return True
        else:
            return False

class WExplore1Resampler(Resampler):

    DECISION = MultiCloneMergeDecision

    def __init__(self, seed=None, pmin=1e-12, pmax=0.1,
                 distance=None,
                 max_n_regions=(10, 10, 10, 10),
                 max_region_sizes=(1, 0.5, 0.35, 0.25),
    ):

        self.decision = self.DECISION

        # parameters
        self.pmin=pmin
        self.pmax=pmax
        self.seed = seed
        self.max_n_regions = max_n_regions
        self.n_levels = len(max_n_regions)
        self.max_region_sizes = max_region_sizes # in nanometers!

        # distance metric
        self.distance = distance

        # state of the resampler, i.e. hierarchy of images defining
        # the regions, this datastructure is a tree of nodes

        self.images = []

        # initialize the hierarchy given the number of levels
        self.root_node = ImageNode(image_idx=None, children=[])


        node = self.root_node
        # graph of the tree
        self._tree = Tree()
        node_id = ()
        self._tree.add_node(node_id, image_idx=None)
        for level in range(self.n_levels):

            # the child node
            child_id += (level,)
            self._tree.add_node(child_id, image_idx=None)

            # add the edge to the graph for the child
            self._tree.add_edge(node_id, child_id)

            new_node = ImageNode(image_idx=None, children=[])
            node.children.append(new_node)
            node = new_node

    def branch_tree(self, parent_id, image):

        # add the new image to the image index
        image_idx = len(self.images)
        self.images.append(image)

        # get the parent for which we will make a new child for
        node = self.root_node
        for child_idx in parent_id:
            node = node.children[child_idx]

        # get the index of the new node starting the new chain
        new_child_idx = len(node.children)

        # once we have parent node we add a new branch down to the
        # bottom level
        for level in range(len(parent_id), self.n_levels):
            # make a new child node for the parent
            child_node = ImageNode(image_idx=image_idx, children=[])
            # add this as a child at the current level
            node.children.append(child_node)
            # move onto the next node
            node = child_node

        # return the new leaf node id
        leaf_node_id = list(parent_id)
        leaf_node_id = leaf_node_id + [0 for i in range(len(parent_id), self.n_levels)]
        leaf_node_id[len(parent_id)] += new_child_idx

        return tuple(leaf_node_id)


    def place_walker(self, walker):
        """Given the current hierarchy of Voronoi regions and the walker
        either assign it to a region or create a new region if
        necessary. This could mutate the state of the hierarchy.

        """

        # check to see if there are any images defined at all
        if self.root_node.image_idx is None:
            # if not we set the first images to the state of the first walker
            image_idx = len(self.images)
            # TODO distance preimage of the state is what should be
            # stored for the image

            # save the image of this walker for defining regions
            self.images.append(walker.state)

            # create the initial nodes for each level with this
            # initial image
            node = self.root_node
            node.image_idx = image_idx
            for level in range(self.n_levels):
                node = node.children[0]
                node.image_idx = image_idx

            assignment = tuple([0 for level in range(self.n_levels)])
            distances = tuple([0.0 for level in range(self.n_levels)])

        # if the image hierarchy has been initialized, assign this
        # walker to it
        else:
            # assign the walker to the defined regions
            assignment, distances = self.assign_walker(walker)

        # check each level's distance to see if it is beyond the
        # maximum allowed distance. If it is create a new branch
        # starting at the parent of the node at the level the walker
        # was closest to
        for level, distance in enumerate(distances):
            if distance > self.max_region_sizes[level]:
                image = walker.state
                parent_id = assignment[:level]
                assignment = self.branch_tree(parent_id, image)
                break

        return assignment


    def assign_walker(self, walker):

        assignment = []
        dists = []
        # perform a n-ary search through the hierarchy of regions by
        # performing a distance calculation to the images at each
        # level starting at the top
        node = self.root_node
        for level in range(self.n_levels):
            level_nodes = node.children

            # perform a distance calculation to all nodes at this
            # level
            image_dists = []
            for level_node in level_nodes:
                # get the image
                image = self.images[level_node.image_idx]
                dist = self.distance.distance(walker.state, image)
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

    def place_walkers(self, walkers):
        """Assign walkers to Voronoi regions. This function will add Voronoi
        cells to the hierarchy leaves if necessary, thus mutating the
        state of the resampler (i.e. the tree of Voronoi
        regions/images). This is the 'Scorer' in terms of the
        resampling framework.

        Returns a tuple of the region assignment for each level of the
        hierarchy for each walker.


        [(level_0_region_idx_0, level_1_region_idx_0, level_2_region_idx_0),
        (level_0_region_idx_1, level_1_region_idx_1, level_2_region_idx_1)]

        For two walkers with 3 levels, with the number of regions per
        level set at (2, 2, 2) (2 for each level) we might have:

        [(0, 1, 0),
         (0, 0, 0)
        ]

        """

        # region assignments
        walker_assignments = [None for i in range(len(walkers))]

        # assign walkers to regions
        for i, walker in enumerate(walkers):
            walker_assignments[i] = self.place_walker(walker)

        return walker_assignments

    def minmax_beneficiaries(self, children):
        #
        # This is a helper function for balancetree, which returns the
        # children with the highest and lowest numbers of walkers. As
        # well as the number of transitions that should occur to even them
        # out (ntrans).
        #

        min_n_walkers = None
        min_child = None

        max_n_walkers = None
        max_child = None

        # test each walker sequentially
        for i, child in enumerate(children):


            n_walkers = self.node[child]['n_walkers']
            n_reduc = self.node[child]['n_reduc']
            n_above_pmin = self.node[child]['n_above_pmin']

            # we need to take into account the balance inherited
            # from the parent when calculating the total number of
            # walkers that can be given to other node/regions
            total_n_walkers = n_walkers + self.node[child]['balance']
            total_n_reduc = n_reduc + self.node[child]['balance']

            # the maximum number of walkers that will exist after
            # cloning, if the number of reducible walkers is 1 or more
            if ((not max_child) or (n_walkers > max_n_walkers)) and \
               (n_reduc >= 1):

                max_n_walkers = n_walkers
                max_child = i

            # the minimum number of walkers that will exist after
            # cloning, if the number of number of walkers above the
            # minimum weight is 1 or more
            if ((not min_child) or (n_walkers < min_n_walkers)) and \
               (n_above_pmin >= 1):

                min_n_walkers = n_walkers
                min_child = i


        return min_child, max_child

    def ntrans():
        # if a minwalk and maxwalk are defined calculate ntrans, the
        # number of 'transitions' which is the number of walkers that
        # will be passed between them.
        if (minwalk and maxwalk):
            # the number of transitions is either the midpoint between
            # them (rounded down to an integer) or the sum of the
            # number of reducible walkers plus the number of toclones
            # for the walker with the highest number of walkers,
            # whichever is smaller
            ntrans = min(int((maxwalk - minwalk)/2),
                         children[highchild].nreduc + children[highchild].toclone)
        # initialize it to 0 if it is not set already
        else:
            ntrans = 0

        return ntrans

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
            # there are more than one child so we accredit balances
            # between them
            if len(children) > 1:

                # if the node had a balance assigned to previously,
                # apply this to the number of walkers it has
                #self.node[parent]['n_reduc'] += self.node[parent]['balance']
                #self.node[parent]['n_walkers'] += self.node[parent]['balance']

                # if the parent has a non-zero balance we either
                # increase (clone) or decrease (merge) the balance

                # these poor children are inheriting a debt and must
                # decrease the total number of their credits :(
                if self.node[parent]['balance'] < 0:

                    # find children with mergeable walkers and account
                    # for them in their balance
                    for child in children:

                        # if this child has any mergeable walkers
                        if self.node[child]['n_reduc'] >= 1:
                            # we use those for paying the parent's debt
                            diff = abs(min(self.node[child]['n_reduc'],
                                           self.node[parent]['balance']))
                            self.node[parent]['balance'] += diff
                            self.node[child]['balance'] -= diff

                        if self.node[parent]['balance'] < 0:
                            raise ValueError("Children cannot pay their parent's debt")


                # these lucky children are inheriting a positive number of
                # credits!! :)
                elif self.node[parent]['balance'] > 0:

                    for child in children:
                        if self.node[child]['n_above_pmin']:
                            self.node[child]['balance'] += self.node[parent]['balance']:
                            self.node[parent]['balance'] = 0

                    if self.node[parent]['balance'] > 0:
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
                min_child, max_child = self.minmax_beneficiaries(children)

                min_child_id = children[min_child]
                total_min_n_walkers = self.node[children[min_child]]['n_reduc'] + \
                                      self.node[children[min_child]]['balance']

                max_child_id = children[max_child]
                total_max_n_walkers = self.node[children[max_child]]['n_reduc'] + \
                                      self.node[children[max_child]]['balance']


                # The min_child needs to have at least one clonable
                # walker and the max_child needs to have at least one
                # mergeable walker. If they do not (None from the
                # minmax function) then we cannot assign a number of
                # shares for them to give each other
                if None in (min_child, max_child):
                    shares = 0
                # if they do exist then we find the number of shares
                # to give
                else:
                    # the sibling with the greater number of shares
                    # (from both previous resamplings and inherited
                    # from the parent) will give shares to the sibling
                    # with the least. The number it will share is the
                    # number that will make them most similar rounding
                    # down (i.e. midpoint)
                    n_shares = math.floor((total_max_n_walkers - total_min_n_walkers)/2)

            # only one child so it just inherits balance
            elif len(children) == 1:
                pass
            # the leaf level, balances must be paid
            else:
                pass

    def getmaxminwalk(self, children):
        #
        # This is a helper function for balancetree, which returns the
        # children with the highest and lowest numbers of walkers. As
        # well as the number of transitions that should occur to even them
        # out (ntrans).
        #

        minwalk = None
        maxwalk = None
        highchild = None
        lowchild = None

        for i, child in enumerate(children):
            # calculate the number of walkers that will be present
            # after cloning
            final_nwalkers = child.nwalkers + child.toclone
            # calculate the number of reducible walkers that will be
            # present after cloning
            final_nreduc = child.nreduc + child.toclone

            # update the maxwalk, which is the maximum number of
            # walkers that will exist after cloning, if the number of
            # reducible walkers is 1 or more
            if (not maxwalk or final_nwalkers > maxwalk) and (final_nreduc >= 1):
                maxwalk = final_nwalkers
                highchild = i

            # update the minwalk, which is the minimum number of
            # walkers that will exist after cloning, if the number of
            # number of walkers above the minimum weight is 1 or more
            if (not minwalk or final_nwalkers < minwalk) and (child.nabovemin >= 1):
                minwalk = final_nwalkers
                lowchild = i

        # if a minwalk and maxwalk are defined calculate ntrans, the
        # number of 'transitions' which is the number of walkers that
        # will be passed between them.
        if (minwalk and maxwalk):
            # the number of transitions is either the midpoint between
            # them (rounded down to an integer) or the sum of the
            # number of reducible walkers plus the number of toclones
            # for the walker with the highest number of walkers,
            # whichever is smaller
            ntrans = min(int((maxwalk - minwalk)/2),
                         children[highchild].nreduc + children[highchild].toclone)
        # initialize it to 0 if it is not set already
        else:
            ntrans = 0

        return minwalk, maxwalk, lowchild, highchild, ntrans


    def balancetree(self, parent):
        #
        # This is a recursive function, that balances each level of the image tree.
        # In each call, the argument is the parent node, and the balancing is done
        # between the children of that parent.
        #
        # The parent node passes on a "balancing debt" or surplus to its children
        # which is saved in the "toclone" variable
        #
        # The main purpose of this function is to update the following class variables:
        #     self.walkers_squashed
        #     self.num_clones
        # these are used in the resampler to write a resampling decision.
        #

        children = parent.children

        # this node has children, balance between them
        if len(children) > 1:

            # add the number of walkers to clone to both the number of
            # reducible walkers and the total number of walkers
            parent.nreduc += parent.toclone
            parent.nwalkers += parent.toclone

            # if there is a negative number of toclone the parent node
            # has a deficit; needs to merge walkers.
            if parent.toclone < 0:

                # find children that have reducible walkers
                for child in children:
                    final_nreduc = child.nreduc + child.toclone
                    if final_nreduc >= 1:
                        dif = min(abs(parent.toclone), final_nreduc)
                        parent.toclone += dif
                        child.toclone -= dif
                if parent.toclone < 0:
                    raise ValueError("Error! Children cannot pay their parent's debt")

            # if the number of toclone is greater than 0 the parent
            # has a surplus and needs to clone walkers
            if parent.toclone > 0:
                # find children that have walkers that can be cloned
                for child in children:
                    if child.nabovemin >= 1:
                        child.toclone += parent.toclone
                        parent.toclone = 0
                if parent.toclone > 0:
                    raise ValueError("Error! Children cannot clone walkers!")

            # balance between the children

            # find the nodes with the highest and lowest numbers of
            # walkers, and the number of transitions that are to occur
            # between them
            minwalk, maxwalk, lowchild, highchild, ntrans = self.getmaxminwalk(children)

            # if the number of transitions is 1 or more
            while (minwalk and maxwalk) and (ntrans >= 1):
                # merge walkers in highchild
                children[lowchild].toclone += ntrans
                # clone walkers in lowchild
                children[highchild].toclone -= ntrans
                # then recalculate the values
                minwalk, maxwalk, lowchild, highchild, ntrans = self.getmaxminwalk(children)

            # recursively call balancetree
            # children are as balanced as they are going to get
            # now run all children through balancetree
            for child in children:
                self.balancetree(child)

        # if there is only one child
        elif len(children) == 1:
            # only one child, just pass on the debt / surplus
            parent.nreduc += parent.toclone
            parent.nwalkers += parent.toclone
            children[0].toclone = parent.toclone
            parent.toclone = 0

            # recursively call balancetree
            self.balancetree(children[0])

        # no children, we are at the lowest level of the tree
        else:
            # figure out walkers to clone or merge

            # if the toclone debt is larger than the number of
            # reducible walkers we won't have enough walkers to merge
            if (-parent.toclone > parent.nreduc):
                raise ValueError("Error! node doesn't have enough walkers to merge")

            # merge and update until there is no debt
            while parent.toclone < 0:
                # MERGE: find the two walkers with the lowest weights
                r1 = None
                minwt = None
                for i in range(parent.nwalkers):
                    twt = self.walkerwt[parent.windx[i]]
                    if (r1 is None) or (twt < minwt):
                        minwt = twt
                        r1 = i
                r2 = None
                minwt = None
                for i in range(parent.nwalkers):
                    if i != r1:
                        twt = self.walkerwt[parent.windx[i]]
                        if (r2 is None) or (twt < minwt):
                            minwt = twt;
                            r2 = i

                r1index = parent.windx[r1]
                r2index = parent.windx[r2]
                r3 = rand.random() * (self.walkerwt[r1index] + self.walkerwt[r2index])
                if r3 < self.walkerwt[r1index]:
                    keep_idx = r1index
                    squash_idx = r2index
                else:
                    keep_idx = r2index
                    squash_idx = r1index
                self.walkerwt[keep_idx] += self.walkerwt[squash_idx]
                self.walkerwt[squash_idx] = 0
                self.walkers_squashed[keep_idx] += [squash_idx] + self.walkers_squashed[squash_idx]
                self.walkers_squashed[squash_idx] = []
                parent.windx.remove(squash_idx)
                parent.nwalkers -=1
                parent.toclone +=1

            # if there is no debt and rather an excess, clone until
            # there is no excess
            while parent.toclone > 0:
                # pick the one with the highest weight
                maxwt = None
                for i,w in enumerate(parent.windx):
                    twt = self.walkerwt[w]
                    if (maxwt is None) or twt > maxwt:
                        maxwt = twt;
                        r1walk = w

                self.num_clones[r1walk] += 1
                parent.nwalkers +=1
                parent.toclone -=1

    def n_walkers_per_region(assignments):
        # determine the number of walkers under each node

        n_region_walkers = {}
        # for each walker assignment record the count for each node it
        # is in in a dictionary of the node id tuples
        for assignment in assignments:
            node_id = ()
            # for each level of the assignment, count it for each
            # branch
            for level in assignment:
                node_id += (level,)

                n_region_walkers[node_id] += 1

        return n_region_walkers

    
    def populatetree(self, parent, level=0, region_assg=None, debug_prints=False):
        #
        # This function uses the walkerreg assignments to determine
        # how many walkers are under each node (nwalkers) at each level
        # of the tree. It also determines the number of those walkers
        # which are reducible (nreduc) as well as the number of walkers
        # that have weights above the minimum probability (nabovemin).
        #
        # If this is called at the "treetop" it will populate the entire
        # tree.
        #
        n_walkers = 0
        n_reducible_walkers = 0
        n_above_minp = 0

        if region_assg is None:
            parent_region_assg = []
        else:
            parent_region_assg = region_assg

        # if there are still levels in this hierarchy
        if level < len(self.max_region_sizes):

            # for the children under this parent, recursively call
            # this function. If they are a leaf node it will bottom
            # out and find the info on the walkers in that
            # leaf-node/region, else it will repeat until then
            for i, child in enumerate(parent.children):

                child_region_assg = copy(parent_region_assg)
                parent_region_assg.append(i)

                # recursively call
                tup = self.populate_tree(child, level=level+1,
                                         region_assg=parent_region_assg)
                child.n_walkers = tup[0]
                child.n_reducible_walkers = tup[1]
                child.n_above_minp = tup[2]

                n_walkers += tup[0]
                n_reducible_walkers += tup[1]
                n_above_minp += tup[2]

                if debug_prints:
                    print(child.region_id, *tup)

        # if we are at the bottom of the hierarchy, we want to save
        # which walkers are in these leaf nodes, and calculate the
        # number of reducible walkers, which can then be used to
        # balance the tree at a higher level
        else:

            leaf_walker_weights = []
            leaf_walker_idxs = []

            # save values for walkers in this leaf node/region/image
            for walker_idx in range(self.n_walkers):

                # if this walker is assigned to this region (we are at
                # the bottom here) then save information about the
                # walker in this node
                if self.walkerreg[walker_idx] == region_assg:

                    # increment the total count of walkers under this node (parent)
                    n_walkers += 1

                    # add this walker index to the list of walker
                    # indices under this node
                    leaf_walker_idxs.append(walker_idx)
                    # save the weight of this walker in the node
                    leaf_walker_weights.append(self.walkerwt[walker_idx])

                    # if this walker is above the minimum
                    # probability/weight increment the number of such
                    # walkers under the node
                    if self.walkerwt[walker_idx] > self.pmin:
                        n_above_minp += 1

            parent.windx = leaf_walker_idxs


            # calculate the number of reducible walkers under this
            # node, which is n_red = n_walkers - 1 with the constraint
            # that the largest possible merge is not greater than the
            # maximimum probability for a walker

            # must be more than 1 walker
            n_reducible_walkers = 0
            if n_walkers > 1:

                # figure out the most possible mergeable walkers
                # assuming they cannot ever be larger than pmax
                sorted_weights = list(np.sort(leaf_walker_weights))
                sum_weights = sorted_weights[0]
                for i in range(1, len(sorted_weights)):
                    sum_weights += sorted_weights[i]

                    # if we still haven't gone past pmax set the n_red
                    # to the current index
                    if sum_weights < self.pmax:
                        n_reducible_walkers = i

        return n_walkers, n_reducible_walkers, n_above_minp

    def clone_merge(self, walker_assignments):

        # calculate values of the nodes on the tree, includes the
        # reducible walkers, total number of walkers, and the number
        # of walkers that are still above the minimum weight
        walker_weights = [walker.weight for walker in walkers]
        _ = self.populate_tree(self.treetop, walker_weights, region_assg=None)



        # balance the tree through cloning and merging (returns merge
        # groups and clone numbers for the walkers)
        merge_groups, walkers_num_clones = self.balance_tree()

        # figure out the exact decisions
        resampling_actions = self.assign_clones(merge_groups, walkers_num_clones)

        return resampling_actions

    def resample(self, walkers):

        ## "Score" the walkers based on the current defined Voronoi
        ## images and assign them to bins/leaf-nodes
        walker_assignments = self.place_walkers(walkers)

        ## Given the assignments ("scores") decide on which to merge
        ## and clone
        resampling_actions = self.clone_merge(walker_assignments)

        ## perform the cloning and merging
        resampled_walkers = self.DECISION.action(walkers, [resampling_actions])

        ## Auxiliary data
        aux_data = {}

        return resampled_walkers, resampling_actions, aux_data


    # def get_closest_image(self, walker, images):

    #     # calculate distance from the walker to all given images
    #     dists = []
    #     for i, image in enumerate(images):
    #         dist = self.distance.distance(walker['positions'], image['positions'])
    #         dists.append(dist)

    #     # get the image that is the closest
    #     closest_image_idx = np.argmin(dists)

    #     return dists[closest_image_idx], closest_image_idx

    # def define_new_region(self, parent, walker):
    #     # the parents region assignment, which we will add to
    #     region_id = copy(parent.region_id)

    #     # the index of the new child/image/region
    #     child_idx = len(parent.children)

    #     # the new region identifier for the new region
    #     region_id.append(child_idx)

    #     # make a new node for the region, save the index of that
    #     # region as a child for the parent node in it's region_idx
    #     newnode = Node(region_id=region_id, positions=deepcopy(walker))
    #     parent.children.append(newnode)

    #     # return the index of the new region node as a child of the parent node
    #     return child_idx


    # # TODO old name: 'getdist'
    # def assign_walker(self, walker, level=0, region_assg=None):

    #     if region_assg is None:
    #         region_assg = []

    #     # children exist for the parent
    #     if len(parent.children) > 0:

    #         # find the closest image in this superregion (i.e. parent at current level)
    #         mindist, close_child_idx = self.get_closest_image(walker, parent.children)

    #         # if the minimum distance is over the cutoff we need to define new regions
    #         if mindist > self.max_region_sizes[level]:
    #             # if there are less children than the maximum number of images for this level
    #             if len(parent.children) < self.max_n_regions[level]:

    #                 # we can create a new region
    #                 close_child_idx = self.define_new_region(parent, walker)

    #         # the region assignment starts with this image/region index
    #         region_assg.append(close_child_idx)
    #         region_assg = self.assign_walkers(parent.children[close_child_idx], walker,
    #                                           level=level+1, region_assg=region_assg)

    #     # No children so we have to make them if we are not at the
    #     # bottom, define new regions for this node and make it a
    #     # parent
    #     elif level < len(self.max_region_sizes):
    #         # define a new region because none exist for this parent
    #         close_child_idx = self.define_new_region(parent, walker)

    #         # update the region_assg for the walker
    #         region_assg.append(close_child_idx)

    #         # recursively call getdist on the just created child one level deeper
    #         region_assg = self.assign_walkers(parent.children[close_child_idx], walker,
    #                                           level=level+1, region_assg=region_assg)

    #     return region_assg


# class Node(object):

#     def __init__(self, nwalkers=0, nreduc=0, nabovemin=0, children=[],
#                  region_id=None, to_clone=0, windx=[], positions=[]
#     ):
#         self.nwalkers = nwalkers
#         self.nreduc = nreduc
#         self.nabovemin = nabovemin
#         self.children = copy(children)
#         if region_id is None:
#             self.region_id = []
#         else:
#             self.region_id = region_id
#         self.toclone = to_clone
#         self.windx = copy(windx)
#         self.positions = copy(positions)
