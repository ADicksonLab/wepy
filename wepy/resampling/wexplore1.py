import multiprocessing as mulproc
import random as rand
import itertools as it
from collections import namedtuple
from copy import copy
from copy import deepcopy

import numpy as np

from wepy.resampling.resamplers.resampler  import Resampler
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision

class Node(object):
    def __init__(self, nwalkers=0, nreduc=0, nabovemin=0, children=[],
                 region_id=None, to_clone=0, windx=[], positions=[]
    ):
        self.nwalkers = nwalkers
        self.nreduc = nreduc
        self.nabovemin = nabovemin
        self.children = copy(children)
        if region_id is None:
            self.region_id = []
        else:
            self.region_id = region_id
        self.toclone = to_clone
        self.windx = copy(windx)
        self.positions = copy(positions)

class WExplore1Resampler(Resampler):

    DECISION = MultiCloneMergeDecision

    def __init__(self, seed=None, pmin=1e-12, pmax=0.1,
                 scorer=None,
                 max_n_regions=(10, 10, 10, 10),
                 max_region_sizes=(1, 0.5, 0.35, 0.25),
    ):

        self.decision = self.DECISION
        self.pmin=pmin
        self.pmax=pmax
        self.seed = seed
        self.scorer = scorer
        self.treetop = Node()
        self.max_n_regions = max_n_regions
        self.max_region_sizes = max_region_sizes # in nanometers!


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
        for i,child in enumerate(children):
            final_nwalkers = child.nwalkers + child.toclone
            final_nreduc = child.nreduc + child.toclone

            if (not maxwalk or final_nwalkers > maxwalk) and (final_nreduc >= 1):
                maxwalk = final_nwalkers
                highchild = i

            if (not minwalk or final_nwalkers < minwalk) and (child.nabovemin >= 1):
                minwalk = final_nwalkers
                lowchild = i

        if (minwalk and maxwalk):
            ntrans = min(int((maxwalk - minwalk)/2), children[highchild].nreduc + children[highchild].toclone)
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
        if len(children) > 1:
        # this node has children, balance between them
            parent.nreduc += parent.toclone
            parent.nwalkers += parent.toclone
            if parent.toclone < 0:
                #import ipdb; ipdb.set_trace()
            # parent node has a deficit; needs to merge walkers
            # find children that have reducible walkers
                for child in children:
                    final_nreduc = child.nreduc + child.toclone
                    if final_nreduc >= 1:
                        dif = min(abs(parent.toclone), final_nreduc)
                        parent.toclone += dif
                        child.toclone -= dif
                if parent.toclone < 0:
                    raise ValueError("Error! Children cannot pay their parent's debt")

            if parent.toclone > 0:
            # parent has a surplus! needs to clone walkers
            # find children that have walkers that can be cloned
                for child in children:
                    if child.nabovemin >= 1:
                        child.toclone += parent.toclone
                        parent.toclone = 0
                if parent.toclone > 0:
                    raise ValueError("Error! Children cannot clone walkers!")

            # balance between the children
            # find the maxwalk and minwalk
            minwalk, maxwalk, lowchild, highchild, ntrans = self.getmaxminwalk(children)
            while (minwalk and maxwalk) and (ntrans >= 1):
                # merge walkers in highchild ; clone walkers in lowchild
                children[lowchild].toclone += ntrans
                children[highchild].toclone -= ntrans
                minwalk, maxwalk, lowchild, highchild, ntrans = self.getmaxminwalk(children)

            # children are as balanced as they are going to get
            # now run all children through balancetree
            for child in children:
                self.balancetree(child)
        elif len(children)==1:
            # only one child, just pass on the debt / surplus
            parent.nreduc += parent.toclone
            parent.nwalkers += parent.toclone

            children[0].toclone = parent.toclone
            parent.toclone = 0
            self.balancetree(children[0])
        else:
            # no children, we are at the lowest level of the tree
            # figure out walkers to clone / merge
            if (-parent.toclone > parent.nreduc):
                raise ValueError("Error! node doesn't have enough walkers to merge")
            if parent.toclone < 0:
                while parent.toclone < 0:
                    # MERGE: find the two walkers with the lowest weights
                    r1 = None
                    minwt = None
                    #import ipdb; ipdb.set_trace()
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
            elif parent.toclone >=0:
                while parent.toclone > 0:
                    # pick the one with the highest weight
                    maxwt = None
                    #import ipdb; ipdb.set_trace()
                    for i,w in enumerate(parent.windx):
                        twt = self.walkerwt[w]
                        if (maxwt is None) or twt > maxwt:
                            maxwt = twt;
                            r1walk = w

                    self.num_clones[r1walk] += 1
                    parent.nwalkers +=1
                    parent.toclone -=1


    def get_closest_image(self, walker, images):

        # calculate distance from the walker to all given images
        dists = []
        for i, image in enumerate(images):
            dist = self.scorer.distance.distance(walker['positions'], image['positions'])
            dists.append(dist)

        # get the image that is the closest
        closest_image_idx = np.argmin(dists)

        return dists[closest_image_idx], closest_image_idx

    def define_new_region(self, parent, walker):
        # the parents region assignment, which we will add to
        region_id = copy(parent.region_id)

        # the index of the new child/image/region
        child_idx = len(parent.children)

        # the new region identifier for the new region
        region_id.append(child_idx)

        # make a new node for the region, save the index of that
        # region as a child for the parent node in it's region_idx
        newnode = Node(region_id=region_id, positions=deepcopy(walker))
        parent.children.append(newnode)

        # return the index of the new region node as a child of the parent node
        return child_idx


    # TODO old name: 'getdist'
    def assign_walkers(self, parent, walker, level=0, region_assg=None):

        if region_assg is None:
            region_assg = []

        # children exist for the parent
        if len(parent.children) > 0:

            # find the closest image in this superregion (i.e. parent at current level)
            mindist, close_child_idx = self.get_closest_image(walker, parent.children)

            # if the minimum distance is over the cutoff we need to define new regions
            if mindist > self.max_region_sizes[level]:
                # if there are less children than the maximum number of images for this level
                if len(parent.children) < self.max_n_regions[level]:

                    # we can create a new region
                    close_child_idx = self.define_new_region(parent, walker)

            # the region assignment starts with this image/region index
            region_assg.append(close_child_idx)
            region_assg = self.assign_walkers(parent.children[close_child_idx], walker, level=level+1, region_assg=region_assg)

        # No children so we have to make them if we are not at the
        # bottom, define new regions for this node and make it a
        # parent
        elif level < len(self.max_region_sizes):
            # define a new region because none exist for this parent
            close_child_idx = self.define_new_region(parent, walker)

            # update the region_assg for the walker
            region_assg.append(close_child_idx)

            # recursively call getdist on the just created child one level deeper
            region_assg = self.assign_walkers(parent.children[close_child_idx], walker, level=level+1, region_assg=region_assg)

        return region_assg

    def populate_tree(self, parent, level=0, region_assg=None, debug_prints=False):
        #
        # This function uses the walkerreg assignments to determine
        # how many w-alkers are under each node (nwalkers) at each level
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
                tup = self.populate_tree(child, level=level+1, region_assg=parent_region_assg)
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

    def resample(self, walkers, debug_prints=False):

        self.n_walkers = len(walkers)

        # region assignments
        self.walkerreg = [None for i in range(self.n_walkers)]

        # merge groups
        self.walkers_squashed = [[] for i in range(self.n_walkers)]
        # number of clones to make for each walker
        self.num_clones = [0 for i in range(self.n_walkers)]

        # weights of walkers
        self.walkerwt = [walker.weight for walker in walkers]
        # assign walkers to regionsg
        for i, walker in enumerate(walkers):
            self.walkerreg[i] = self.assign_walkers(self.treetop, walker, level=0, region_assg=None)

        if debug_prints:
            print("region assignments:")
            for i in range(self.n_walkers):
                print(i,self.walkerreg[i])

        # populate the tree
        n_walkers, n_reducible_walkers, n_above_minp = self.populate_tree(self.treetop, region_assg=None)

        # balance the tree (determines walkers_squashed and num_clones)
        self.balancetree(self.treetop)

        if debug_prints:
            print("walkers squashed:")
            for i in range(self.n_walkers):
                print(i,self.walkers_squashed[i])
            print("num_clones:",self.num_clones)

        resampling_actions = self.assign_clones(self.walkers_squashed, self.num_clones)

        resampling_actions = [resampling_actions]
        # actually do the cloning and merging of the walkers
        resampled_walkers = self.DECISION.action(walkers, resampling_actions)

        data = {'assignments' : np.array(self.walkerreg)}

        return resampled_walkers, resampling_actions, data
