import multiprocessing as mulproc
import random as rand
import itertools as it
from collections import namedtuple
from copy import copy

from wepy.resampling.resampler import Resampler
from wepy.resampling.decisions.clone_merge import CloneMergeDecision

class Node(object):
    def __init__(self, nwalkers=0, nreduc=0, nabovemin=0, children=[], ID=[], to_clone=0, windx=[], xyz=[]):
        self.nwalkers = nwalkers
        self.nreduc = nreduc
        self.nabovemin = nabovemin
        self.children = tchildren
        self.ID = ID
        self.to_clone = to_clone
        self.windx = windx
        self.xyz = xyz

class WExplore1Resampler(Resampler):

    DECISION = CloneMergeDecision

    def __init__(self, seed=None, pmin=1e-12, pmax=0.1, distance_function=None, maximage=[10,10,10,10], cellsize=[1.0,0.5,0.35,0.25]):
        self.pmin=pmin
        self.pmax=pmax
        self.seed = seed
        self.distance_function = distance_function
        self.treetop = Node()
        self.maximage = maximage
        self.cellsize = cellsize # in nanometers!

    def getmaxminwalk(children):
        #
        # This is a helper function for balancetree, which returns the
        # children with the highest and lowest numbers of walkers. As
        # well as the number of transitions that should occur to even them
        # out (ntrans).
        #
        minwalk = None
        maxwalk = None
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
        return (minwalk,maxwalk,lowchild,highchild,ntrans)

    def balancetree(parent):
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
            # parent node has a deficit; needs to merge walkers
            # find children that have reducible walkers
                for child in children:
                    final_nreduc = child.nreduc + child.toclone
                    if final_nreduc >= 1:
                        dif = min(parent.toclone, final_nreduc)
                        parent.toclone += dif
                        child.toclone -= dif
                if parent.toclone < 0:
                    raise ValueError("Error! Children cannot pay their parent's debt")

            if parent.toclone > 0:
            # parent has a surplus! needs to clone walkers
            # find children that have walkers that can be cloned
                for child in children:
                    if child.nabovemin >= 1:
                        child_toclone += parent.toclone
                        parent.toclone = 0
                if parent.toclone > 0:
                    raise ValueError("Error! Children cannot clone walkers!")

            # balance between the children
            # find the maxwalk and minwalk
            (minwalk,maxwalk,lowchild,highchild,ntrans) = self.getmaxminwalk(children)
            while (minwalk and maxwalk) and (ntrans >= 1):
                # merge walkers in highchild ; clone walkers in lowchild
                children[lowchild].toclone += ntrans
                children[highchild].toclone -= ntrans
                (minwalk,maxwalk,lowchild,highchild,ntrans) = self.getmaxminwalk(children)

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
            if (parent.toclone > parent.nreduc):
                raise ValueError("Error! node cannot pay its debt!")
            if parent.toclone < 0:
                while parent.toclone < 0:
                    # MERGE: find the two walkers with the lowest weights
                    r1 = None
                    minwt = None
                    for i in range(parent.nwalkers):
                        twt = self.walkerwt[parent.windex[i]]
                        if not r1 or twt < minwt:
                            minwt = twt
                            r1 = i
                    r2 = None
                    minwt = None
                    for i in range(parent.nwalkers):
                        if i != r1:
                            twt = self.walkerwt[parent.windex[i]]
                            if not r2 or twt < minwt):
                                minwt = twt;
                                r2 = i

                    r1index = parent.windex[r1]
                    r2index = parent.windex[r2]
                    r3 = rand(self.walkerwt[r1index] + self.walkerwt[r2index])
                    if r3 < self.walkerwt[r1index]:
                        keep_idx = r1index
                        squash_idx = r2index
                    else:
                        keep_idx = r2index
                        squash_idx = r1index
                    self.walkerwt[keep_idx] += self.walkerwt[squash_idx]
                    self.walkerwt[squash_idx] = 0
                    self.walkers_squashed[keep_idx] += squash_idx + self.walkers_squashed[squash_idx]
                    self.walkers_squashed[squash_idx] = []
                    parent.windex.remove(squash_idx)
                    parent.nwalkers -=1
                    parent.toclone +=1
            elif parent.toclone >=0:
                while parent.toclone > 0:
                    # pick the one with the highest weight
                    r1 = None
                    maxwt = None
                    for i in range(parent.nwalkers):
                        twt = walkerwt[parent.windex[i]]
                        if r1 or twt > maxwt:
                            maxwt = twt;
                            r1 = i
                            r1walk = parent.windex[r1]

                    self.num_clones[r1walk]++
                    parent.nwalkers +=1
                    parent.toclone -=1
        return

    def getclosest(xyz, children):
        # looks through the set of children and computes the closest
        # using a distance metric
        mindist = None
        closeregind = None
        for i,c in enumerate(children):
            d = self.distance_function.xyzdistance(xyz,c.xyz)
            if not mindist or d < mindist:
                mindist = d
                closeregind = i
        return mindist, closeregind

    def definenew(level, parent, xyz):
        tID = copy(parent.ID)
        index = len(parent.children)
        tID.append(index)
        newnode = Node(ID=tID, xyz=xyz)
        parent.children.append(newnode)

        return index


    def getdist(parent, xyz, level=0, ID=[]):
        newind = []
        if len(parent.children) > 0:
            mindist, closereg = self.getclosest(xyz, parent.children)
            if mindist > self.cellsize[level]:
                if len(parent.children < self.maximage[level]):
                    closereg = self.definenew(level, parent, xyz)

                ind.append(closereg)
                newind = self.getdist(parent.children[closereg], xyz, level=level+1, ID=ind)
        else:
            # bottom level
            if level == len(self.cellsize):
                newind = ind
            else:
                mindist, closereg = self.getclosest(xyz, parent.children)
                if mindist > self.cellsize[level]:
                    closereg = self.definenew(level, parent, xyz)
                ind.append(closereg)
        return newind

    def populatetree(parent, level=0, ID=[]):
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
        nunder = 0
        nreduc = 0
        nabovemin = 0
        if level < maxlevel:
            for i,child in enumerate(parent.children):
                tID = copy(ID)
                tID.append(i)
                cwalk, creduc, cabovemin = self.populatetree(child, level=level+1, ID=tID)
                if cwalk > 0:
                    node('nwalkers') = cwalk
                    node('nreduc') = creduc
                    node('nabovemin') = cabovemin
                parent.children.append(node)
            nunder += cwalk
            nreduc +=creduc
            nabovemin += cabovemin
        else:
            avgwt = 0
            wts = []
            for i in range(nwalk-1):
                wreg = self.walkerreg[i]
                if self.walkerreg[i] == ID: # element-wise comparison of two lists
                    nunder +=1
                    parent.windx.append(i)
                    wts.append(self.walkerwt[i])
                    if walkerwt[i] > self.pmin:
                        nabovemin += 1
            nreduc = 0
            if nunder > 1:
                sortwts = wts.sort()
                wtsum = sortwts[0]
                for i in range(1,len(sortwts)):
                    wtsum += sortwts[i]
                    if wtsum < self.pmax:
                        nreduc = i
        return nunder, nreduc, nabovemin

    def resample(self, walkers, debug_prints=False):

        n_walkers = len(walkers)
        self.walkers_squashed = [[] for i in range(n_walkers)]
        self.num_clones = [0 for i in range(n_walkers)]

        self.walkerwt = [walker.weight for walker in walkers]

        # assign walkers to regions
        for i,w in enumerate(walkers):
            self.walkerreg[i] = self.getdist(self.treetop, w.xyz)

        if debug_prints:
            print("region assignments:")
            for i in range(n_walkers):
                print(i,self.walkerreg[i])

        # populate the tree
        self.populatetree(self.treetop)

        # balance the tree (determines walkers_squashed and num_clones)
        self.balancetree(self.treetop)

        if debug_prints:
            print("walkers squashed:")
            for i in range(n_walkers):
                print(i,self.walkers_squashed[i])
            print("num_clones:",self.num_clones)

        resampling_actions = [actions_from_list(self.walkers_squashed,self.num_clones)]

        # actually do the cloning and merging of the walkers
        resampled_walkers = self.DECISION.action(walkers, resampling_actions)

        data = {'assignments' : np.array(self.walkerreg)}

        return resampled_walkers, resampling_actions, data
