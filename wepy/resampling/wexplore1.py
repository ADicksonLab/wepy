import multiprocessing as mulproc
import random as rand
import itertools as it
from  collections import namedtuple

from wepy.resampling.resampler import Resampler, ResamplingRecord
from wepy.resampling.clone_merge import NothingInstructionRecord, CloneInstructionRecord,\
                                        SquashInstructionRecord, KeepMergeInstructionRecord
from wepy.resampling.clone_merge import CloneMergeDecision, CLONE_MERGE_INSTRUCTION_DTYPES

maximage = (10, 10, 10, 10, 10)
cellsize = (6.0, 3.0, 1.5, 0.8, 0.4)

class Node(object):
    def __init__(self, nwalkers, nreduc, nabovemin, tchildren , ID, tclone, windx, coordinates):
        self.nwalkers = nwalkers
        self.nreduc = nreduc
        self.nabovemin = nabovemin
        self.children = tchildren
        self.ID = ID
        self.tclone = tclone
        self.windx = windx
        self.coordinates = coordinates
    def append(self, index, child):
        self.tchildren.append(child)


def balancetree(parentref):
    tID4 = parenref.ID
    childern = parentref.children
    if len(children) > 1:
        parentref.nreduc += parentref.toclone
        parentref.nwalkers += parentref.toclone
        if parenref.toclone < 0:
            for child in children:
                final_nreduc = child.nreduc + child.toclone
                if final_nreduc >= 1:
                    dif = min(parentf.toclone, final_nreduc)
                    parenref.toclone += dif
                    child.toclone -= dif
            if parenref.toclone < 0:
                tID = parenref.ID

        if parentref.toclone > 0:
            for child in childern:
                if child.nabovemin >= 1:
                    child_toclone += parenref.toclone
                    parentref.toclone = 0

            if parentref.toclone > 0:
                tID = parentref.ID
        maxwalk = None
        minwalk = None
        for child in childern:
            final_nwalkers = child.nwalkers + child.toclone
            final_nreduc = child.nreduc + child.toclone
            if (maxwalk or  final_nwalkers > $maxwalk) and (final_nreduc >= 1):
                maxwalk = final_nwalkers
                highchild = child

            if (minwalk or  final_nwalkers < minwalk) and (child.nabovemin >= 1):
                minwalk = final_nwalkers
                lowchild = child


        if (minwalk and  maxwalk):
            ntrans = min(int((maxwalk - minwalk)/2), highchild.nreduc + highchild.toclone)

        while (minwalk and defined $maxwalk) and  (ntrans >= 1):
            # merge walkers in highchild ; clone walkers in lowchild
            lowchild.toclone += ntrans
            highchild.toclone -= ntrans

            minwalk = None
            maxwalk = None
            for child in  children:
                final_nwalkers = child.nwalkers + child.toclone
                final_nreduc = child.nreduc + child.toclone

                if (maxwalk or final_nwalkers > maxwalk) and (final_nreduc >= 1):
                    maxwalk = final_nwalkers
                    highchild = child

                if (minwalk or final_nwalkers < minwalk) and (child.nabovemin >= 1):
                    minwalk = final_nwalkers
                    lowchild = child

            if (minwalk and maxwalk):
                ntrans = min(int((maxwalk - minwalk)/2), highchild.nreduc + highchild.toclone)

        # children are as balanced as they are going to get
        #my @submitted = map { 0 } @children;
        while (min(submitted) == 0):   # submit all children to balancetree (merges before clones)
            lowestchild = None
            lowclone = None
            for i in range(len(childern)):
                if (submitted[i] and (lowclone and children[i].toclone < lowclone)):
                        lowclone = children[i].toclone
                        lowestchild = i
                balancetree(children[lowestchild])
                # submitting merges first will free up space for clones
                submitted[lowestchild] = 1
    elif len(childern)==1:
        parenref.nreduc += parenref.toclone
        parentref.nwalkers += parentref.toclone

        children[0].toclone = parentref.toclone
        parentref.toclone = 0
        balancetree(children[0])
    else:
        if (parentref.toclone > parentref.nreduc):
            #my @tID = @{$parentref->{"ID"}};
            t1 = parentref.toclone
            t2 = parentref.nreduc
        if parenref.toclone < 0:
            while parentref.toclone < 0:
                # MERGE: find the two walkers with the lowest weights
                r1 = None
                minwt = None
                for i in range(parentref.nwalkers-1):
                    twt = walkerwt[parentref.windex[i]]
                    if r1 or twt < minwt:
                        minwt = twt
                        r1 = i
                r2 = None
                minwt = None

                for i in range(parentref.nwalkers):
                        if i != r1:
                            twt = walkerwt[parentref.windex[i]]
                        if r2 or twt < minwt):
                            minwt = twt;
                            r2 = i

                r1index = parentref.windex[r1]
                r2index = parentref.windex[r2]
                r3 = rand(walkerwt[r1index] + walkerwt[r2index])
                if r3 < walkerwt[r1index]:
                    walkerwt[r1index] += walkerwt[r2index]
                    walkerwt[r2index] = 0
                    walkerreg[r2index] = -3    # just so you don't accidentally use it
                    walkeroldreg[r2index] = -3    # just so you don't accidentally use it
                    freewalkers.append(r2index)
                    #splice(@{$parentref->{"windex"}},$r2,1);
                    parentref.nwalkers -=1
                else:
                    walkerwt[r2index] += walkerwt[r1index]
                    walkerwt[r1index] = 0
                    walkerreg[r1index] = -3
                    walkeroldreg[$r1index] = -3
                    freewalkers.append(r1index)
                    #splice(@{$parentref->{"windex"}},$r1,1);

                parenref.toclone +=1
        elif parentref.toclone >=0:
            while parentref.toclone > 0:
                # pick the one with the highest weight
                r1 = None
                maxwt = None
                for i in range(parentref.nwalkers):
                    twt = walkerwt[parentref.windex[i]]
                    if r1 or twt > maxwt:
                        maxwt = $twt;
                        r1 = i
                    r1walk = parentref.windex[r1]
                    #tind = pop(freewalkers)
                    parentref.windex.append($tind)
                    walkerwt[tind] = 0.5*walkerwt[r1walk]
                    walkerwt[r1walk] = 0.5*$walkerwt[r1walk]
                    walkerreg[tind] = walkerreg[r1walk]
                    walkeroldreg[tind] = walkeroldreg[r1walk];
                    parentref.nwalkers +=1
                    parentref.toclone -=1

def getdist(parentref, level, ind):
    newind = []
    children = parentref.children
    ochild = children
    if len(children) > 0:
        mindist , closereg = getclosest(0, children, mindist, closereg)
        if mindist > cellsize[level]:
            closereg = definenew(level, childern, parentref, closereg, mindist)

        ind.append(closereg)
        level += 1
        newind = getdist(parenref[closereg], tlevel, ind)
    else:
        # bottom level
        if level == len(cellsize):
            newind = ind
        else:
            mindist , closereg = getclosest(0, children, mindist, closereg)
            if mindist > cellsize[level]:
                closereg = definenew(level, childern, parentref, closereg, mindist)

            ind.append(closereg)
    return newind


def getregind(ID):
    ind = 0
    for i in range(0, len(ID)-1):
        prod = 1
        if i+1 <= len(ID)-1:
            for j in range(i+1, len(ID)-1):
                prod *= maximage[i]
        ind += ID[i] * prod
    return ind

def buildtree(parentref, level, ID):
    under = 0
    nreduc = 0
    nabovemin = 0
    if level < maxlevel:
        for i in range(maximage(level)-1):
            tchirdern = []
            twalkers = []
            tID = ID
            tID.append(i)
            node = Node(nwalkers=0, nreduc=0, nabovemin =0, tchildren=tchildren,
                        ID=tID, tclone=0, windx=twalkers)
            cwalk, creduc, cabovemin = buildtree(node, level+1,tID)
            if cwalk > 0:
                node('nwalkers') = cwalk
                node('nreduc') = creduc
                node('nabovemin') = cabovemin
                # push(@{$parentref->{"children"}},\%node);
            under += cwalk
            nreduc +=creduc
            nabovemin += cabovemin
    else:
        ind = getregind(ID)
        avgwt = 0
        wts = []
        for i in range(nwalk-1):
            wreg = wlakerreg[i]
            if wreg < 0 and useoldregifvoid):
                wreg = walkeroldreg[i]
            if wreg = ind:
                nunder +=1
                #push(@{$parentref->{"windex"}},$i);
                wts.append(walkerwt[i])
                if walkerwt[i] > pmin :
                    nabovemin += 1
        nreduc  = 0
        if (under>1):
            sortwts = wts.sort()
            sum = sortwts[0]
            for i in range(1: len(sortwts)-1):
                sum += sortwts[i]
                if sum < pmax:
                    nreduc = i
    return under, nreduc, nabovemin





class WExplore1Resampler(Resampler):

    DECISION = CloneMergeDecision
    INSTRUCTION_DTYPES = CLONE_MERGE_INSTRUCTION_DTYPES

    def __init__(self




    def _clone_merge(self, walkers, clone_merge_resampling_record, debug_prints):

        resampled_walkers = walkers.copy()
        # each stage in the resampling for that cycle
        for stage_idx, stage in enumerate(clone_merge_resampling_record):
            # the rest of the stages parents are based on the previous stage
            for parent_idx, resampling_record in enumerate(stage):
                # do merging
                if resampling_record.decision is CloneMergeDecision.SQUASH.value:

                    squash_walker = walkers[parent_idx]
                    merge_walker =  walkers[resampling_record.instruction[0]]
                    merged_walker = squash_walker.squash(merge_walker)
                    resampled_walkers[resampling_record.instruction[0]] = merged_walker

                elif resampling_record.decision  is CloneMergeDecision.CLONE.value:

                     clone_walker = walkers[resampling_record.instruction[0]]
                     cloned_walkers = clone_walker.clone()
                     resampled_walkers [parent_idx] = cloned_walkers.pop()
                     resampled_walkers [resampling_record.instruction[1]] = cloned_walkers.pop()

                elif resampling_record.decision in [CloneMergeDecision.KEEP_MERGE.value,
                                                    CloneMergeDecision.NOTHING.value] :
                    pass
                else:
                    # do nothing
                    pass
            walkers = resampled_walkers.copy()

        weights = []
        if debug_prints:
            n_walkers = len(resampled_walkers)
            result_template_str = "|".join(["{:^10}" for i in range(n_walkers+1)])
            walker_weight_str = result_template_str.format("weight",
                    *[str(walker.weight) for walker in resampled_walkers])
            for walker in resampled_walkers:
                weights.append(walker.weight)

            print ('walkers weight sume (end)=', np.array(weights).sum())

        return resampled_walkers

    def _clone_merge_even(self, walkers, clone_merge_resampling_record, debug_prints, wt, amp):

        resampled_walkers = walkers.copy()
        # each stage in the resampling for that cycle
        for stage_idx, stage in enumerate(clone_merge_resampling_record):
            # the rest of the stages parents are based on the previous stage
            for parent_idx, resampling_record in enumerate(stage):
                # do merging
                if resampling_record.decision is CloneMergeDecision.SQUASH.value:

                    squash_walker = walkers[parent_idx]
                    merge_walker =  walkers[resampling_record.instruction[0]]
                    merged_walker = squash_walker.squash(merge_walker)
                    resampled_walkers[resampling_record.instruction[0]] = merged_walker

                elif resampling_record.decision  is CloneMergeDecision.CLONE.value:

                     clone_walker = walkers[resampling_record.instruction[0]]
                     cloned_walkers = clone_walker.clone()
                     resampled_walkers [parent_idx] = cloned_walkers.pop()
                     resampled_walkers [resampling_record.instruction[1]] = cloned_walkers.pop()
                     # adjust weights of cloned walkers to equal orig_wt / n_clones
                     resampled_walkers[parent_idx].weight = wt[parent_idx]/amp[parent_idx]
                     resampled_walkers[resampling_record.instruction[1]].weight = wt[parent_idx]/amp[parent_idx]

                elif resampling_record.decision in [CloneMergeDecision.KEEP_MERGE.value, CloneMergeDecision.NOTHING.value] :
                    pass
                else:
                    # do nothing
                    pass
            walkers = resampled_walkers.copy()

        weights = []
        if debug_prints:
            n_walkers = len(resampled_walkers)
            result_template_str = "|".join(["{:^10}" for i in range(n_walkers+1)])
            walker_weight_str = result_template_str.format("weight",
                    *[str(walker.weight) for walker in resampled_walkers])
            for walker in resampled_walkers:
                weights.append(walker.weight)

            print ('walkers weight sume (end)=', np.array(weights).sum())

        return resampled_walkers

    def decide_clone_merge(self, n_walkers, walkerwt, amp, distance_matrix, debug_prints=False):

        spreads =[]
        resampling_actions = []
        new_wt = walkerwt.copy()
        new_amp = amp.copy()
        # initialize the actions to nothing, will be overwritten

        # calculate the initial spread which will be optimized
        spread, wsum = self._calcspread(n_walkers, walkerwt, new_amp, distance_matrix)
        spreads.append(spread)

        # maximize the variance through cloning and merging
        if debug_prints:
            print("Starting variance optimization:", spread)
        productive = True
        n_clone_merges = 0
        while productive:
            productive = False
            # find min and max wsums, alter new_amp
            minwind = None
            maxwind = None
            walker_actions = [ResamplingRecord(decision=CloneMergeDecision.NOTHING.value,
                                               instruction=NothingInstructionRecord(slot=i))
                              for i in range(n_walkers)]

            # selects a walker with minimum wsum and a walker with maximum wsum
            # walker with the highest wsum (distance to other walkers) will be tagged for cloning (stored in maxwind)
            max_tups = [(value, i) for i, value in enumerate(wsum)
                        if (new_amp[i] >= 1) and (new_wt[i]/(new_amp[i] + 1) > self.pmin)]
            if len(max_tups):
                maxvalue, maxwind = max(max_tups)

            # walker with the lowest wsum (distance to other walkers) will be tagged for merging (stored in minwind)
            min_tups = [(value, i) for i,value in enumerate(wsum)
                        if new_amp[i] == 1 and (new_wt[i]  < self.pmax)]
            if len(min_tups) > 0:
                minvalue, minwind = min(min_tups)

            # does minwind have an eligible merging partner?
            closedist = self.merge_dist
            closewalk = None
            condition_list = np.array([i is not None for i in [minwind,maxwind]])
            if condition_list.all() and minwind != maxwind:

                closewalk_availabe = set(range(n_walkers)).difference([minwind,maxwind])
                closewalk_availabe = [idx for idx in closewalk_availabe
                                      if new_amp[idx]==1 and (new_wt[idx] + new_wt[minwind] < self.pmax)]
                if len(closewalk_availabe) > 0:
                    tups = [(distance_matrix[minwind][i], i) for i in closewalk_availabe
                                            if distance_matrix[minwind][i] < (self.merge_dist)]
                    if len(tups) > 0:
                        closedist, closewalk = min(tups)


            # did we find a closewalk?
            condition_list = np.array([i is not None for i in [minwind,maxwind,closewalk]])
            if condition_list.all() :

                # change new_amp
                tempsum = new_wt[minwind] + new_wt[closewalk]
                new_amp[minwind] = new_wt[minwind]/tempsum
                new_amp[closewalk] = new_wt[closewalk]/tempsum
                new_amp[maxwind] += 1

                # re-determine spread function, and wsum values
                newspread, wsum = self._calcspread(n_walkers, new_wt, new_amp, distance_matrix)

                if newspread > spread:
                    spreads.append(newspread)
                    if debug_prints:
                        print("Variance move to", newspread, "accepted")

                    n_clone_merges += 1
                    productive = True
                    spread = newspread

                    # make a decision on which walker to keep (minwind, or closewalk)
                    r = rand.uniform(0.0, new_wt[closewalk] + new_wt[minwind])
                     # keeps closewalk and gets rid of minwind
                    if r < new_wt[closewalk]:
                        keep_idx = closewalk
                        squash_idx = minwind

                    # keep minwind, get rid of closewalk
                    else:
                        keep_idx = minwind
                        squash_idx = closewalk

                    # update weight
                    new_wt[keep_idx] += new_wt[squash_idx]
                    new_wt[squash_idx] = 0.0

                    # update new_amps
                    new_amp[squash_idx] = 0
                    new_amp[keep_idx] = 1

                    # recording the actions
                    walker_actions[squash_idx] = ResamplingRecord(
                                decision=CloneMergeDecision.SQUASH.value,
                                instruction=SquashInstructionRecord(merge_slot=keep_idx))
                    walker_actions[keep_idx] = ResamplingRecord(
                                decision=CloneMergeDecision.KEEP_MERGE.value,
                                instruction=SquashInstructionRecord(merge_slot=keep_idx))
                   # record  the clone instruction for keeping the the track of cloning
                    clone_idx = maxwind
                    walker_actions[clone_idx] = ResamplingRecord(
                        decision=CloneMergeDecision.CLONE.value,
                        instruction = CloneInstructionRecord(slot_a=clone_idx, slot_b=squash_idx))
                    resampling_actions.append(walker_actions)
                    # new spread for starting new stage
                    newspread, wsum = self._calcspread(n_walkers, new_wt, new_amp, distance_matrix)
                    spreads.append(newspread)
                    if debug_prints:
                        print("variance after selection:", newspread)

                # if not productive
                else:
                    new_amp[minwind] = 1
                    new_amp[closewalk] = 1
                    new_amp[maxwind] -= 1

        if n_clone_merges == 0:
            return([walker_actions]), spreads[-1], new_wt, new_amp
        else:
            return resampling_actions, spreads[-1], new_wt, new_amp

    def resample(self, walkers, debug_prints=False):

        n_walkers = len(walkers)
        walkerwt = [walker.weight for walker in walkers]
        amp = [1 for i in range(n_walkers)]

        # calculate distance matrix
        distance_matrix = self.distance_function.distance(walkers)

        if debug_prints:
            print("distance_matrix")
            print(distance_matrix)

        # determine cloning and merging actions to be performed, by maximizing the spread
        resampling_actions, spread, new_wt, new_amp = self.decide_clone_merge(n_walkers, walkerwt,
                                                             amp, distance_matrix,
                                                             debug_prints=debug_prints)

        # actually do the cloning and merging of the walkers
        resampled_walkers = self._clone_merge_even(walkers, resampling_actions, debug_prints, new_wt, new_amp)

        data = {'distance_matrix' : distance_matrix, 'spread' : np.array([spread]) }

        return resampled_walkers, resampling_actions, data
