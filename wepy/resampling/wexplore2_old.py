
import multiprocessing as mulproc
import random as rand
from itertools import combinations
from functools import partial

import numpy as np
import mdtraj as mdj

from wepy.walker import merge
from wepy.resampling.clone_merge import CloneMergeDecision
from wepy.resampling.resampler import Resampler, ResamplingRecord
from wepy.resampling.clone_merge import NothingInstructionRecord ,CloneInstructionRecord ,SquashInstructionRecord 
from wepy.resampling.clone_merge import KeepMergeInstructionRecord ,CLONE_MERGE_DECISION_INSTRUCTION_MAP 
                                     
                                     
class WExplore2Resampler(Resampler):
    
    def __init__(self,reference_traj, seed=None, pmin=1e-12, pmax=0.1, dpower=4, merge_dist=0.25):
        self.ref = reference_traj
        self.pmin=pmin
        self.lpmin = np.log(pmin/100)        
        self.pmax=pmax
        self.dpower = dpower
        self.merge_dist = merge_dist
        self.seed = seed
        if seed is not None:
            rand.seed(seed)
        
        
    def _rmsd(self, traj, ref, idx):
        return np.sqrt(3*np.sum(np.square(traj.xyz[:, idx, :] - ref.xyz[:, idx, :]),
                                axis=(1, 2))/idx.shape[0])


    def _maketraj(self, positions):
        Newxyz = np.zeros((1, self.ref.n_atoms, 3))
        
        for i in range(len(positions)):
            Newxyz[0,i,:] = ([positions[i]._value[0], positions[i]._value[1],
                                                        positions[i]._value[2]])
        return mdj.Trajectory(Newxyz, self.ref.topology)
        
    def selection(self,):
        self.ref =self.ref.remove_solvent()
        lig_idx = self.ref.topology.select('resname "2RV"')
        b_selection = mdj.compute_neighbors(self.ref, 0.8, lig_idx)
        b_selection = np.delete(b_selection, lig_idx)
        return lig_idx, b_selection 
        
    def calculate_rmsd(self, lig_idx, b_selection, positions_a, positions_b):
        ref_traj = self._maketraj(positions_a[0:self.ref.n_atoms])
        traj = self._maketraj(positions_b[0:self.ref.n_atoms])
        traj = traj.superpose(ref_traj, atom_indices=b_selection)
        return  self._rmsd(traj, ref_traj, lig_idx)
    
    def _calcspread(self, n_walkers, walkerwt, amp, distance_matrix):
        spread = 0
        wsum = np.zeros(n_walkers)
        wtfac = np.zeros(n_walkers)
        for i in range(n_walkers):
            if walkerwt[i] > 0 and amp[i] > 0:
                wtfac[i] = np.log(walkerwt[i]/amp[i]) - self.lpmin
            else:
                wtfac[i] = 0
            if wtfac[i] < 0:
                wtfac[i] = 0

        for i in range(n_walkers - 1):
            if amp[i] > 0:
                for j in range(i+1, n_walkers):
                    if amp[j] > 0:
                        d = (distance_matrix[i][j] ** self.dpower) * wtfac[i] * wtfac[j];
                        spread += d * amp[i] * amp[j];
                        wsum[i] += d * amp[j];
                        wsum[j] += d * amp[i];

        return spread, wsum


    def _clone_merge(self, walkers, clone_merge_resampling_record,debug_prints):
        
        resampled_walkers = []
        
        # each stage in the resampling for that cycle
        for stage_idx, stage in enumerate(clone_merge_resampling_record):
            # the rest of the stages parents are based on the previous stage
            for parent_idx, resampling_record in enumerate(stage):
                # do merging
                if resampling_record.decision is CloneMergeDecision.NOTHING:
                    child_idx = parent_idx
                    child_walker = walkers[child_idx]
                    resampled_walkers.append(child_walker)

                elif resampling_record.decision is CloneMergeDecision.KEEP_MERGE:

                    squash_walker = walkers[resampling_record.value.slot]
                    merge_walker = walkers[parent_idx] 
                    merged_walker = squash_walker.squash(merge_walker)
                    
                    resampled_walkers.append(merged_walker)
                         
                elif resampling_record.decision  is CloneMergeDecision.CLONE:
                     clone_walker = walkers[parent_idx]
                     cloned_walkers = clone_walker.clone(len(resampling_records.value))
                     resampled_walkers.extend(cloned_walkers)          
                else:
                    # do nothing 
                    pass
        n_walkers = len (resampled_walkers)
        
        if debug_prints:
            result_template_str = "|".join(["{:^10}" for i in range(n_walkers+1)])
            walker_weight_str = result_template_str.format("weight",
                    *[str(walker.weight) for walker in resampled_walkers])
            print(walker_weight_str)
                

        return resampled_walkers 
                
 
    def decide_clone_merge(self, n_walkers, walkerwt, amp, distance_matrix, debug_prints=False):

        resampling_actions = []
        # initialize the actions to nothing, will be overwritten
        walker_actions = [ResamplingRecord(decision=CloneMergeDecision.NOTHING,
                                               value=NothingInstructionRecord(slot=i))
                              for i in range(n_walkers)]

        # calculate the initial spread which will be optimized
        spread, wsum = self._calcspread(n_walkers, walkerwt, amp, distance_matrix)

        # maximize the variance through cloning and merging
        if debug_prints:
            print("Starting variance optimization:", spread)
        productive = True        
        while productive:
            productive = False
            # find min and max wsums, alter amp
            maxwsum = -1
            minwsum = float('inf')
            minwind = None
            maxwind = None
            for i in range(n_walkers):
                # determine eligibility and find highest wsum (to clone) and lowest wsum (to merge)
                if amp[i] >= 1 and walkerwt[i] > self.pmin:  # find the most clonable walker
                    if wsum[i] > maxwsum:
                        maxwsum = wsum[i]
                        maxwind = i
                if amp[i] == 1 and  walkerwt[i] < self.pmax:
                    # find the most mergeable walker
                    if wsum[i] < minwsum:
                        minwsum = wsum[i]
                        minwind = i
 
            # does minwind have an eligible merging partner?
            closedist = self.merge_dist
            closewalk = None
            if minwind is not None and maxwind is not None and minwind != maxwind:
                for j in range( n_walkers):
                    if j != minwind and j != maxwind:
                        if (distance_matrix[minwind][j] < closedist and
                                                (amp[j] == 1) and
                                                (walkerwt[j] < self.pmax)):
                            closedist = distance_matrix[minwind][j]
                            closewalk = j

            # did we find a closewalk?
            if minwind is not None and maxwind is not None and closewalk is not None:
                if amp[minwind] != 1:
                    die("Error! minwind", minwind, "has self.amp =", amp[minwind])
                if amp[closewalk] != 1:
                    die("Error! closewalk", closewalk, "has self.amp=", amp[closewalk])

                # change self.amp
                tempsum = walkerwt[minwind] + walkerwt[closewalk]
                amp[minwind] = walkerwt[minwind]/tempsum
                amp[closewalk] = walkerwt[closewalk]/tempsum
                amp[maxwind] += 1

                # re-determine spread function, and wsum values
                newspread, wsum = self._calcspread(n_walkers, walkerwt, amp, distance_matrix)
                
                if newspread > spread:
                    #if debug_prints:
                        #print("Variance move to", newspread, "accepted")
                      
                    print ('w_minwind[{}]={}, w_closewalk[{}]={}'.format(minwind, walkerwt[minwind],
                                                                       closewalk, walkerwt[closewalk]))  
                    productive = True
                    spread = newspread
                    
                    # make a decision on which walker to keep (minwind, or closewalk)
                    # keeps closewalk and gets rid of minwind
                    r = rand.uniform(0.0, walkerwt[closewalk] + walkerwt[minwind])
                    if r < walkerwt[closewalk]:

                        # merge the weights
                        walkerwt[closewalk] += walkerwt[minwind]
                        walkerwt[minwind] = 0.0

                        # update amps
                        amp[minwind] = 0
                        amp[closewalk] = 1

                        # recording the actions
                        walker_actions[minwind] = ResamplingRecord(
                                    decision=CloneMergeDecision.SQUASH,
                                    value=SquashInstructionRecord(merge_slot=closewalk))
                        walker_actions[closewalk] = ResamplingRecord(
                                    decision=CloneMergeDecision.KEEP_MERGE,
                                    value=SquashInstructionRecord(merge_slot=closewalk))
                        keep_idx = closewalk
                        squash_idx = minwind 
                     
                    else:
                        # keep minwind, get rid of closewalk
                        # merge the weights
                        
                        walkerwt[minwind] += walkerwt[closewalk]
                        walkerwt[closewalk] = 0.0

                        # update amps
                        amp[closewalk] = 0
                        amp[minwind] = 1

                        # keep track of the actions
                        walker_actions[closewalk] = ResamplingRecord(
                                    decision=CloneMergeDecision.SQUASH,
                                    value=SquashInstructionRecord(merge_slot=minwind))
                        walker_actions[minwind] = ResamplingRecord(
                                    decision=CloneMergeDecision.KEEP_MERGE,
                                    value=SquashInstructionRecord(merge_slot=minwind))
                        keep_idx = minwind
                        squash_idx = closewalk
                    
                    newspread, wsum = self._calcspread(n_walkers, walkerwt, amp, distance_matrix)
                    if debug_prints:
                    #    print("variance after selection:", newspread)
                        print ('minwind= {}, closewalk= {} , maxwind ={}'.format(minwind,closewalk,maxwind))
                        
                        print ('squash_idx ={}, keep_idx ={} ,maxwind ={}\n'.format(squash_idx,keep_idx,maxwind))

                        
                        
                        



                # if not productive
                else:
                    amp[minwind] = 1
                    amp[closewalk] = 1
                    amp[maxwind] -= 1
        #squash_idx = rand.choice([walker_idx for walker_idx in squash_available])
        #keep_walker = rand.choices(walkers, weights=weights)
        # index of the kept walker
        # keep_idx = walkers.index(keep_walker)

        # end of while productive loop:  done with cloning and merging steps
        # set walkerdist arrays
        walkerdist = wsum

        # Perform the cloning and merging steps
        # start with an empty free walker array, add all walkers with amp = 0
        freewalkers = [i for i, value in enumerate(amp) if value == 0]
        cloneable = [i for i, value in enumerate(amp) if value > 1]
        #print(freewalkers)
        #print(cloneable)
        
        # for each self.amp[i] > 1, clone!
        for r1walk  in range(n_walkers):
            if amp[r1walk] > 1:
                nclone = amp[r1walk]-1
                inds = []
                for i in range(nclone):
                    try:
                        tind = freewalkers.pop()
                    except:
                        raise("Error! Not enough free walkers!")
                    if r1walk == tind:
                        raise("Error!  free walker is equal to the clone!")
                    else:
                        inds.append(tind)
                

                newwt = walkerwt[r1walk]/(nclone+1)
                walkerwt[r1walk] = newwt
                for tind in inds:
                    walkerwt[tind] = newwt
                    walkerdist[tind] = walkerdist[r1walk]
                       # done cloning and meging
                # add r1walka index just for keeping tracks of cloning
                 
                walker_actions[r1walk] = ResamplingRecord(
                                    decision=CloneMergeDecision.CLONE,
                                    value=tuple(inds))
        resampling_actions.append(walker_actions)

        if debug_prints:
            result_template_str = "|".join(["{:^10}" for i in range(n_walkers+1)])
 
            print("---------------------")


            # walker slot indices
            slot_str = result_template_str.format("slot", *[i for i in range(n_walkers)])
            print(slot_str)

            # the resampling actions
            action_str = result_template_str.format("decision",
                *[str(tup[0].name) for tup in walker_actions])
            print(action_str)
            data_str = result_template_str.format("instruct",
                *[','.join([str(i) for i in tup[1]]) for tup in walker_actions])
            print(data_str)
               
             
        if len(freewalkers) > 0:
            #raise("Error! walkers left over after merging and cloning")
            print(len(freewalkers))
            return []
        else:
            # return the final state of the resampled walkers after all
            # stages, and the records of resampling
            return  resampling_actions

                
    def resample(self, walkers, debug_prints=False):
        
        calc_list = []
        print ("Starting resampling") 
        
        walkers = walkers
        n_walkers = len(walkers)
        print ( n_walkers)
        walkerwt = [walker.weight for walker in walkers]
        amp = [1 for i in range(n_walkers)]
        
        
        # parallel  function is computing distance_matrix 
        
            
        # calc_list = list(combinations([i for i in range(n_walkers)], 2))
        # positions_lsit = [(walkers[index[0]].positions, walkers[index[1]].positions)
        #                    for index in calc_list]
                          
        # pool = mulproc.Pool(mulproc.cpu_count())
        # lig_idx, b_selection = self.selection()
        # function = partial(self.calculate_rmsd, lig_idx, b_selection)
        # results = pool.map(function, positions_lsit)
        # if debug_prints:
        #     print (results)
        # pool.close()

        # # associate pairs to distances
        # indexitem = zip(calc_list , results)

        # # make the distance matrix
        # distance_matrix = np.zeros((n_walkers, n_walkers))        
        # for index in indexitem:        
        #     distance_matrix[index[0][0]][index[0][1]] = index[1]
        #     distance_matrix[index[0][1]][index[0][0]] = index[1]

        distance_matrix = np.zeros((n_walkers,n_walkers))
        lig_idx, b_selection = self.selection()
        for i in range(n_walkers):
            for j in range(i+1, n_walkers):
                d = self.calculate_rmsd (lig_idx, b_selection, walkers[i].positions, walkers[j].positions)
                distance_matrix[i][j] = d
                distance_matrix [j][i] = d                   
                                   
        if debug_prints:
            print ("distance_matrix")
            print (distance_matrix)

        # determine cloning and merging actions to be performed, by maximizing the spread
        resampling_actions = self.decide_clone_merge(n_walkers, walkerwt, amp, distance_matrix, debug_prints=debug_prints)

        if debug_prints:
            print ("Ending resampling")

        # actually do the cloning and merging of the walkers
        resampled_walkers = self._clone_merge(walkers, resampling_actions,debug_prints)
        if len (walkers) != len (resampled_walkers):
            print ("Error")
        
        return resampled_walkers, resampling_actions 
