
import multiprocessing as mulproc
import random as rand
from itertools import combinations
from functools import partial

import numpy as np
import mdtraj as mdj

from wepy.walker import merge
from wepy.resampling.clone_merge import CloneMergeDecision, clone_parent_panel
from wepy.resampling.resampler import Resampler, ResamplingRecord
from wepy.resampling.clone_merge import NothingInstructionRecord, CloneInstructionRecord, SquashInstructionRecord 
from wepy.resampling.clone_merge import KeepMergeInstructionRecord, CLONE_MERGE_DECISION_INSTRUCTION_MAP 
                                     

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


    def _clone_merge(self, walkers, clone_merge_resampling_record, debug_prints):
         
        
        resampled_walkers = walkers
        # each stage in the resampling for that cycle
        for stage_idx, stage in enumerate(clone_merge_resampling_record):
            # the rest of the stages parents are based on the previous stage
            for parent_idx, resampling_record in enumerate(stage):
                # do merging                
                if resampling_record.decision is CloneMergeDecision.KEEP_MERGE:
                    
                    squash_walker = walkers[resampling_record.value[0]]
                    merge_walker = walkers[parent_idx] 
                    merged_walker = squash_walker.squash(merge_walker)
                    resampled_walkers[parent_idx] = merged_walker
                    
                elif resampling_record.decision  is CloneMergeDecision.CLONE:
                    
                     clone_walker = walkers[resampling_record.value[0]]
                     cloned_walkers = clone_walker.clone()
                     resampled_walkers [parent_idx] = cloned_walkers.pop()
                     resampled_walkers [resampling_record.value[1]] = cloned_walkers.pop()
                     
                elif resampling_record.decision in [CloneMergeDecision.SQUASH, CloneMergeDecision.NOTHING] :
                    pass
                else:
                    # do nothing 
                    pass
                walkers = resampled_walkers        
        
        if debug_prints:
            n_walkers = len(resampled_walkers)
            result_template_str = "|".join(["{:^10}" for i in range(n_walkers+1)])
            walker_weight_str = result_template_str.format("weight",
                    *[str(walker.weight) for walker in resampled_walkers])
            print(walker_weight_str)
                    
                
        return resampled_walkers 
                
 
    def decide_clone_merge(self, n_walkers, walkerwt, amp, distance_matrix, debug_prints=False):

        resampling_actions = []
        # initialize the actions to nothing, will be overwritten
        
        # calculate the initial spread which will be optimized
        spread, wsum = self._calcspread(n_walkers, walkerwt, amp, distance_matrix)

        # maximize the variance through cloning and merging
        if debug_prints:
            print("Starting variance optimization:", spread)
        productive = True
        n_clone_merges = 0
        while productive:
            productive = False
            # find min and max wsums, alter amp
            minwind = None
            maxwind = None
            walker_actions = [ResamplingRecord(decision=CloneMergeDecision.NOTHING,
                                               value=NothingInstructionRecord(slot=i))
                              for i in range(n_walkers)]

            # selects a walker with minimum wsum and a walker with maximum wsum
            try:
                maxvalue, maxwind = max((value, i) for i,value in enumerate(wsum)
                                        if amp[i] >= 1 and (walkerwt[i] > self.pmin))
            except: pass    
            try:
                minvalue, minwind = min((value, i) for i,value in enumerate(wsum)                       
                                        if amp[i] == 1 and (walkerwt[i]  < self.pmax))
            except: pass

            # does minwind have an eligible merging partner?
            closedist = self.merge_dist
            closewalk = None
            condition_list = np.array([i is not None for i in [minwind,maxwind]])
            if condition_list.all() and minwind != maxwind:
                
                closewalk_availabe = set(range(n_walkers)).difference([minwind,maxwind])
                closewalk_availabe = [idx for idx in closewalk_availabe
                                      if amp[idx]==1 and (walkerwt[idx] < self.pmax)]
                if len(closewalk_availabe) > 0:
                    try:
                        closedist, closewalk = min((distance_matrix[minwind][i], i) for i in closewalk_availabe
                                                if distance_matrix[minwind][i] < (self.merge_dist))
                    except: pass

             
            # did we find a closewalk?
            condition_list = np.array([i is not None for i in [minwind,maxwind,closewalk]])
            if condition_list.all() :
               #print ("check_list", closewalk) 
            #if minwind is not None and maxwind is not None and closewalk is not None:
                
                # change amp
                tempsum = walkerwt[minwind] + walkerwt[closewalk]
                amp[minwind] = walkerwt[minwind]/tempsum
                amp[closewalk] = walkerwt[closewalk]/tempsum
                amp[maxwind] += 1

                # re-determine spread function, and wsum values
                newspread, wsum = self._calcspread(n_walkers, walkerwt, amp, distance_matrix)

                if newspread > spread:
                    if debug_prints:
                        print("Variance move to", newspread, "accepted")
                      
                        
                    # print ('w_minwind[{}]={}, w_closewalk [{}]={}'.format(minwind,walkerwt[minwind],
                    #                                                    closewalk, walkerwt[closewalk]))                       
         
                    n_clone_merges += 1    
                    productive = True
                    spread = newspread                    
                    # make a decision on which walker to keep (minwind, or closewalk)
                    
                    squash_available = [minwind, closewalk]
                    
                    weights = [walkerwt[walker] for walker in squash_available]
                    
                    # index of squashed walker
                    keep_idx  = rand.choices(squash_available, weights=weights).pop()
                    
                    # index of the kept walker

                    
                    squash_idx= set(squash_available).difference({keep_idx}).pop()
                    # update weigh
                    walkerwt[keep_idx] += walkerwt[squash_idx]
                    walkerwt[squash_idx] = 0.0

                    # update amps
                    amp[squash_idx] = 0
                    amp[keep_idx] = 1

                    # recording the actions
                    walker_actions[squash_idx] = ResamplingRecord(
                                decision=CloneMergeDecision.SQUASH,
                                value=SquashInstructionRecord(merge_slot=keep_idx))
                    walker_actions[keep_idx] = ResamplingRecord(
                                decision=CloneMergeDecision.KEEP_MERGE,
                                value=SquashInstructionRecord(merge_slot=keep_idx))
                   # record  the clone instruction for keeping the the track of cloning
                    clone_idx = maxwind
                    walker_actions[clone_idx] = ResamplingRecord(
                        decision=CloneMergeDecision.CLONE,
                        value = CloneInstructionRecord(slot_a=clone_idx, slot_b=squash_idx))
                    resampling_actions.append(walker_actions)
                    # new spread for satrting new stage
                    newspread, wsum = self._calcspread(n_walkers, walkerwt, amp, distance_matrix)

                    if debug_prints:
                        print("variance after selection:", newspread)
                          #print ('minwind= {}, closewalk= {} , maxwind ={}'.format(minwind,closewalk,maxwind))
                          #print ('squash_idx ={}, keep_idx ={} ,maxwind ={}\n'.format(squash_idx,keep_idx,maxwind))
                          
                          
                          

                # if not productive
                else:
                    amp[minwind] = 1
                    amp[closewalk] = 1
                    amp[maxwind] -= 1
                    
        if debug_prints:
            self.print_actions(n_walkers, resampling_actions)
        # return the final state of the resampled walkers after all
        if n_clone_merges == 0:
            return[]
        else:
            return  resampling_actions

    def print_actions(self, n_walkers, clone_merge_resampling_record):
        

        result_template_str = "|".join(["{:^10}" for i in range(n_walkers+1)])

        print("---------------------")

        # walker slot indices
        slot_str = result_template_str.format("slot", *[i for i in range(n_walkers)])
        print(slot_str)
        
        for stage_idx, walker_actions in enumerate(clone_merge_resampling_record):
            print("Resampling Stage: {}".format(stage_idx))
            print("---------------------")

            # the resampling actions
            action_str = result_template_str.format("decision",
                                                    *[str(tup[0].name) for tup in walker_actions])
            print(action_str)
            data_str = result_template_str.format("instruct",
                                                  *[','.join([str(i) for i in tup[1]]) for tup in walker_actions])
            print(data_str)

    def resample(self, walkers, debug_prints=False):
        
        
        print ("Starting resampling") 
        
        walkers = walkers
        n_walkers = len(walkers)
        walkerwt = [walker.weight for walker in walkers]
        amp = [1 for i in range(n_walkers)]

        # calculate distance matrix
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
            print ("Error in length")
        
        return resampled_walkers, resampling_actions 
