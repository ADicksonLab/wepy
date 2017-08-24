
import multiprocessing as mulproc
import random as rand
from itertools import combinations
from functools import partial

import numpy as np
import numpy.linalg as la

import mdtraj as mdj
import itertools as it



from wepy.walker import merge
from wepy.resampling.clone_merge import CloneMergeDecision, clone_parent_panel
from wepy.resampling.resampler import Resampler, ResamplingRecord
from wepy.resampling.clone_merge import NothingInstructionRecord, CloneInstructionRecord, SquashInstructionRecord 
from wepy.resampling.clone_merge import KeepMergeInstructionRecord, CLONE_MERGE_DECISION_INSTRUCTION_MAP 
from wepy.boundary import BoundaryConditions





class UnbindingBC(BoundaryConditions):
    def __init__(self, initial_state=None, cutoff_distance=1.0, topology=None, ligand_idxs=None, binding_site_idxs=None):
        assert initial_state is not None, "Must give an initial state"
        assert topology is not None, "Must give a reference topology"
        assert ligand_idxs is not None
        assert binding_site_idxs is not None
        assert type(cutoff_distance) is float

        self.initial_state = initial_state
        self.cutoff_distance = cutoff_distance
        self.topology = topology

        self.ligand_idxs = ligand_idxs
        self.binding_site_idxs = binding_site_idxs
    
    def in_boundary(self, walker):

        # calc box length
        cell_lengths = np.array([[self.calc_length(v._value) for v in walker.box_vectors]])
        
        
        # TODO order of cell angles
        # calc angles
        
        cell_angles = np.array([[self.calc_angle(walker.box_vectors._value[i], walker.box_vectors._value[j])
                                 for i, j in [(0,1), (1,2), (2,0)]]])



        

        
        # make a traj out of it so we can calculate distances through the periodic boundary conditions
        walker_traj = mdj.Trajectory(self.to_mdtraj(walker.positions[0:self.topology.n_atoms]), topology=self.topology,
                                     unitcell_lengths=cell_lengths, unitcell_angles=cell_angles) 

        # calculate the distances through periodic boundary conditions
        # and get hte minimum distance
        min_distance = np.min(mdj.compute_distances(walker_traj,
                                                    it.product(self.ligand_idxs, self.binding_site_idxs)))

        # test to see if the ligand is unbound
        if min_distance >= self.cutoff_distance:
            return True
        else:
            return False
        
    def calc_angle(self, v1, v2):
        return np.degrees(np.arccos(np.dot(v1, v2)/(la.norm(v1) * la.norm(v2))))

    def calc_length(self, v):
        return la.norm(v)
    
    
    def to_mdtraj(self, positions):
        n_atoms = self.topology.n_atoms 
        
        xyz = np.zeros((1, n_atoms, 3))
        
        for i in range(n_atoms):
            xyz[0,i,:] = ([positions[i]._value[0], positions[i]._value[1],
                                                        positions[i]._value[2]])
        return xyz

    def warp_walkers(self, walkers):

        new_walkers = []
        warped_walkers_idxs = []
        
        for walker_idx, walker in enumerate(walkers):
           if self.in_boundary(walker):
               warped_walkers_idxs.append(walker_idx)
               new_walkers.append(self.initial_state)
           else:
               new_walkers.append(walker)
               
        return new_walkers, warped_walkers_idxs

  
        
class WExplore2Resampler(Resampler):
    
    def __init__(self, seed=None, pmin=1e-12, pmax=0.1, dpower=4, merge_dist=0.25,
                 topology=None, ligand_idxs=None, binding_site_idxs=None):
        self.pmin=pmin
        self.lpmin = np.log(pmin/100)        
        self.pmax=pmax
        self.dpower = dpower
        self.merge_dist = merge_dist
        self.seed = seed
        if seed is not None:
            rand.seed(seed)
        self.topology = topology
        self.ligand_idxs = ligand_idxs
        self.binding_site_idxs = binding_site_idxs
        
    def rmsd(self, traj, ref, idx):
        return np.sqrt(3*np.sum(np.square(traj.xyz[:, idx, :] - ref.xyz[:, idx, :]),
                                axis=(1, 2))/idx.shape[0])


    def maketraj(self, positions):
        n_atoms = self.topology.n_atoms
        xyz = np.zeros((1, n_atoms, 3))
        
        for i in range(n_atoms):
            xyz[0,i,:] = ([positions[i]._value[0], positions[i]._value[1],
                                                        positions[i]._value[2]])
        return mdj.Trajectory(xyz, self.topology)
        
        
    def calculate_rmsd(self, positions_a, positions_b):
        traj_a = self.maketraj(positions_a)
        traj_b = self.maketraj(positions_b)
        traj_b = traj_b.superpose(traj_a, atom_indices=self.binding_site_idxs)
        return  self.rmsd(traj_a, traj_b, self.ligand_idxs)
    
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
        
        resampled_walkers = walkers.copy()
        # each stage in the resampling for that cycle
        for stage_idx, stage in enumerate(clone_merge_resampling_record):
            # the rest of the stages parents are based on the previous stage
            for parent_idx, resampling_record in enumerate(stage):
                # do merging                
                if resampling_record.decision is CloneMergeDecision.SQUASH:
                    
                    squash_walker = walkers[parent_idx]
                    merge_walker =  walkers[resampling_record.value[0]]
                    merged_walker = squash_walker.squash(merge_walker)
                    resampled_walkers[resampling_record.value[0]] = merged_walker
                  
                elif resampling_record.decision  is CloneMergeDecision.CLONE:
                    
                     clone_walker = walkers[resampling_record.value[0]]
                     cloned_walkers = clone_walker.clone()
                     resampled_walkers [parent_idx] = cloned_walkers.pop()
                     resampled_walkers [resampling_record.value[1]] = cloned_walkers.pop()
                     
                elif resampling_record.decision in [CloneMergeDecision.KEEP_MERGE, CloneMergeDecision.NOTHING] :
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
            print(walker_weight_str)
            for walker in resampled_walkers:
                weights.append(walker.weight)
            
                    
        print ('walkers weight sume (end)=', np.array(weights).sum())        
        return resampled_walkers 
                
 
    def decide_clone_merge(self, n_walkers, walkerwt, amp, distance_matrix, debug_prints=False):

        spreads =[] 
        resampling_actions = []
        # initialize the actions to nothing, will be overwritten
        
        # calculate the initial spread which will be optimized
        spread, wsum = self._calcspread(n_walkers, walkerwt, amp, distance_matrix)
        spreads.append(spread)

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
                spreads.append(newspread)

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
                    spreads.append(newspread) 
                    if debug_prints:
                        print("variance after selection:", newspread)
                          
                          
                          
                          
                          

                # if not productive
                else:
                    amp[minwind] = 1
                    amp[closewalk] = 1
                    amp[maxwind] -= 1
                    
        # if debug_prints:
        #     self.print_actions(n_walkers, resampling_actions)
        # return the final state of the resampled walkers after all
        if n_clone_merges == 0:
            return[], spreads
        else:
            return  resampling_actions, spreads

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
        
        n_walkers = len(walkers)
        walkerwt = [walker.weight for walker in walkers]
        amp = [1 for i in range(n_walkers)]

        # calculate distance matrix
        distance_matrix = np.zeros((n_walkers,n_walkers))
        
        for i in range(n_walkers):
            for j in range(i+1, n_walkers):
                d = self.calculate_rmsd (walkers[i].positions[0:self.topology.n_atoms], walkers[j].positions[0:self.topology.n_atoms])
                distance_matrix[i][j] = d
                distance_matrix [j][i] = d                   
                                   
        if debug_prints:
            print ("distance_matrix")
            print (distance_matrix)

        # determine cloning and merging actions to be performed, by maximizing the spread
        resampling_actions, spreads = self.decide_clone_merge(n_walkers, walkerwt, amp, distance_matrix, debug_prints=debug_prints)

        
        # actually do the cloning and merging of the walkers
        resampled_walkers = self._clone_merge(walkers, resampling_actions,debug_prints)
        
        return resampled_walkers, resampling_actions, distance_matrix, spreads 
