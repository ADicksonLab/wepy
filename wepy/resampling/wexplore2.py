import multiprocessing as mulproc
import random as rand
import itertools as it

import numpy as np

from wepy.resampling.resamplers.resampler import Resampler
from wepy.resampling.decisions.clone_merge import CloneMergeDecision

class WExplore2Resampler(Resampler):

    DECISION = CloneMergeDecision

    def __init__(self, seed=None, pmin=1e-12, pmax=0.1, dpower=4, merge_dist=2.5,
                 distance_function=None):
        self.pmin=pmin
        self.lpmin = np.log(pmin/100)
        self.pmax=pmax
        self.dpower = dpower
        self.merge_dist = merge_dist
        self.seed = seed

        # ARD: test here if it is suitable?
        self.distance_function = distance_function

        if seed is not None:
            rand.seed(seed)

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
                        d = ((distance_matrix[i][j])**self.dpower) * wtfac[i] * wtfac[j];
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
                if resampling_record.decision is self.DECISION.ENUM.SQUASH.value:

                    squash_walker = walkers[parent_idx]
                    merge_walker =  walkers[resampling_record.instruction[0]]
                    merged_walker = squash_walker.squash(merge_walker)
                    resampled_walkers[resampling_record.instruction[0]] = merged_walker

                elif resampling_record.decision  is self.DECISION.ENUM.CLONE.value:

                     clone_walker = walkers[resampling_record.instruction[0]]
                     cloned_walkers = clone_walker.clone()
                     resampled_walkers [parent_idx] = cloned_walkers.pop()
                     resampled_walkers [resampling_record.instruction[1]] = cloned_walkers.pop()

                elif resampling_record.decision in [self.DECISION.ENUM.KEEP_MERGE.value,
                                                    self.DECISION.ENUM.NOTHING.value] :
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
        walkers_squashed = [[] for i in range(n_walkers)]
        num_clones = [0 for i in range(n_walkers)]
        
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
        while productive:
            productive = False
            # find min and max wsums, alter new_amp
            minwind = None
            maxwind = None

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

                closewalk_available = set(range(n_walkers)).difference([minwind,maxwind])
                closewalk_available = [idx for idx in closewalk_available
                                      if new_amp[idx]==1 and (new_wt[idx] + new_wt[minwind] < self.pmax)]
                if len(closewalk_available) > 0:
                    tups = [(distance_matrix[minwind][i], i) for i in closewalk_available
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

                    walkers_squashed[keep_idx] += squash_idx + walkers_squashed[squash_idx]
                    walkers_squashed[squash_idx] = []
                    num_clones[clone_idx]++

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

        walker_actions = [ResamplingRecord(decision=CloneMergeDecision.NOTHING.value,
                        instruction=NothingInstructionRecord(slot=i)) for i in range(n_walkers)]

        free_walkers = []
        for w in range(n_walkers):
            if num_clones[w] > 0 and len(walkers_squashed[w]) > 0:
                Raise("Error! cloning and merging occuring with the same walker")

            # add walkers squashed by w onto the free_walkers list
            if len(walkers_squashed[w]) > 0:
                free_walkers += walkers_squashed[w]
                for squash_idx in walkers_squashed[w]:
                    walker_actions[squash_idx] = ResamplingRecord(
		        decision=CloneMergeDecision.SQUASH.value,
		        instruction=SquashInstructionRecord(merge_slot=w))
                walker_actions[w] = ResamplingRecord(
		    decision=CloneMergeDecision.KEEP_MERGE.value,
		    instruction=SquashInstructionRecord(merge_slot=w))
            if num_clones[w] > 0:
                slots = []
                for i in range(num_clones[w]):
                    slots.append(free_walkers.pop())
                walker_actions[w] = ResamplingRecord(
                    decision=CloneMergeDecision.CLONE.value,
                    instruction = CloneInstructionRecord(slots=tuple(slots)))
            
        return([walker_actions]), spreads[-1]

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
        resampling_actions, spread = self.decide_clone_merge(n_walkers, walkerwt,
                                                             amp, distance_matrix,
                                                             debug_prints=debug_prints)

        # actually do the cloning and merging of the walkers
        resampled_walkers = CloneMergeDecision.action(walkers, resampling_actions)
        
        data = {'distance_matrix' : distance_matrix, 'spread' : np.array([spread]) }

        return resampled_walkers, resampling_actions, data

class OpenMMDistance(object):
    def __init__(self, topology= None, ligand_idxs=None, binding_site_idxs=None):
        self.topology = topology
        self.ligand_idxs = ligand_idxs
        self.binding_site_idxs = binding_site_idxs


    def _calc_angle(self, v1, v2):
        return np.degrees(np.arccos(np.dot(v1, v2)/(la.norm(v1) * la.norm(v2))))

    def _calc_length(self, v):
        return la.norm(v)

    def _pos_to_array(self, positions):
        n_atoms = self.topology.n_atoms

        xyz = np.zeros((1, n_atoms, 3))

        for i in range(n_atoms):
            xyz[0,i,:] = ([positions[i]._value[0],
                           positions[i]._value[1],
                           positions[i]._value[2]])
        return xyz

    def rmsd(self, traj, ref, idx):
        return np.sqrt(np.sum(np.square(traj.xyz[:, idx, :] - ref.xyz[:, idx, :]),
                                axis=(1, 2))/idx.shape[0])
    def move_ligand(self, positions, boxsize_x, boxsize_y, boxsize_z):
        positions = np.copy(positions)

        ligand_center = [np.array((0.0,0.0,0.0))]
        binding_site_center = [np.array((0.0,0.0,0.0))]

        # calculate center of mass ligand
        for idx in self.ligand_idxs:
            ligand_center += positions[:, idx, :]

        ligand_center = ligand_center/len(self.ligand_idxs)

        for idx in self.binding_site_idxs:
            binding_site_center += positions[:, idx, :]

        binding_site_center = binding_site_center/len(self.binding_site_idxs)

        diff = ligand_center - binding_site_center

        V = [np.array((0.0, 0.0, 0.0))]
         # x direction
        if diff[0][0] > boxsize_x /2 :
            V[0][0] = boxsize_x /2
        elif  diff[0][0] < -boxsize_x /2 :
            V[0][0] = -boxsize_x/2
            #  y direction
        if diff[0][1] > boxsize_y/2 :
            V[0][1] = boxsize_y/2
        elif  diff[0][1] < -boxsize_y/2 :
            V[0][1] = -boxsize_y/2
            # z direction
        if diff[0][2] > boxsize_z /2 :
            V[0][2] = boxsize_z /2
        elif  diff[0][1] < -boxsize_z /2 :
            V[0][2] = -boxsize_z/2
            # translate  ligand
        for idx in self.ligand_idxs:
           positions[:, idx, :] += V

        return positions

    def maketraj(self, walker):
        # convert box_vectors to angles and lengths for mdtraj
        # calc box length
        cell_lengths = np.array([[self._calc_length(v._value) for v in walker.box_vectors]])

        # TODO order of cell angles
        # calc angles
        cell_angles = np.array([[self._calc_angle(walker.box_vectors._value[i],
                                                 walker.box_vectors._value[j])
                                 for i, j in [(0,1), (1,2), (2,0)]]])

        # moves ligand inside the box
        positions = self.move_ligand(self._pos_to_array(walker.positions),
                                     cell_lengths[0][0], cell_lengths[0][1], cell_lengths[0][2])

        # make a traj out of it so we can calculate distances through
        # the periodic boundary conditions
        walker_traj = mdj.Trajectory(positions,
                                     topology=self.topology,
                                     unitcell_lengths=cell_lengths,
                                     unitcell_angles=cell_angles)
        return  walker_traj

    def distance(self, walkers):
        num_walkers = len(walkers)
        distance_matrix = np.zeros((num_walkers, num_walkers))
        for i in range(num_walkers):
            ref_traj = self.maketraj(walkers[i])
            for j in range(i+1, num_walkers):
                target_traj = self.maketraj(walkers[j])
                target_traj = target_traj.superpose(ref_traj, atom_indices=self.binding_site_idxs)
                d = self.rmsd(target_traj, ref_traj, self.ligand_idxs)
                distance_matrix[i][j] = d
                distance_matrix [j][i] = d

        return distance_matrix

