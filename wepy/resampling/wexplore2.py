import random as rand

import numpy as np
import numpy.linalg as la

import mdtraj as mdj

from wepy.resampling.resampler import Resampler, ResamplingRecord
from wepy.resampling.clone_merge import NothingInstructionRecord, CloneInstructionRecord,\
                                        SquashInstructionRecord, KeepMergeInstructionRecord
from wepy.resampling.clone_merge import CloneMergeDecision, CLONE_MERGE_INSTRUCTION_DTYPES

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

class WExplore2Resampler(Resampler):

    DECISION = CloneMergeDecision
    INSTRUCTION_DTYPES = CLONE_MERGE_INSTRUCTION_DTYPES

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
        self.distance_function = OpenMMDistance(self.topology,
                                                self.ligand_idxs, self.binding_site_idxs)

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
            walker_actions = [ResamplingRecord(decision=CloneMergeDecision.NOTHING.value,
                                               instruction=NothingInstructionRecord(slot=i))
                              for i in range(n_walkers)]

            # selects a walker with minimum wsum and a walker with maximum wsum
            max_tups = [(value, i) for i, value in enumerate(wsum)
                                     if (amp[i] >= 1) and (walkerwt[i] > self.pmin)]
            if len(max_tups):
                maxvalue, maxwind = max(max_tups)

            min_tups = [(value, i) for i,value in enumerate(wsum)
                                    if amp[i] == 1 and (walkerwt[i]  < self.pmax)]
            if len(min_tups) > 0:
                minvalue, minwind = min(min_tups)

            # does minwind have an eligible merging partner?
            closedist = self.merge_dist
            closewalk = None
            condition_list = np.array([i is not None for i in [minwind,maxwind]])
            if condition_list.all() and minwind != maxwind:

                closewalk_availabe = set(range(n_walkers)).difference([minwind,maxwind])
                closewalk_availabe = [idx for idx in closewalk_availabe
                                      if amp[idx]==1 and (walkerwt[idx] < self.pmax)]
                if len(closewalk_availabe) > 0:
                    tups = [(distance_matrix[minwind][i], i) for i in closewalk_availabe
                                            if distance_matrix[minwind][i] < (self.merge_dist)]
                    if len(tups) > 0:
                        closedist, closewalk = min(tups)


            # did we find a closewalk?
            condition_list = np.array([i is not None for i in [minwind,maxwind,closewalk]])
            if condition_list.all() :

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

                    n_clone_merges += 1
                    productive = True
                    spread = newspread

                    # make a decision on which walker to keep (minwind, or closewalk)
                    r = rand.uniform(0.0, walkerwt[closewalk] + walkerwt[minwind])
                     # keeps closewalk and gets rid of minwind
                    if r < walkerwt[closewalk]:
                        keep_idx = closewalk
                        squash_idx = minwind

                    # keep minwind, get rid of closewalk
                    else:
                        keep_idx = minwind
                        squash_idx = closewalk

                    # update weight
                    walkerwt[keep_idx] += walkerwt[squash_idx]
                    walkerwt[squash_idx] = 0.0

                    # update amps
                    amp[squash_idx] = 0
                    amp[keep_idx] = 1

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

        if n_clone_merges == 0:
            return([walker_actions]), spreads[-1]
        else:
            return  resampling_actions, spreads[-1]

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
        resampled_walkers = self._clone_merge(walkers, resampling_actions, debug_prints=debug_prints)

        data = {'distance_matrix' : distance_matrix, 'spread' : np.array([spread])}
        return resampled_walkers, resampling_actions, data
