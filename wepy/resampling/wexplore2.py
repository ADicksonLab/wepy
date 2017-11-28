import geomm.recentering
import geomm.rmsd

import multiprocessing as mulproc
import random as rand
import itertools as it
from copy import copy
from copy import deepcopy

import numpy as np
import numpy.linalg as la

import mdtraj as mdj
import simtk.unit as unit

from wepy.resampling.resampler import Resampler, ResamplingRecord
from wepy.resampling.clone_merge import NothingInstructionRecord, CloneInstructionRecord,\
                                        SquashInstructionRecord, KeepMergeInstructionRecord
from wepy.resampling.clone_merge import CloneMergeDecision, CLONE_MERGE_INSTRUCTION_DTYPES

class OpenMMUnbindingDistance(object):
    # The distance function here returns a distance matrix where the element (d_ij) is the
    # RMSD between walkers i and j.  The RMSD is computed using the geomm package, by aligning
    # to the binding site atoms, and taking the RMSD of the ligand atoms.  It uses alternative maps
    # for the binding site atoms when defined, and aligns to all alternative maps, returning the
    # minimum RMSD computed over all maps.
    def __init__(self, topology=None, ligand_idxs=None, binding_site_idxs=None, alt_maps=None):
        self.topology = topology
        self.ligand_idxs = ligand_idxs
        self.binding_site_idxs = binding_site_idxs
        self.alt_maps = alt_maps
        # alt_maps are alternative mappings to the binding site
        # this program now assumes that all atoms in alternative maps are contained in binding_site_idxs list

    def _xyz_from_walkers(self, walkers, keep_atoms=[]):
        if len(keep_atoms) == 0:
            keep_atoms = range(np.shape(walkers[0].positions)[0])
            
        return np.stack(([np.array(w.positions.value_in_unit(unit.nanometer))[keep_atoms,:] for w in walkers]),axis=0)

    def _box_from_walkers(self, walkers):
        return np.stack(([np.array([la.norm(v._value) for v in w.box_vectors]) for w in walkers]),axis=0)
        
    def distance(self, walkers):
        num_walkers = len(walkers)

        small_lig_idxs = np.array(range(len(self.ligand_idxs)))
        small_bs_idxs = np.array(range(len(self.ligand_idxs),len(self.ligand_idxs)+len(self.binding_site_idxs)))
        keep_atoms = np.concatenate((self.ligand_idxs,self.binding_site_idxs),axis=0)
        
        small_pos = self._xyz_from_walkers(walkers,keep_atoms)
        box_lengths = self._box_from_walkers(walkers)
        newpos_small = geomm.recentering.recenter_receptor_ligand(small_pos,box_lengths,ligand_idxs=small_lig_idxs,receptor_idxs=small_bs_idxs)

        small_top = self.topology.subset(keep_atoms)
        traj_rec = mdj.Trajectory(newpos_small,small_top)

        traj_rec.superpose(traj_rec,atom_indices=small_bs_idxs)
        d = np.zeros((num_walkers,num_walkers))
        for i in range(num_walkers-1):
            d[i][i] = 0
            for j in range(i+1,num_walkers):
                # return the distance matrix in Angstroms
                d[i][j] = 10.0*geomm.rmsd.rmsd_one_frame(traj_rec.xyz[i],traj_rec.xyz[j],small_lig_idxs)
                d[j][i] = d[i][j]

        if self.alt_maps is not None:
            # figure out the "small" alternative maps
            small_alt_maps = deepcopy(self.alt_maps)
            for i, a in enumerate(self.alt_maps):
                for j, e in enumerate(a):
                    try:
                        small_alt_maps[i][j] = list(self.binding_site_idxs).index(e) + len(self.ligand_idxs)
                    except:
                        raise Exception('Alternative maps are assumed to be permutations of existing binding site indices')

            for alt_map in small_alt_maps:
                alt_traj_rec = mdj.Trajectory(newpos_small,small_top)
                alt_traj_rec.superpose(alt_traj_rec,atom_indices=small_bs_idxs,ref_atom_indices=alt_map)
                for i in range(num_walkers-1):
                    for j in range(i+1,num_walkers):
                        dtest = geomm.rmsd.rmsd_one_frame(traj_rec.xyz[i],alt_traj_rec.xyz[j],small_lig_idxs)
                        if dtest < d[i][j]:
                            d[i][j] = dtest
                            d[j][i] = dtest
        
        return d

class OpenMMRebindingDistance(object):
    # The distance function here returns a distance matrix where the element (d_ij) is the
    # difference between 1/RMSD_0(i) and 1/RMSD_0(j).  Where RMSD_0(i) is the RMSD of walker i
    # to the reference structure (comp_xyz), which is typically the crystallographic bound state.
    # The RMSDs to the bound state are computed using the geomm package, by aligning
    # to the binding site atoms, and taking the RMSD of the ligand atoms.  It uses alternative maps
    # for the binding site atoms when defined, and aligns to all alternative maps, returning the
    # minimum RMSD computed over all maps.
    def __init__(self, topology=None, ligand_idxs=None, binding_site_idxs=None, alt_maps=None, comp_xyz=None):
        self.topology = topology
        self.ligand_idxs = ligand_idxs
        self.binding_site_idxs = binding_site_idxs
        self.alt_maps = alt_maps
        
        self.comp_traj = self._make_comp_traj(comp_xyz)
        # alt_maps are alternative mappings to the binding site
        # this program now assumes that all atoms in alternative maps are contained in binding_site_idxs list
        # comp_xyz are the xyz coordinates of a reference state (usually the bound state)
        # this assumes that the xyz comes from the same topology

    def _make_comp_traj(self, comp_xyz):
        small_lig_idxs = np.array(range(len(self.ligand_idxs)))
        small_bs_idxs = np.array(range(len(self.ligand_idxs),len(self.ligand_idxs)+len(self.binding_site_idxs)))
        keep_atoms = np.concatenate((self.ligand_idxs,self.binding_site_idxs),axis=0)
        small_top = self.topology.subset(keep_atoms)
#        import pdb; pdb.set_trace()
        small_pos = np.array(comp_xyz)[:,keep_atoms,:]
        
        return mdj.Trajectory(small_pos,small_top)
        
    def _xyz_from_walkers(self, walkers, keep_atoms=[]):
        if len(keep_atoms) == 0:
            keep_atoms = range(np.shape(walkers[0].positions)[0])
            
        return np.stack(([np.array(w.positions.value_in_unit(unit.nanometer))[keep_atoms,:] for w in walkers]),axis=0)

    def _box_from_walkers(self, walkers):
        return np.stack(([np.array([la.norm(v._value) for v in w.box_vectors]) for w in walkers]),axis=0)
        
    def get_rmsd_native(self, walkers):
        num_walkers = len(walkers)

        small_lig_idxs = np.array(range(len(self.ligand_idxs)))
        small_bs_idxs = np.array(range(len(self.ligand_idxs),len(self.ligand_idxs)+len(self.binding_site_idxs)))
        keep_atoms = np.concatenate((self.ligand_idxs,self.binding_site_idxs),axis=0)
        
        small_pos = self._xyz_from_walkers(walkers,keep_atoms)
        box_lengths = self._box_from_walkers(walkers)
        newpos_small = geomm.recentering.recenter_receptor_ligand(small_pos,box_lengths,ligand_idxs=small_lig_idxs,receptor_idxs=small_bs_idxs)

        small_top = self.topology.subset(keep_atoms)
        traj_rec = mdj.Trajectory(newpos_small,small_top)

        traj_rec.superpose(self.comp_traj,atom_indices=small_bs_idxs)
        rmsd_native = np.zeros((num_walkers))
        for i in range(num_walkers):
            rmsd_native[i] = geomm.rmsd.rmsd_one_frame(traj_rec.xyz[i],self.comp_traj.xyz[0],small_lig_idxs)

        if self.alt_maps is not None:
            # figure out the "small" alternative maps
            small_alt_maps = deepcopy(self.alt_maps)
            for i, a in enumerate(self.alt_maps):
                for j, e in enumerate(a):
                    try:
                        small_alt_maps[i][j] = list(self.binding_site_idxs).index(e) + len(self.ligand_idxs)
                    except:
                        raise Exception('Alternative maps are assumed to be permutations of existing binding site indices')

            for alt_map in small_alt_maps:
                alt_traj_rec = mdj.Trajectory(newpos_small,small_top)
                alt_traj_rec.superpose(self.comp_traj,atom_indices=small_bs_idxs,ref_atom_indices=alt_map)
                for i in range(num_walkers):
                    dtest = geomm.rmsd.rmsd_one_frame(alt_traj_rec.xyz[i],self.comp_traj.xyz[0],small_lig_idxs)
                    if dtest < rmsd_native[i]:
                        rmsd_native[i] = dtest
        return rmsd_native
        
    def distance(self, walkers):
        num_walkers = len(walkers)
        rmsd_native = self.get_rmsd_native(walkers)
        d = np.zeros((num_walkers,num_walkers))
        for i in range(num_walkers-1):
            d[i][i] = 0
            for j in range(i+1,num_walkers):
                d[i][j] = abs(1./rmsd_native[i] - 1./rmsd_native[j])                
                d[j][i] = d[i][j]

        return d

class OpenMMNormalModeDistance(object):
    # The distance function here returns a distance matrix where the element (d_ij) is the
    # distance in "normal mode space". The NM coordinates are determined by aligning a structure to
    # align_xyz, and obtaining the dot product of a subset of coordinates (specified by align_idxs, typically
    # C-alphas), to a set of modes contained in modefile.
    def __init__(self, topology=None, align_idxs=None, align_xyz=None, n_modes=5, modefile=None):
        self.topology = topology
        self.n_modes = n_modes
        self.align_idxs = align_idxs
        
        assert len(align_xyz[0]) == len(align_idxs), "align_xyz and align_idxs must have the same number of atoms"
        self.small_top = self.topology.subset(align_idxs)
        self.align_traj = mdj.Trajectory(align_xyz,small_top)
        
        try:
            modes = np.loadtxt(modefile)
        except:
            raise Exception('Error reading from modefile: ',modefile)
        for m in modes.T:
            assert len(m) == 3*len(align_idxs), "Number of elements in each mode must be 3X the number of atoms"
        self.modes = modes.T

    def _xyz_from_walkers(self, walkers, keep_atoms=[]):
        if len(keep_atoms) == 0:
            keep_atoms = range(np.shape(walkers[0].positions)[0])
            
        return np.stack(([np.array(w.positions.value_in_unit(unit.nanometer))[keep_atoms,:] for w in walkers]),axis=0)

    def _box_from_walkers(self, walkers):
        return np.stack(([np.array([la.norm(v._value) for v in w.box_vectors]) for w in walkers]),axis=0)
        
    def distance(self, walkers):
        num_walkers = len(walkers)

        keep_atoms = np.array(self.align_idxs)
        small_pos = self._xyz_from_walkers(walkers,keep_atoms)
        box_lengths = self._box_from_walkers(walkers)
        
        traj_rec = mdj.Trajectory(small_pos,self.small_top)
        traj_rec.superpose(self.align_traj)

        vecs = [np.zeros((self.n_modes)) for i in range(n_walkers)]
        for i in range(n_walkers):
            coor_angstroms = traj_rec.xyz[i,:,:].flatten()*10.0
            for modenum in range(self.n_modes):
                vecs[i][modenum] = np.dot(coor_angstroms, self.modes[modenum])

        # calculate distance matrix in normal mode space
        d = np.zeros((n_walkers,n_walkers))
        for i in range(n_walkers):
            for j in range(i+1, n_walkers):
                dval = np.linalg.norm(vecs[i]-vecs[j],ord=2)
                d[i][j] = dval
                d[j][i] = dval

        return d
    
class WExplore2Resampler(Resampler):

    DECISION = CloneMergeDecision
    INSTRUCTION_DTYPES = CLONE_MERGE_INSTRUCTION_DTYPES

    def __init__(self, seed=None, pmin=1e-12, pmax=0.1, dpower=4, merge_dist=2.5,
                 topology=None, ligand_idxs=None, binding_site_idxs=None, alternative_maps=None,
                 distance_function='unbinding', comp_xyz=None, n_modes=None, modefile=None,nmalign_idxs=None):
        self.pmin=pmin
        self.lpmin = np.log(pmin/100)
        self.pmax=pmax
        self.dpower = dpower
        self.merge_dist = merge_dist
        self.seed = seed
        if seed is not None:
            rand.seed(seed)
        self.topology = topology
        supported_dfs = ['unbinding','rebinding','normal_mode']
        if distance_function == 'unbinding':
            self.distance_function = OpenMMUnbindingDistance(topology=topology,
                                                             ligand_idxs=ligand_idxs,
                                                             binding_site_idxs=binding_site_idxs,
                                                             alt_maps=alternative_maps)
        elif distance_function == 'rebinding':
            self.distance_function = OpenMMRebindingDistance(topology=topology,
                                                             ligand_idxs=ligand_idxs,
                                                             binding_site_idxs=binding_site_idxs,
                                                             alt_maps=alternative_maps,
                                                             comp_xyz=comp_xyz)
        elif distance_function == 'normal_mode':
            self.distance_function = OpenMMNormalModeDistance(topology=topology,
                                                              align_idxs=nmalign_idxs,
                                                              align_xyz=comp_xyz,
                                                              n_modes=n_modes,
                                                              modefile=modefile)
        else:
            raise Exception('distance function',distance_function,'is not recognized! Supported distance functions are as follows:',supported_dfs)

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
