from copy import copy
from copy import deepcopy

import numpy as np
import numpy.linalg as la
import simtk.unit as unit

import geomm.recentering
import geomm.rmsd

import mdtraj as mdj

class OpenMMDistance(object):
    """ Class for distance metrics that take in OpenMM walkers
     and return matrix of distances
    """
    def _xyz_from_walkers(self, walkers, keep_atoms=[]):
        if len(keep_atoms) == 0:
            keep_atoms = range(np.shape(walkers[0].positions)[0])
            
        return np.stack(([np.array(w.positions.value_in_unit(unit.nanometer))[keep_atoms,:] for w in walkers]),axis=0)

    def _box_from_walkers(self, walkers):
        return np.stack(([np.array([la.norm(v._value) for v in w.box_vectors]) for w in walkers]),axis=0)

class OpenMMUnbindingDistance(OpenMMDistance):
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

class OpenMMRebindingDistance(OpenMMDistance):
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

class OpenMMNormalModeDistance(OpenMMDistance):
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

