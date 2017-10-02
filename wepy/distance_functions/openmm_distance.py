import numpy as np
import numpy.linalg as la
import mdtraj as mdj


from wepy.openmm import OpenMMRunner, OpenMMWalker
from wepy.distance_functions.distance import Distance

class OpenMMDistance(Distance):
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
