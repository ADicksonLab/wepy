import numpy as np

import mdtraj as mdj

from wepy.openmm import OpenMMRunner, OpenMMWalker
from wepy.distance_functions.distance import Distance

class OpenMMDistance(Distance):
    def __init__(self, topology= None, ligand_idxs=None, binding_site_idxs=None):
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


    def distance(self, walkers):
        n_walkers = len (walkers)
        distance_matrix = np.zeros((n_walkers, n_walkers))
        for i in range(n_walkers):
            for j in range(i+1, n_walkers):
                d = self.calculate_rmsd(walkers[i].positions[0:self.topology.n_atoms],
                                        walkers[j].positions[0:self.topology.n_atoms])
                                
                distance_matrix[i][j] = d
                distance_matrix [j][i] = d

        return distance_matrix
