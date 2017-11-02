import numpy as np

import mdtraj as mdj
import numpy as np
import numpy.linalg as la

import networkx as nx
from tree import monte_carlo_minimization, make_graph

from wepy.resampling.clone_merge import CloneMergeDecision, clone_parent_table, clone_parent_panel
from wepy.resampling.resampler import Resampler, ResamplingRecord
from wepy.resampling.clone_merge import NothingInstructionRecord
from wepy.hdf5 import TrajHDF5
from wepy.hdf5 import TRAJ_DATA_FIELDS

class GetResult(object):
    def __init__(self, hd):
        self.hd = hd
        
        

   # function for extracting positions
   
    def get_data(self, run, walker_idx, cycle_idx, data):
        return list(self.hd['/runs/{}/trajectories/{}/{}'.
                          format(run, walker_idx, data)])[cycle_idx]

    def calc_angle(self, v1, v2):
       return np.degrees(np.arccos(np.dot(v1, v2)/(la.norm(v1) * la.norm(v2))))

    def calc_length(self, v):
       return la.norm(v)


     def make_cycle_tarj(self, run, walker_num,  cycle_idx, topology):
        times = []
        weights = [ ]

        unitcell_angles = []

        unitcell_lengths = []

        xyz = np.zeros((walker_num, topology.n_atoms,3))

        positions = list(self.hd['/runs/{}/trajectories/{}/positions'.
                          format(run, walker_idx, data)])[0:walker_num]
        
        times = list(self.hd['/runs/{}/trajectories/{}/time'.
                           format(run, walker_idx, data)])[0:walker_num]
        
        
    def make_traj(self, run, walkers_idx_list, cycle_idx, topology):
    
       
        walker_num = len(walkers_idx_list)
       
        times = []
        weights = [ ]

        unitcell_angles = []
        unitcell_lengths = []

        xyz = np.zeros((walker_num, topology.n_atoms,3))
        for walker_idx in walkers_idx_list:
           xyz[walker_idx,:,:] = self.get_data(run, walker_idx, cycle_idx,'positions')
           box_vectors = self.get_data(run, walker_idx, cycle_idx, 'box_vectors')
           weight = self.get_data(run, walker_idx, cycle_idx, 'weights')
           weights.append(weight)
           time = self.get_data(run, walker_idx, cycle_idx, 'time')
           times.append(time)
           
           

           unitcell_lengths.append( [self.calc_length(v) for v in box_vectors])
           unitcell_angles.append(np.array([self.calc_angle(box_vectors[i], box_vectors[j])
                                            for i, j in [(0,1), (1,2), (2,0)]]))
        
        traj = mdj.Trajectory(xyz, topology, time=times,
                                    unitcell_lengths=unitcell_angles, unitcell_angles=unitcell_angles)
        return traj
    def get_cycle_data(self, run, cycle_idx, walker_num, data_type):
        
        Data = []
        for walker_idx in range(walker_num):
            weight = self.get_data(run, walker_idx, cycle_idx, data_type)
            Data.append(weight)
        return Data
