import mdtraj as mdj
import numpy as np

# class for saving trajectories in hdf5 format
class trajectory_save:
    def  __init__(self,topolgy):
        self.toplogy = topolgy
        #self.saveformat = saveformat
        
    # converts openmm_positions format to mdtraj format    
    def mdtraj_positions(self, openmm_positions):
        
        n_atoms = self.toplogy.n_atoms
        
        xyz = np.zeros(( n_atoms, 3))
        positions = openmm_positions
        
        
        for i in range(len(positions)):
            xyz[i,:] = ([positions[i]._value[0], positions[i]._value[1],
                                                        positions[i]._value[2]])
        
        return xyz
            
          
    def save(self, filename, resampled_positions, unitcell_lengths, unitcell_angles):

        n_frames = len (resampled_positions)
        
        xyz = np.zeros((n_frames, self.toplogy.n_atoms,3))        
        
        # make positions for every frame
        for i in range(n_frames):
            xyz[i,:,:] = self.mdtraj_positions(resampled_positions[i])
        
        # make time array for trajectory
        time = np.array([i for i in range(n_frames)])
                         
        newtraj = mdj.Trajectory(xyz,self.toplogy, time=time,
                                    unitcell_lengths=unitcell_angles, unitcell_angles=unitcell_angles)
        # save in hdf5 format
        newtraj.save_hdf5(filename)     
