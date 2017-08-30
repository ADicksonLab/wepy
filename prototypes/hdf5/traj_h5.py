import mdtraj as mdj
import numpy as np
import h5py

from wepy.hdf5 import TrajHDF5

traj = mdj.load_dcd('test_traj.dcd', top="sEH_TPPU_system.pdb")

# load the topology from the hdf5 file
top_h5 = h5py.File("mdtraj_top.h5")
# it is in bytes so we need to decode to a string, which is in JSON format
top_str = top_h5['topology'][0].decode()

# time
time = traj.time

# positions
positions_q = [traj.openmm_positions(i) for i in range(traj.n_frames)]
positions = np.array([e._value for e in positions_q])
positions_unit = positions_q[0].unit.get_name()

# box_vectors
box_vectors_q = [traj.openmm_boxes(i) for i in range(traj.n_frames)]
box_vectors = np.array([e._value for e in box_vectors_q])
box_vectors_unit = box_vectors_q[0].unit.get_name()

traj_h5 = TrajHDF5('test_traj.h5', mode='w',
                   topology=top_str,
                   positions=positions,
                   time=time,
                   box_vectors=box_vectors,
                   positions_unit=positions_unit,
                   time_unit='second',
                   box_vectors_unit=box_vectors_unit)

traj_h5.close()
