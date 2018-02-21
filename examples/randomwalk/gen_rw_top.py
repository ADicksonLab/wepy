"""
This module creates a dummy trajectory with one dummy atom and a fake topology
for the randomwalk system using mdtraj package. Additionally, this module saves
the topology of this system in JSON format.
"""

import os

import numpy as np
import pandas as pd
import mdtraj as mdj
import json
import h5py


if __name__=="__main__":

    #RandomWalk system with one atom
    n_atoms = 1

    # make one dummy atom
    data = []
    for i in range(n_atoms):
        data.append(dict(serial=i, name="H", element="H", resSeq=i + 1, resName="UNK", chainID=0))

    # convert to pandas data frame format
    data = pd.DataFrame(data)

    # creates the array of position for the system
    xyz = np.zeros((1, 1, 3))

    # defines box vector lengths
    unitcell_lengths = 0.943 * np.ones((1, 3))
    unitcell_angles = 90 * np.ones((1, 3))

    # create the topology and the trajectory of teh randomwalk systes
    top = mdj.Topology.from_dataframe(data, bonds=np.zeros((0, 2), dtype='int'))
    traj = mdj.Trajectory(xyz, top, unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)

    #save the trajectory in h5 and pdb format
    traj.save_hdf5("tmp_mdtraj_system.h5")

    # we need a JSON string for now in the topology section of the
    # HDF5 so we just load the topology from the hdf5 file
    top_h5 = h5py.File("tmp_mdtraj_system.h5")

    # it is in bytes so we need to decode to a string, which is in JSON format
    json_top_str = top_h5['topology'][0].decode()
    top_h5.close()

    # write the JSON topology out
    with open("randomwalk_system.top.json", mode='w') as json_wf:
        json_wf.write(json_top_str)

    # remove tmporary h5 file
    os.remove("tmp_mdtraj_system.h5")
