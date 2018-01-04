import pickle
from wepy.openmm import OpenMMRunner, OpenMMWalker
from wepy.resampling.distances import OpenMMUnbindingDistance, OpenMMRebindingDistance

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj
import numpy as np

pklpath = 'data/hg_walkers.pkl'
pdbpath = 'data/hg_complex.pdb'

with open(pklpath, 'rb') as f:
    initial_walkers = pickle.load(f)

pdb = mdj.load_pdb(pdbpath)

lig_idxs = pdb.topology.select('resname "GST"')
bs_idxs = [3,6,10,13,17,20,24,27,31,34,38,41,45,48,53,55]

my_unbinding_distance = OpenMMUnbindingDistance(topology=pdb.topology,
                                                ligand_idxs=lig_idxs,
                                                binding_site_idxs=bs_idxs)

dmat_unb = my_unbinding_distance.distance(initial_walkers)
rmsd0 = np.loadtxt('data/hg_rmsd0.dat')

passed1 = True
for i in range(1,48):
    if (np.abs(rmsd0[i]-dmat_unb[0][i]) > 0.01):
        passed1 = False

my_rebinding_distance = OpenMMRebindingDistance(topology=pdb.top,
                                                ligand_idxs=lig_idxs,
                                                binding_site_idxs=bs_idxs,
                                                comp_xyz=pdb.xyz)

dmat_reb = my_rebinding_distance.distance(initial_walkers)

rmsd_nat = np.loadtxt('data/hg_rmsd_nat.dat')
passed2 = True
for i in range(1,48):
    if (np.abs(10*np.abs(1/rmsd_nat[i]-1/rmsd_nat[0]) - dmat_reb[0][i]) > 0.01):
        passed2 = False

allpassed = True
if passed1:
    print("Test1 PASSED")
else:
    print("Test1 FAILED")
    allpassed = False

if passed2:
    print("Test2 PASSED")
else:
    print("Test2 FAILED")
    allpassed = False

if allpassed:
    print("All tests PASSED successfully")
else:
    print("One or more tests FAILED!")
    

