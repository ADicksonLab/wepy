import pickle
import numpy as np

from mastic.interactions.hydrogen_bond import HydrogenBondType
import mastic.profile as masticprof
from mastic.interaction_space import InteractionSpace

from wepy.resampling.distances import OpenMMHBondDistance
import mdtraj as mdj
#import ipdb; ipdb.set_trace()

print("Loading system type..")
system_pkl_path = 'data/sEH_TPPU_SystemType.pkl'
with open(system_pkl_path, 'rb') as rf:
    sEH_TPPU_SystemType = pickle.load(rf)

print("Building inx space..")
# generate the interaction space we will be profiling
inx_space = InteractionSpace(sEH_TPPU_SystemType)

# we want associations for all combinations of members for a degree 2
# interaction (e.g. hydrogen bonds)
# so we could use this method I've commented out or just define it ourselves
# assoc_terms = sEH_TPPU_SystemType.association_polynomial(
#     degree=2,
#     permute=True,
#     replace=True,
#     return_idxs=True)

assoc_terms = [(0,1), (1,0)]
# make the unit associations, interaction classes, and add to interaction space
for assoc_term in assoc_terms:
    # make the unit AssociationTypes
    assoc_idx = sEH_TPPU_SystemType.make_unit_association_type(assoc_term)
    association_type = sEH_TPPU_SystemType.association_types[assoc_idx]

    # make HydrogenBondType interaction classes for this association
    # in the inx_space
    inx_space.add_association_subspace(association_type, HydrogenBondType)

print("Building profiler..")
# make a Profiler for the inx space
profiler = masticprof.InxSpaceProfiler(inx_space)


pdbpath = 'data/sEH_TPPU_system.pdb'
pdb = mdj.load_pdb(pdbpath)
lig_idxs = pdb.topology.select('resname "2RV"')
protein_idxs = np.array([atom.index for atom in pdb.topology.atoms if atom.residue.is_protein])

print("Setting up distance metric..")
hbond_distance = OpenMMHBondDistance(ligand_idxs=lig_idxs,
                                     protein_idxs=protein_idxs,
                                     profiler=profiler,
                                     sys_type=sEH_TPPU_SystemType)

print("Loading walkers..")
walker_pkl_path = 'data/sEH_walkers.pkl'
with open(walker_pkl_path, 'rb') as rf:
    sEH_walkers = pickle.load(rf)

print("Computing distance..")
dmat = hbond_distance.distance(sEH_walkers)
