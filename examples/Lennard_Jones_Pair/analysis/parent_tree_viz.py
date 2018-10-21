import numpy as np

from wepy.hdf5 import WepyHDF5
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.analysis.contig_tree import ContigTree
from wepy.analysis.parents import ParentForest

# Load wepy hdf5 file into python script
wepy_h5 = WepyHDF5('../outputs/results.wepy.h5', mode = 'r+')

with wepy_h5:

    # make a contig tree
    contig_tree = ContigTree(wepy_h5, decision_class=MultiCloneMergeDecision,
                             boundary_condition_class=UnbindingBC)

    # get a parent matrix from this to make a parent forest network
    # object
