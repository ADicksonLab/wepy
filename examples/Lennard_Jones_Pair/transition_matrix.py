import numpy as np
import networkx as nx

from wepy.hdf5 import WepyHDF5
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.tree.ancestor import sliding_window
import wepy.analysis.transitions

# Load wepy hdf5 file into python script
wepy_h5 = WepyHDF5('wexplore1_results.wepy.h5', mode = 'r')
wepy_h5.open()

# make a parent matrix from the hdf5 resampling records
resampling_panel = wepy_h5.run_resampling_panel(0)
parent_panel = MultiCloneMergeDecision.parent_panel(resampling_panel)
parent_matrix = MultiCloneMergeDecision.net_parent_table(parent_panel)

# use the parent matrix to generate the sliding windows
window_length = 10
windows = sliding_window(parent_matrix, window_length)

# make the transition matrix from the windows
