import numpy as np

from wepy.hdf5 import WepyHDF5
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.tree.ancestor import sliding_window
from wepy.analysis.transitions import run_transition_probability_matrix

# Load wepy hdf5 file into python script
wepy_h5 = WepyHDF5('wexplore1_results.wepy.h5', mode = 'r+')
wepy_h5.open()

run_idx = 0
assg_key = 'rand_assg_idx'
n_classifications = 50
# make random assignments

# observable function
def rand_assg(fields_d, *args, **kwargs):
    assignments = np.random.random_integers(0, n_classifications,
                                            size=fields_d['weights'].shape[0])
    return assignments

# compute this random assignment "observable"
wepy_h5.compute_observable(rand_assg, ['weights'],
                           save_to_hdf5=assg_key, return_results=False)

# make a parent matrix from the hdf5 resampling records
resampling_panel = wepy_h5.run_resampling_panel(run_idx)
parent_panel = MultiCloneMergeDecision.parent_panel(resampling_panel)
parent_matrix = MultiCloneMergeDecision.net_parent_table(parent_panel)

# use the parent matrix to generate the sliding windows
window_length = 10
windows = sliding_window(parent_matrix, window_length)

# make the transition matrix from the windows
transprob_mat = run_transition_probability_matrix(wepy_h5, run_idx,
                                                  "observables/{}".format(assg_key), windows)
