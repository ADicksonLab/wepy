import numpy as np

from wepy.hdf5 import WepyHDF5
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.analysis.transitions import run_transition_probability_matrix
from wepy.analysis.network import StateNetwork

# Load wepy hdf5 file into python script
wepy_h5 = WepyHDF5('../outputs/results.wepy.h5', mode = 'r+')
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


# make a state network using the key of the assignment and the
# file. Without edges this is just a collection of "macrostates" or
# groupings of microstates from the simulation data
random_macrostates = StateNetwork(wepy_h5, "observables/{}".format(assg_key))

# we can do things like make a trajectory in mdtraj and output as a
# dcd for a state
state = 32
traj = random_macrostates.state_to_mdtraj(state)
traj.save_dcd("rand_state_{}.dcd".format(state))

