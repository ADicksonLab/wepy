import numpy as np

from wepy.hdf5 import WepyHDF5
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.analysis.transitions import run_transition_probability_matrix
from wepy.analysis.network import MacroStateNetwork
from wepy.analysis.contig_tree import ContigTree

# Load wepy hdf5 file into python script
wepy_h5 = WepyHDF5('../outputs/results.wepy.h5', mode = 'r+')
run_idx = 0
assg_key = 'rand_assg_idx'
n_classifications = 50
# make random assignments

# observable function
def rand_assg(fields_d, *args, **kwargs):
    assignments = np.random.random_integers(0, n_classifications,
                                            size=fields_d['weights'].shape[0])
    return assignments

with wepy_h5:

    # compute this random assignment "observable"
    wepy_h5.compute_observable(rand_assg, ['weights'],
                               save_to_hdf5=assg_key, return_results=False)


    # make a contig tree
    contig_tree = ContigTree(wepy_h5, decision_class=MultiCloneMergeDecision,
                             boundary_condition_class=UnbindingBC)


    # make a state network using the key of the assignment and the
    # file. Without edges this is just a collection of "macrostates" or
    # groupings of microstates from the simulation data
    random_macrostates = MacroStateNetwork(contig_tree, "observables/{}".format(assg_key))

# we can do things like make a trajectory in mdtraj and output as a
# dcd for a state
state = 32
traj = random_macrostates.state_to_mdtraj(state)
traj.save_dcd("rand_state_{}.dcd".format(state))

# we also can automatically compute the weights of the macrostates.
random_macrostates.set_macrostate_weights()

# this sets them as macrostate (node) attributes
print(random_macrostates.get_node_attribute(state, 'Weight'))

