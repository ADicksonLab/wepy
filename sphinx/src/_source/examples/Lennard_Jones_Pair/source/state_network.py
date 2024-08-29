from pathlib import Path

import numpy as np

from wepy.hdf5 import WepyHDF5
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.analysis.transitions import run_transition_probability_matrix
from wepy.analysis.network import MacroStateNetwork
from wepy.analysis.contig_tree import ContigTree

output_dir = Path('_output')
sim_dir = output_dir / 'we'

# Load wepy hdf5 file into python script
wepy_h5 = WepyHDF5(sim_dir / 'results.wepy.h5', mode = 'r+')
run_idx = 0
assg_key = 'rand_assg_idx'
n_classifications = 4
random_seed = 1

np.random.seed(random_seed)

# make random assignments

# observable function
def rand_assg(fields_d, *args, **kwargs):
    assignments = np.random.randint(0, n_classifications,
                                            size=fields_d['weights'].shape)
    return assignments

with wepy_h5:

    # compute this random assignment "observable"
    wepy_h5.compute_observable(
        rand_assg,
        ['weights'],
        (),
        save_to_hdf5=assg_key,
        return_results=False,
    )

# make a contig tree
contig_tree = ContigTree(
    wepy_h5,
    decision_class=MultiCloneMergeDecision,
    boundary_condition_class=UnbindingBC,
)


# make a state network using the key of the assignment and the
# file. Without edges this is just a collection of "macrostates"
# or groupings of microstates from the simulation data. We also
# set some sort of lag time to compute transition probabilities
# with

random_macrostates = MacroStateNetwork(contig_tree,
                                       assg_field_key="observables/{}".format(assg_key),
                                       transition_lag_time=3)

node_id = random_macrostates.node_ids[0]

# we can do things like make a trajectory in mdtraj and output as a
# dcd for a state
traj = random_macrostates.state_to_mdtraj(node_id)
traj.save_dcd(str(output_dir / "state.dcd".format(node_id)))

# we also can automatically compute the weights of the macrostates.
random_macrostates.set_macrostate_weights()

# this sets them as macrostate (node) attributes
print("node {} weight:".format(node_id))
print(random_macrostates.get_node_attribute(node_id, 'Weight'))

# we can also get a transition probability matrix from this
print(random_macrostates.probmat)


# furthermore you can write the network out to a GEXF file that
# can be used for visualization
random_macrostates.write_gexf(str(output_dir / "random_macrostates.csn.gexf"))
