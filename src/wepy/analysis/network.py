"""Module that allows for imposing a kinetically connected network
structure of weighted ensemble simulation data.
"""

from collections import defaultdict
from copy import deepcopy

import networkx as nx

from wepy.analysis.transitions import transition_counts, counts_d_to_matrix, \
                                      normalize_counts

class MacroStateNetworkError(Exception):
    """Errors specific to MacroStateNetwork requirements."""
    pass

class MacroStateNetwork():
    """Provides an abstraction over weighted ensemble data in the form of
    a kinetically connected network.

    The MacroStateNetwork refers to any grouping of the so called
    "micro" states that were observed during simulation,
    i.e. trajectory frames, and not necessarily in the usual sense
    used in statistical mechanics. Although it is the perfect vehicle
    for working with such macrostates.

    Because walker trajectories in weighted ensemble there is a
    natural way to generate the edges between the macrostate nodes in
    the network. These edges are determined automatically and a lag
    time can also be specified, which is useful in the creation of
    Markov State Models.
    """

    ASSIGNMENTS = 'assignments'
    """Key for the microstates that are assigned to a macrostate."""

    def __init__(self, contig_tree, assg_field_key=None, assignments=None,
                 transition_lag_time=2):
        """Create a network of macrostates from the simulation microstates
        using a field in the trajectory data or precomputed assignments.

        Either 'assg_field_key' or 'assignments' must be given, but not
        both.

        The lag time is default set to 2, which is the natural connection
        between microstates. The lag time can be increased to vary the
        kinetic accuracy of transition probabilities generated through
        Markov State Modelling.

        Alternatively the MacroStateNetwork can also be though of as just
        a way of mapping macrostate properties to the underlying
        microstate data.

        Arguments
        ---------
        contig_tree : ContigTree object

        assg_field_key : str, conditionally optional on 'assignments'
            The field in the WepyHDF5 dataset you want to generate macrostates for.

        assignments : list of array_like of dim (num_traj, num_cycles),
                      conditionally optional on 'assg_field_key'
            List of assignments for all frames in each run.
        """

        self._graph = nx.DiGraph()

        assert not (assg_field_key is None and assignments is None), \
            "either assg_field_key or assignments must be given"

        assert assg_field_key is not None or assignments is not None, \
            "one of assg_field_key or assignments must be given"

        self._contig_tree = contig_tree
        self._wepy_h5 = self._contig_tree.wepy_h5

        self._assg_field_key = assg_field_key

        # the temporary assignments dictionary
        self._node_assignments = None
        # and temporary raw assignments
        self._assignments = None

        # map the keys to their lists of assignments, depending on
        # whether or not we are using a field from the HDF5 traj or
        # assignments provided separately
        if assg_field_key is not None:
            assert type(assg_field_key) == str, "assignment key must be a string"

            self._key_init(assg_field_key)
        else:
            self._assignments_init(assignments)

        # once we have made th dictionary add the nodes to the network
        # and reassign the assignments to the nodes
        self._node_idxs = {}
        for node_idx, assg_item in enumerate(self._node_assignments.items()):
            assg_key, assigs = assg_item
            self._graph.add_node(assg_key, node_idx=node_idx, assignments=assigs)
            self._node_idxs[assg_key] = node_idx

        # then we compute the total weight of the macrostate and set
        # that as the default node weight
        #self.set_macrostate_weights()

        # now count the transitions between the states and set those
        # as the edges between nodes

        # first get the sliding window transitions from the contig
        # tree, once we set edges for a tree we don't really want to
        # have multiple sets of transitions on the same network so we
        # don't provide the method to add different assignments
        if transition_lag_time is not None:

            # set the lag time attribute
            self._transition_lag_time = transition_lag_time

            # get the transitions
            transitions = []
            for window in self._contig_tree.sliding_windows(self._transition_lag_time):

                transition = [window[0], window[-1]]

                # convert the window trace on the contig to a trace
                # over the runs
                transitions.append(transition)

            # then get the counts for those edges
            counts_d = transition_counts(self._assignments, transitions)

            # create the edges and set the counts into them
            for edge, trans_counts in counts_d.items():
                self._graph.add_edge(*edge, counts=trans_counts)

            # then we also want to get the transition probabilities so
            # we get the counts matrix and compute the probabilities
            # we first have to replace the keys of the counts of the
            # node_ids with the node_idxs
            node_id_to_idx_dict = self.node_id_to_idx_dict()
            self._countsmat = counts_d_to_matrix(
                                {(node_id_to_idx_dict[edge[0]],
                                  node_id_to_idx_dict[edge[1]]) : counts
                                 for edge, counts in counts_d.items()})
            self._probmat = normalize_counts(self._countsmat)

            # then we add these attributes to the edges in the network
            node_idx_to_id_dict = self.node_id_to_idx_dict()
            for i_id, j_id in self._graph.edges:
                # i and j are the node idxs so we need to get the
                # actual node_ids of them
                i_idx = node_idx_to_id_dict[i_id]
                j_idx = node_idx_to_id_dict[j_id]

                # convert to a normal float and set it as an explicitly named attribute
                self._graph.edges[i_id, j_id]['transition_probability'] = \
                                                            float(self._probmat[i_idx, j_idx])

                # we also set the general purpose default weight of
                # the edge to be this.
                self._graph.edges[i_id, j_id]['Weight'] = \
                                                float(self._probmat[i_idx, j_idx])


        # then get rid of the assignments dictionary, this information
        # can be accessed from the network
        del self._node_assignments
        del self._assignments

    def _key_init(self, assg_field_key):
        """Initialize the assignments structures given the field key to use.

        Parameters
        ----------
        assg_field_key : str

        """

        # the key for the assignment in the wepy dataset
        self._assg_field_key = assg_field_key

        # blank assignments
        assignments = [[[] for traj_idx in range(self._wepy_h5.num_run_trajs(run_idx))]
                             for run_idx in self._wepy_h5.run_idxs]

        # the raw assignments
        curr_run_idx = -1
        for idx_tup, fields_d in self._wepy_h5.iter_trajs_fields(
                                         [self.assg_field_key], idxs=True):
            run_idx = idx_tup[0]
            traj_idx = idx_tup[1]
            assg_field = fields_d[self.assg_field_key]

            assignments[run_idx][traj_idx].extend(assg_field)

        # then just call the assignments constructor to do it the same
        # way
        self._assignments_init(assignments)

    def _assignments_init(self, assignments):
        """Given the assignments structure sets up the other necessary
        structures.

        Parameters
        ----------
        assignments : list of array_like of dim (num_traj, num_cycles),
                      conditionally optional on 'assg_field_key'
            List of assignments for all frames in each run.

        """

        # set the raw assignments to the temporary attribute
        self._assignments = assignments

        # this is the dictionary mapping node_id -> the (run_idx, traj_idx, cycle_idx) frames
        self._node_assignments = defaultdict(list)

        for run_idx, run in enumerate(assignments):
            for traj_idx, traj in enumerate(run):
                for frame_idx, assignment in enumerate(traj):
                    self._node_assignments[assignment].append( (run_idx, traj_idx, frame_idx) )

    def node_id_to_idx(self, assg_key):
        """Convert a node_id (which is the assignment value) to a canonical index.

        Parameters
        ----------
        assg_key : node_id

        Returns
        -------
        node_idx : int

        """
        return self.node_id_to_idx_dict()[assg_key]

    def node_idx_to_id(self, node_idx):
        """Convert a node index to its node id.

        Parameters
        ----------
        node_idx : int

        Returns
        -------
        node_id : node_id


        """
        return self.node_idx_to_id_dict()[node_idx]

    def node_id_to_idx_dict(self):
        """Generate a full mapping of node_ids to node_idxs."""
        return self._node_idxs

    def node_idx_to_id_dict(self):
        """Generate a full mapping of node_idxs to node_ids."""
        # just reverse the dictionary and return
        return {node_idx : node_id for node_id, node_idx in self._node_idxs}


    @property
    def graph(self):
        """The networkx.DiGraph of the macrostate network."""
        return self._graph

    @property
    def num_states(self):
        """The number of states in the network."""
        return len(self.graph)

    @property
    def node_ids(self):
        """A list of the node_ids."""
        return list(self.graph.nodes)

    @property
    def contig_tree(self):
        """The underlying ContigTree"""
        return self._contig_tree

    @property
    def wepy_h5(self):
        """The underlying WepyHDF5 object."""
        return self._wepy_h5

    @property
    def assg_field_key(self):
        """The string key of the field used to make macro states from the WepyHDF5 dataset.

        Raises
        ------
        MacroStateNetworkError
            If this wasn't used to construct the MacroStateNetwork.

        """
        try:
            return self._assg_field_key
        except AttributeError:
            raise MacroStateNetworkError("Assignments were manually defined, no key.")

    @property
    def countsmat(self):
        """Return the transition counts matrix of the network.

        Raises
        ------
        MacroStateNetworkError
            If no lag time was given.

        """
        try:
            return self._countsmat
        except AttributeError:
            raise MacroStateNetworkError("transition counts matrix not calculated")

    @property
    def probmat(self):
        """Return the transition probability matrix of the network.

        Raises
        ------
        MacroStateNetworkError
            If no lag time was given.

        """

        try:
            return self._probmat
        except AttributeError:
            raise MacroStateNetworkError("transition probability matrix not set")

    def get_node_attributes(self, node_id):
        """Returns the node attributes of the macrostate.

        Parameters
        ----------
        node_id : node_id

        Returns
        -------
        macrostate_attrs : dict

        """
        return self.graph.nodes[node_id]

    def get_node_attribute(self, node_id, attribute_key):
        """Return the value for a specific node and attribute.

        Parameters
        ----------
        node_id : node_id

        attribute_key : str

        Returns
        -------
        node_attribute

        """
        return self.get_node_attributes(node_id)[attribute_key]

    def node_assignments(self, node_id):
        """Return the microstates assigned to this macrostate as a run trace.

        Parameters
        ----------
        node_id : node_id

        Returns
        -------
        node_assignments : list of tuples of ints (run_idx, traj_idx, cycle_idx)
            Run trace of the nodes assigned to this macrostate.

        """
        return self.get_node_attribute(node_id, self.ASSIGNMENTS)

    def get_node_fields(self, node_id, fields):
        """Return the trajectory fields for all the microstates in the
        specified macrostate.

        Parameters
        ----------
        node_id : node_id

        fields : list of str
            Field name to retrieve.

        Returns
        -------

        fields : dict of str: array_like
           A dictionary mapping the names of the fields to an array of the field.
           Like fields of a trace.

        """
        node_trace = self.node_assignments(node_id)

        # use the node_trace to get the weights from the HDF5
        with self.wepy_h5:
            fields_d = self.wepy_h5.get_trace_fields(node_trace, fields)

        return fields_d

    def iter_nodes_fields(self, fields):
        """Iterate over all nodes and return the field values for all the
        microstates for each.

        Parameters
        ----------
        fields : list of str

        Returns
        -------
        nodes_fields : dict of node_id: (dict of field: array_like)
            A dictionary with an entry for each node.
            Each node has it's own dictionary of node fields for each microstate.

        """

        nodes_d = {}
        for node_id in self.graph.nodes:
            fields_d = self.get_node_fields(node_id, fields)
            nodes_d[node_id] = fields_d

        return nodes_d

    def set_nodes_field(self, key, values_dict):
        """Set node attributes for the key and values for each node.

        Parameters
        ----------
        key : str

        values_dict : dict of node_id: values

        """
        for node_id, value in values_dict.items():
            self.graph.nodes[node_id][key] = value


    def microstate_weights(self):
        """Returns the weights of each microstate on the basis of macrostates.

        Returns
        -------
        microstate_weights : dict of node_id: ndarray

        """

        node_weights = {}
        for node_id in self.graph.nodes:
            # get the trace of the frames in the node
            node_trace = self.node_assignments(node_id)

            # use the node_trace to get the weights from the HDF5
            with self.wepy_h5:
                trace_weights = self.wepy_h5.get_trace_fields(node_trace, ['weights'])['weights']

            node_weights[node_id] = trace_weights

        return node_weights

    def macrostate_weights(self):
        """Compute the total weight of each macrostate.

        Returns
        -------
        macrostate_weights : dict of node_id: float

        """

        macrostate_weights = {}
        microstate_weights = self.microstate_weights()
        for node_id, weights in microstate_weights.items():
            macrostate_weights[node_id] = float(sum(weights)[0])

        return macrostate_weights

    def set_macrostate_weights(self):
        """Compute the macrostate weights and set them as node attributes."""
        self.set_nodes_field('Weight', self.macrostate_weights())

    def state_to_mdtraj(self, node_id, alt_rep=None):
        """Generate an mdtraj.Trajectory object from a macrostate.

        By default uses the "main_rep" in the WepyHDF5
        object. Alternative representations of the topology can be
        specified.

        Parameters
        ----------
        node_id : node_id

        alt_rep : str
             (Default value = None)

        Returns
        -------
        traj : mdtraj.Trajectory

        """
        with self.wepy_h5:
            return self.wepy_h5.trace_to_mdtraj(self.node_assignments(node_id), alt_rep=alt_rep)

    def write_gexf(self, filepath):
        """Writes a graph file in the gexf format of the network.

        Parameters
        ----------
        filepath : str

        """

        # to do this we need to get rid of the assignments in the
        # nodes though since this is not really supported or good to
        # store in a gexf file which is more for visualization as an
        # XML format, so we copy and modify then write the copy
        gexf_graph = deepcopy(self._graph)
        for node in gexf_graph:
            del gexf_graph.nodes[node][self.ASSIGNMENTS]

        nx.write_gexf(gexf_graph, filepath)

    # TODO: need to implement these
    # def node_map(self, func, *args, map_func, idxs=False, node_sel=None):
    #     """Map a function over the nodes.

    #     Parameters
    #     ----------
    #     func :
            
    #     *args :
            
    #     map_func :
            
    #     idxs :
    #          (Default value = False)
    #     node_sel :
    #          (Default value = None)

    #     Returns
    #     -------

    #     """
    #     pass

    # def node_fields_map(self, func, fields, *args, map_func=map, idxs=False, node_sel=None):
    #     """

    #     Parameters
    #     ----------
    #     func :
            
    #     fields :
            
    #     *args :
            
    #     map_func :
    #          (Default value = map)
    #     idxs :
    #          (Default value = False)
    #     node_sel :
    #          (Default value = None)

    #     Returns
    #     -------

    #     """
    #     pass

    # def compute_macrostate_attr(self, func, fields, *args,
    #                             map_func=map,
    #                             node_sel=None,
    #                             idxs=False,
    #                             attr_name=None,
    #                             return_results=True):
    #     """

    #     Parameters
    #     ----------
    #     func :
            
    #     fields :
            
    #     *args :
            
    #     map_func :
    #          (Default value = map)
    #     node_sel :
    #          (Default value = None)
    #     idxs :
    #          (Default value = False)
    #     attr_name :
    #          (Default value = None)
    #     return_results :
    #          (Default value = True)

    #     Returns
    #     -------

    #     """
    #     pass

