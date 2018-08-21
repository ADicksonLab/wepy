from collections import defaultdict
from copy import deepcopy

import networkx as nx

from wepy.analysis.transitions import transition_counts, counts_d_to_matrix, \
                                      normalize_counts

class MacroStateNetworkError(Exception):
    pass

class MacroStateNetwork():


    ASSIGNMENTS = 'assignments'

    def __init__(self, contig_tree, assg_field_key=None, assignments=None,
                 transition_lag_time=2):

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

        # the key for the assignment in the wepy dataset
        self._assg_field_key = assg_field_key

        # blank assignments
        assignments = [[[] for traj_idx in range(self._wepy_h5.n_run_trajs(run_idx))]
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

        # set the raw assignments to the temporary attribute
        self._assignments = assignments

        # this is the dictionary mapping node_id -> the (run_idx, traj_idx, cycle_idx) frames
        self._node_assignments = defaultdict(list)

        for run_idx, run in enumerate(assignments):
            for traj_idx, traj in enumerate(run):
                for frame_idx, assignment in enumerate(traj):
                    self._node_assignments[assignment].append( (run_idx, traj_idx, frame_idx) )

    def node_id_to_idx(self, assg_key):
        return self.node_id_to_idx_dict()[assg_key]

    def node_idx_to_id(self, node_idx):
        return self.node_idx_to_id_dict()[node_idx]

    def node_id_to_idx_dict(self):
        return self._node_idxs

    def node_idx_to_id_dict(self):
        # just reverse the dictionary and return
        return {node_idx : node_id for node_id, node_idx in self._node_idxs}

    @property
    def graph(self):
        return self._graph

    @property
    def contig_tree(self):
        return self._contig_tree

    @property
    def wepy_h5(self):
        return self._wepy_h5

    @property
    def assg_field_key(self):
        return self._assg_field_key

    @property
    def countsmat(self):
        try:
            return self._countsmat
        except AttributeError:
            raise MacroStateNetworkError("transition counts matrix not calculated")

    @property
    def probmat(self):
        try:
            return self._probmat
        except AttributeError:
            raise MacroStateNetworkError("transition probability matrix not set")

    def get_node_attributes(self, node_id):
        pass

    def get_node_attribute(self, node_id, attribute_key):
        pass

    def node_assignments(self, node_id):
        return self.graph.nodes[node_id][self.ASSIGNMENTS]

    def get_node_fields(self, node_id, fields):
        node_trace = self.node_assignments(node_id)

        # use the node_trace to get the weights from the HDF5
        fields_d = self.wepy_h5.get_trace_fields(node_trace, fields)

        return fields_d

    def iter_nodes_fields(self, fields):

        nodes_d = {}
        for node_id in self.graph.nodes:
            fields_d = self.get_node_fields(node_id, fields)
            nodes_d[node_id] = fields_d

        return nodes_d



    def set_nodes_field(self, key, values_dict):
        for node_id, value in values_dict.items():
            self.graph.nodes[node_id][key] = value

    def node_map(self, func, *args, map_func, idxs=False, node_sel=None):
        pass

    def node_fields_map(self, func, fields, *args, map_func=map, idxs=False, node_sel=None):
        pass

    def compute_macrostate_attr(self, func, fields, *args,
                                map_func=map,
                                node_sel=None,
                                idxs=False,
                                attr_name=None,
                                return_results=True):
        pass


    def microstate_weights(self):
        """Calculates and returns the sums of the weights of all the nodes as
        a dictionary mapping node_id -> frame weights"""

        node_weights = {}
        for node_id in self.graph.nodes:
            # get the trace of the frames in the node
            node_trace = self.node_assignments(node_id)

            # use the node_trace to get the weights from the HDF5
            trace_weights = self.wepy_h5.get_trace_fields(node_trace, ['weights'])['weights']

            node_weights[node_id] = trace_weights

        return node_weights

    def macrostate_weights(self):

        macrostate_weights = {}
        microstate_weights = self.microstate_weights()
        for node_id, weights in microstate_weights.items():
            macrostate_weights[node_id] = float(sum(weights)[0])

        return macrostate_weights

    def set_macrostate_weights(self):
        self.set_nodes_field('Weight', self.macrostate_weights())

    def state_to_mdtraj(self, node_id, alt_rep=None):
        return self.wepy_h5.trace_to_mdtraj(self.node_assignments(node_id), alt_rep=alt_rep)

    def write_gexf(self, filepath):

        # to do this we need to get rid of the assignments in the
        # nodes though since this is not really supported or good to
        # store in a gexf file which is more for visualization as an
        # XML format, so we copy and modify then write the copy
        gexf_graph = deepcopy(self._graph)
        for node in gexf_graph:
            del gexf_graph.nodes[node][self.ASSIGNMENTS]

        nx.write_gexf(gexf_graph, filepath)
