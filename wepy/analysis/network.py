from collections import defaultdict

import networkx as nx

from wepy.analysis.transitions import transition_counts, counts_d_to_matrix, \
                                      normalize_counts

class MacroStateNetwork(nx.DiGraph):

    def __init__(self, contig_tree, assg_field_key=None, assignments=None,
                 transition_lag_time=2):

        super().__init__()

        assert not (assg_field_key is None and assignments is None), \
            "either assg_field_key or assignments must be given"

        assert assg_field_key is not None or assignments is not None, \
            "one of assg_field_key or assignments must be given"

        self._contig_tree = contig_tree
        self._wepy_hdf5 = self._contig_tree.wepy_h5

        self._assg_field_key = None

        # the temporary assignments dictionary
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
        for assg_key, assigs in self._assignments.items():
            self.add_node(assg_key, assignments=assigs)


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

            # the keys of this are the edges, so we add them to our
            # network
            self.add_edges_from(counts_d.keys())

            # then we also want to get the transition probabilities so
            # we get the counts matrix and compute the probabilities
            self._countsmat = counts_d_to_matrix(counts_d)
            self._probmat = normalize_counts(self._countsmat)

            # then we add these attributes to the edges in the network
            for i, j in self.edges:
                self.edge[(i,j)]['counts'] = self._countsmat[i,j]
                self.edge[(i,j)]['probability'] = self._probmat[i,j]

        # then get rid of the assignments dictionary, this information
        # can be accessed from the network
        del self._assignments



        return transitions


    def _key_init(self, assg_field_key):
        # the key for the assignment in the wepy dataset
        self._assg_field_key = assg_field_key

        # make a dictionary that maps the assignment keys to frames
        # across the whole file {assignment_key : (run_idx, traj_idx, frame_idx)}
        self._assignments = defaultdict(list)

        for idx_tup, fields_d in self._wepy_hdf5.iter_trajs_fields([self.assg_field_key], idxs=True):
            run_idx = idx_tup[0]
            traj_idx = idx_tup[1]
            assg_field = fields_d[self.assg_field_key]

            for frame_idx, assg in enumerate(assg_field):
                self._assignments[assg].append( (run_idx, traj_idx, frame_idx) )

    def _assignments_init(self, assignments):

        self._assignments = defaultdict(list)

        for run_idx, run in enumerate(assignments):
            for traj_idx, traj in enumerate(run):
                for frame_idx, assignment in enumerate(traj):
                    self._assignments[assignment].append( (run_idx, traj_idx, frame_idx) )


    @property
    def contig_tree(self):
        return self._contig_tree

    @property
    def wepy_hdf5(self):
        return self._wepy_hdf5

    @property
    def assg_field_key(self):
        return self._assg_field_key

    @property
    def assg_values(self):
        return list(self._assignments.keys())

    @property
    def assignments(self):
        return self._assignments

    def state_fields(self, state_label, fields):
        fields = self.wepy_hdf5.get_trace_fields(self.assignments[state_label], fields)
        return fields

    def state_to_mdtraj(self, state_label, alt_rep=None):
        return self.wepy_hdf5.trace_to_mdtraj(self.assignments[state_label], alt_rep=alt_rep)
