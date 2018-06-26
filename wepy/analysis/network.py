from collections import defaultdict

import networkx as nx

class MacroStateNetwork(nx.DiGraph):

    def __init__(self, wepy_hdf5, assg_field_key=None, assignments=None,
                 transition_lag_time=1):

        super().__init__()

        assert not (assg_field_key is None and assignments is None), \
            "either assg_field_key or assignments must be given"

        assert assg_field_key is not None or assignments is not None, \
            "one of assg_field_key or assignments must be given"

        self._wepy_hdf5 = wepy_hdf5

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

        # then get rid of the assignments dictionary, this information
        # can be accessed from the network
        del self._assignments

        # now count the transitions between the states and set those
        # as the edges between nodes, this is a multistep process

        # 1. generate the resampling panel for the contig tree of all
        # the runs

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
