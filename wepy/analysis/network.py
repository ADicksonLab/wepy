from wepy.analysis.tree import sliding_window
from wepy.analysis.transitions import run_transition_counts_matrix

class StateNetwork():

    def __init__(self, wepy_hdf5, assg_key, edge_matrices=None):

        self._wepy_hdf5 = wepy_hdf5

        # the key for the assignment in the wepy dataset
        self._assg_key = assg_key

        # make a dictionary that maps the assignment keys to frames
        # across the whole file {assignment_key : (run_idx, traj_idx, frame_idx)}
        self._assignments = {}
        for idx_tup, fields_d in self._wepy_hdf5.iter_trajs_fields([self.assg_key], idxs=True):
            run_idx = idx_tup[0]
            traj_idx = idx_tup[1]
            assg_field = fields_d[self.assg_key]

            for frame_idx, assg in enumerate(assg_field):
                self._assignments[assg].append( (run_idx, traj_idx, frame_idx) )


        if edge_matrices is None:
            self._edge_matrices = {}
        else:
            self._edge_matrices = edge_matrices


    @property
    def wepy_hdf5(self):
        return self._wepy_hdf5

    @property
    def assg_key(self):
        return self._assg_key

    @property
    def assignments(self):
        return self._assignments

    @property
    def edge_matrices(self):
        return self._edge_matrices

    def state_fields(self, state_label, fields):
        fields = self.wepy_hdf5.get_trace_fields(self.assignments[state_label], fields)
        return fields

    def state_to_mdtraj(self, state_label, alt_rep=None):
        return self.wepy_hdf5.trace_to_mdtraj(self.assignments[state_label], alt_rep=alt_rep)
