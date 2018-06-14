from collections import defaultdict

class MacroStateNetwork():

    def __init__(self, wepy_hdf5, assg_key=None, assignments=None):

        assert not (assg_key is None and assignments is None), \
            "either assg_key or assignments must be given"

        assert assg_key is not None or assignments is not None, \
            "one of assg_key or assignments must be given"

        self._wepy_hdf5 = wepy_hdf5

        if assg_key is not None:
            assert type(assg_key) == str, "assignment key must be a string"

            self._key_init(assg_key)

        else:
            self._assignments_init(assignments)

    def _key_init(self, assg_key):
        # the key for the assignment in the wepy dataset
        self._assg_key = assg_key

        # make a dictionary that maps the assignment keys to frames
        # across the whole file {assignment_key : (run_idx, traj_idx, frame_idx)}
        self._assignments = defaultdict(list)

        for idx_tup, fields_d in self._wepy_hdf5.iter_trajs_fields([self.assg_key], idxs=True):
            run_idx = idx_tup[0]
            traj_idx = idx_tup[1]
            assg_field = fields_d[self.assg_key]

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
    def assg_key(self):
        return self._assg_key

    @property
    def assignments(self):
        return self._assignments

    def state_fields(self, state_label, fields):
        fields = self.wepy_hdf5.get_trace_fields(self.assignments[state_label], fields)
        return fields

    def state_to_mdtraj(self, state_label, alt_rep=None):
        return self.wepy_hdf5.trace_to_mdtraj(self.assignments[state_label], alt_rep=alt_rep)
