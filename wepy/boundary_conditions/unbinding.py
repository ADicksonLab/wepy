from wepy.boundary_conditions.boundary import BoundaryConditions

UNBINDING_INSTRUCT_DTYPE = [('target', int)]

class UnbindingBC(BoundaryConditions):

    WAPR_INSTRUCT_DTYPE = UNBINDING_INSTRUCT_DTYPE

    def __init__(self, initial_state=None,
                 cutoff_distance=1.0,
                 topology=None,
                 ligand_idxs=None,
                 binding_site_idxs=None):

        # test input
        assert initial_state is not None, "Must give an initial state"
        assert topology is not None, "Must give a reference topology"
        assert ligand_idxs is not None
        assert binding_site_idxs is not None
        assert type(cutoff_distance) is float

        self.initial_state = initial_state
        self.cutoff_distance = cutoff_distance
        self.topology = topology

        self.ligand_idxs = ligand_idxs
        self.binding_site_idxs = binding_site_idxs

    def _calc_angle(self, v1, v2):
        return np.degrees(np.arccos(np.dot(v1, v2)/(la.norm(v1) * la.norm(v2))))

    def _calc_length(self, v):
        return la.norm(v)


    def _pos_to_array(self, positions):
        n_atoms = self.topology.n_atoms

        xyz = np.zeros((1, n_atoms, 3))

        for i in range(n_atoms):
            xyz[0,i,:] = ([positions[i]._value[0],
                           positions[i]._value[1],
                           positions[i]._value[2]])
        return xyz

    def _calc_min_distance(walker):
        # convert box_vectors to angles and lengths for mdtraj
        # calc box length
        cell_lengths = np.array([[self._calc_length(v._value) for v in walker.box_vectors]])

        # TODO order of cell angles
        # calc angles
        cell_angles = np.array([[self._calc_angle(walker.box_vectors._value[i],
                                                 walker.box_vectors._value[j])
                                 for i, j in [(0,1), (1,2), (2,0)]]])

        # make a traj out of it so we can calculate distances through
        # the periodic boundary conditions
        walker_traj = mdj.Trajectory(self._pos_to_array(walker.positions[0:self.topology.n_atoms]),
                                     topology=self.topology,
                                     unitcell_lengths=cell_lengths,
                                     unitcell_angles=cell_angles)

        # calculate the distances through periodic boundary conditions
        # and get hte minimum distance
        min_distance = np.min(mdj.compute_distances(walker_traj,
                                                    it.product(self.ligand_idxs,
                                                               self.binding_site_idxs)))
        return min_distance

    def check_boundaries(self, walker):

        min_distance = self._calc_min_distance(walker)

        # test to see if the ligand is unbound
        unbound = False
        if min_distance >= self.cutoff_distance:
            unbound = True

        return unbound, min_distance

    def warp_walkers(self, walkers):

        new_walkers = []
        warped_walkers_records = []
        min_distances = []

        for walker_idx, walker in enumerate(walkers):
            unbinding, min_distance = self.check_boundaries(walker)
            min_distances.append(min_distance)
            if unbinding:
               warped_walkers_records.append( (walker_idx, (0,)) )
               new_walkers.append(self.initial_state)
            else:
                new_walkers.append(walker)

        data = {'min_distances' : np.array(min_distances)}
        return new_walkers, warped_walkers_records, data
