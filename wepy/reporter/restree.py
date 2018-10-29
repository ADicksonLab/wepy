from wepy.reporter.reporter import ProgressiveFileReporter

from wepy.analysis.parents import resampling_panel, \
                                  parent_panel, net_parent_table,\
                                  parent_table_discontinuities

class ResTreeReporter(ProgressiveFileReporter):

    FILE_ORDER = ('gexf_restree_path',)

    SUGGESTED_EXTENSIONS = ('restree.gexf')

    def __init__(self, resampler=None,
                 decision_class=None,
                 boundary_condition_class=None,
                 spacing=5.0,
                 default_node_radius=3.0):

        assert resampler is not None, \
            "Must provide a resampler, this is used to get the correct records "
            "from resampling data and is not saved in this object"

        assert decision_class is not None, "must give the decision class"
        assert boundary_condition_class is not None, "must give the boundary condition class"


        # parameters for the layout
        self.spacing = spacing
        self.default_node_radius = default_node_radius

        # get the fields for the records of the resampling records, we
        # don't need to store the whole resampler
        self._resampling_record_field_names = resampler.resampling_record_field_names()


        # make a namedtuple record for these records
        self._ResamplingRecord = namedtuple('{}_Record'.format('Resampling'),
                        ['cycle_idx'] + self._resampling_record_field_names[run_record_key])

        # we do the same for the bounadry condition class, although we
        # also need the class itself
        self._bc_class = boundary_condition_class
        self._warping_record_field_names = self._bc_class.warping_record_field_names()

        self._WarpingRecord = namedtuple('{}_Record'.format('Warping'),
                                    ['cycle_idx'] + self._warping_record_field_names[run_record_key])



        # we need the decision class for getting parent child relationships
        self._decision_class = decision_class

        # initialize the parent table that will be generated as the
        # simulation progresses
        self._parent_table = []

        # also keep track of the weights of the walkers
        self._walker_weights = []

    @property
    def parent_table(self):
        return self._parent_table

    def report(self, cycle_idx=None,
               resampled_walkers=None,
               warp_data=None,
               progress_data=None,
               resampling_data=None):


        # we basically want to generate a new parent table from the
        # records we were just given and then add that to our existing
        # parent table

        # we first have to generate the resampling and warping records
        # though adding the cycle_idx

        resampling_records = []
        for datum in resampling_data:
            # for each datum (a dictionary) we get only the fields we
            # need for the records
            record_d = {datum[field_name] : value
                    for field_name, value in self._resampling_record_field_names}

            # then make a namedtuple record
            record = self._ResamplingRecord(cycle_idx=cycle_idx, **record_d)

            resampling_records.append(record)

        warping_records = []
        for datum in warping_data:
            # for each datum (a dictionary) we get only the fields we
            # need for the records
            record_d = {datum[field_name] : value
                    for field_name, value in self._warping_record_field_names}

            # then make a namedtuple record
            record = self._WarpingRecord(cycle_idx=cycle_idx, **record_d)

            warping_records.append(record)

        # get the weights of the resampled walkers since we will want
        # to plot them
        walker_weights = [walker.weight for walker in resampled_walkers]

        # add these to the collected weights
        self._walker_weights.append(walker_weights)

        # so we make a resampling panel from the records, then the
        # parent panel, and then the net parent table
        res_panel = resampling_panel(resampling_records, is_sorted=False)
        par_panel = parent_panel(self._decision_class, res_panel)
        parent_table = net_parent_table(par_panel)

        # then we get the discontinuites due to warping through
        # boundary conditions
        parent_table = parent_table_discontinuities(self._boundary_condition_class,
                                                    parent_table, warping_records)


        # add these to the main parent table
        self._parent_table.extend(parent_table)

        # now use this to create a new ParentForest without a WepyHDF5
        # file view
        parent_forest = ParentForest(parent_table=self._parent_table)

        # compute the size of the nodes to plot, which is the free_energy
        free_energies = []
        for cycle_weights in self._walker_weights:
            free_energys.append(geomm.free_energy(np.array(cycle_weights)))

        # put these into the parent forest graph as node attributes,
        # so we can get them out as a lookup table by node id for
        # assigning to nodes in the layout
        parent_forest.set_attrs_by_array('free_energy', free_energies)

        # get the free energies out as a dictionary, flattening it to
        # a single value. Set the nodes with None for their FE to a
        # small number.

        # The only ones that should be none are the roots which will
        # be 1.0 / N
        n_roots = len(self._parent_table[0])
        root_weight = 1.0 / n_roots

        node_fes = {}
        for node_id, fe_arr in layout_forest.get_node_attributes('free_energy').items():

            if fe_arr is None:
                node_fes[node_id] = root_weight
            else:
                node_fes[node_id] = fe_arr[0]


        # now we make the graph which will contain the layout
        # information
        layout_forest = LayoutGraph(parent_forest.graph)

        # we are going to output to the gexf format so we use the
        # pertinent methods

        # we set the sizes
        layout_forest.set_node_gexf_sizes(node_fes)

        # now we get to the part where we make the layout (positioning
        # of nodes), so we parametrize a layout engine
        tree_layout = ResamplingTreeLayout(spacing=self.spacing,
                                           node_radius=self.default_node_radius)


        node_coords = tree_layout.layout(parent_forest,
                                         node_radii=node_fes)

        layout_forest.set_node_gexf_positions(node_coords)


        # now we can write this graph to the gexf format to the file
        nx.write_gexf(layout_forest.viz_graph, self.gexf_restree_path)
