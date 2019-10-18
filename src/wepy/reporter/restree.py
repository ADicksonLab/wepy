"""Provides a reporter for generating graphical depictions of
resampling parent trees.
"""

from collections import namedtuple

import numpy as np
import networkx as nx
from matplotlib import cm

from geomm.free_energy import free_energy

from wepy.reporter.reporter import ProgressiveFileReporter
from wepy.analysis.network_layouts.tree import ResamplingTreeLayout
from wepy.analysis.network_layouts.layout_graph import LayoutGraph

from wepy.analysis.parents import resampling_panel, \
                                  parent_panel, net_parent_table,\
                                  ParentForest


class ResTreeReporter(ProgressiveFileReporter):
    """Reporter that generates resampling parent trees in the GEXF
    format."""

    FILE_ORDER = ('gexf_restree_path',)

    SUGGESTED_EXTENSIONS = ('restree.gexf',)

    MAX_PROGRESS_NORM = 1.0

    DISCONTINUOUS_NODE_SHAPE = 'square'
    """The shape of nodes that signify a discontinuity in the lineage."""

    DEFAULT_NODE_SHAPE = 'disc'
    """The shape of normal nodes that signify a continuity in the lineage."""

    def __init__(self,
                 resampler=None,
                 boundary_conditions=None,
                 row_spacing=None,
                 step_spacing=None,
                 default_node_radius=None,
                 progress_key=None,
                 max_progress_value=None,
                 colormap_name='plasma',
                 **kwargs):
        """Constructor for the ResTreeReporter.

        Parameters
        ----------

        resampler : Resampler
            Used to generate parental relations from resampling
            records.

        boundary_condition : BoundaryCondition, optional
            Used to determine discontinuities in the lineages.
             (Default value = None)

        row_spacing : float, default: 5.0
            Spacing between nodes in a single row in layout.

        step_spacing : float, default: 20.0
            Spacing between the rows of nodes in each step of the layout.

        default_node_radius : float, default: 1.0
            Default node radius to use.

        progress_key : str
            The key of the value in the progress records to use for
            coloring nodes in the tree.

        max_progress_value : float or None
            The maximum value to consider for progress values, if None
            no max value will be used.

        colormap_name : str, default: 'plasma'
            The name of the colormap to use from the matplotlib
            colormap library (i.e. 'matplotlib.cm.get_cmap')

        """

        assert resampler is not None, \
            "Must provide a resampler, this is used to get the correct records "\
            "from resampling data and is not saved in this object"

        assert boundary_conditions is not None, "must give the boundary condition class"

        # TODO: this should be made optional
        assert progress_key is not None, "Currently, a progress key must be given."

        super().__init__(**kwargs)

        # parameters for the layout
        self.row_spacing = row_spacing
        self.step_spacing = step_spacing
        self.default_node_radius = default_node_radius

        # this is the key we will use to get values for the progress out
        self.progress_key = progress_key
        self.max_progress_value = max_progress_value
        # get the actual colormap that will be used from matplotlib
        self.colormap = cm.get_cmap(name=colormap_name)

        # we need the decision class for getting parent child
        # relationships
        self._decision_class = resampler.decision

        # get the fields for the records of the resampling records, we
        # don't need to store the whole resampler
        self._resampling_record_field_names = resampler.resampling_record_field_names()

        # we also will need to know the shape and dtype of them
        self._resampling_field_names = resampler.resampling_field_names()
        self._resampling_field_shapes = resampler.resampling_field_shapes()
        self._resampling_field_dtypes = resampler.resampling_field_dtypes()


        # make a namedtuple record for these records
        # self._ResamplingRecord = namedtuple('{}_Record'.format('Resampling'),
        #                 ['cycle_idx'] + list(self._resampling_record_field_names))

        # we do the same for the bounadry condition class, although we
        # also need the class itself
        self._bc = boundary_conditions
        self._warping_record_field_names = self._bc.warping_record_field_names()

        self._warping_field_names = self._bc.warping_field_names()
        self._warping_field_shapes = self._bc.warping_field_shapes()
        self._warping_field_dtypes = self._bc.warping_field_dtypes()

        # self._WarpingRecord = namedtuple('{}_Record'.format('Warping'),
        #                             ['cycle_idx'] + list(self._warping_record_field_names))



        # initialize the parent table that will be generated as the
        # simulation progresses
        self._parent_table = []

        # the node ids of the warped nodes (ones with -1 parents)
        self._discontinuous_nodes = []

        # also keep track of the weights of the walkers
        self._walker_weights = []

        # and the progress values for each walker
        self._walkers_progress = []

    @property
    def parent_table(self):
        """The net parent table datastructure."""
        return self._parent_table

    def _make_resampling_record(self, record_d, cycle_idx):
        """Make a namedtuple resampling record from a dictionary
        representation.

        Parameters
        ----------
        record_d : dict of str : value
            The dictionary of values to make a namedtuple record from.
        cycle_idx : int
            The cycle index the record is associated with.

        Returns
        -------

        record : namedtuple

        """

        # make a namedtuple record for these records
        ResamplingRecord = namedtuple('{}_Record'.format('Resampling'),
                        ['cycle_idx'] + list(self._resampling_record_field_names))


        record = self._make_record(record_d, cycle_idx,
                                   self._resampling_record_field_names,
                                   self._resampling_field_names, self._resampling_field_shapes,
                                   ResamplingRecord)

        return record

    def _make_warping_record(self, record_d, cycle_idx):
        """Make a namedtuple warping record from a dictionary representation.

        Parameters
        ----------
        record_d : dict of str : value
            The dictionary of values to make a namedtuple record from.
        cycle_idx : int
            The cycle index the record is associated with.

        Returns
        -------

        record : namedtuple

        """

        # make a namedtuple record for these records
        WarpingRecord = namedtuple('{}_Record'.format('Warping'),
                                   ['cycle_idx'] + list(self._warping_record_field_names))

        record = self._make_record(record_d, cycle_idx,
                                   self._warping_record_field_names,
                                   self._warping_field_names, self._warping_field_shapes,
                                   WarpingRecord)

        return record


    @staticmethod
    def _make_record(record_d, cycle_idx,
                     record_field_names, field_names, field_shapes, rec_namedtuple):
        """Generic record making function.

        Parameters
        ----------
        record_d : dict of str : value

        cycle_idx : int

        record_field_names : list of str
            The record field names to use for the record.

        field_names : list of str
            All of the field names available for the field type.

        field_shapes : list of tuple of int
            Shapes for each field

        rec_namedtuple : namedtuple object
            The namedtuple class to make records out of

        Returns
        -------

        record : namedtuple object

        """

        # go through each field of the record
        rec_d = {'cycle_idx' : cycle_idx}
        for field_name, field_value in record_d.items():

            field_idx = field_names.index(field_name)
            field_shape = field_shapes[field_idx]

            # if it is not part of the record we just keep going and
            # don't bother
            if field_name not in record_field_names:
                continue

            # otherwise we need to do some formatting
            else:

                # if it is variable length (shape is Ellipsis) or if
                # it has more than one element cast all elements to
                # tuples
                if field_shape is Ellipsis:
                    value = tuple(field_value)

                # if it is not variable length make sure it is not more than a
                # 1D feature vector
                elif len(field_shape) > 2:
                    raise TypeError(
                        "cannot convert fields with feature vectors more than 1 dimension,"
                        " was given {} for {}".format(
                            field_value.shape[1:], field_name))

                # if it is only a rank 1 feature vector and it has more than
                # one element make a tuple out of it
                elif field_shape[0] > 1:
                    value = tuple(field_value)

                # otherwise just get the single value instead of keeping it as
                # a single valued feature vector
                else:
                    value = field_value[0]

            rec_d[field_name] = value

        # after you make the dictionary of this convert it to a
        # namedtuple record
        record = rec_namedtuple(*(rec_d[key] for key in rec_namedtuple._fields))

        return record


    def report(self, cycle_idx=None,
               resampled_walkers=None,
               warp_data=None,
               progress_data=None,
               resampling_data=None,
               **kwargs):
        """Generate the resampling tree GEXF file.

        Parameters
        ----------
        cycle_idx : int

        resampled_walkers : list of Walker

        warp_data : dict of str : value

        progress_data : dict of str : value

        resampling_data : dict of str : value

        """


        # we basically want to generate a new parent table from the
        # records we were just given and then add that to our existing
        # parent table

        # we first have to generate the resampling and warping records
        # though adding the cycle_idx

        resampling_records = [self._make_resampling_record(rec_d, cycle_idx)
                              for rec_d in resampling_data]

        warping_records = [self._make_warping_record(rec_d, cycle_idx)
                              for rec_d in warp_data]

        # tabulate the discontinuities
        for warping_record in warping_records:
            # if this record classifies as discontinuous
            if self._bc.warping_discontinuity(warping_record):
                # then we save it as one of the nodes that is discontinuous
                disc_node_id = (warping_record.cycle_idx, warping_record.walker_idx)
                self._discontinuous_nodes.append(disc_node_id)

        # get the weights of the resampled walkers since we will want
        # to plot them
        walker_weights = [walker.weight for walker in resampled_walkers]
        # add these to the collected weights
        self._walker_weights.append(walker_weights)

        # get the progress values for all of the walkers
        walkers_progress = [progress for progress in progress_data[self.progress_key]]
        self._walkers_progress.append(walkers_progress)

        # so we make a resampling panel from the records

        # TODO: this uses the cycle_idx in the records and thus wehn
        # you receive just one cycle it adds a empty records to the
        # beginning of it and only the last entry is the one we are
        # looking for so for now we just get it, and wrap back in a list
        res_panel = [resampling_panel(resampling_records, is_sorted=False)[-1]]

        # then the parent panel, and then the net parent table
        par_panel = parent_panel(self._decision_class, res_panel)
        cycle_parent_table = net_parent_table(par_panel)

        # add these to the main parent table
        self._parent_table.extend(cycle_parent_table)

        # now use this to create a new ParentForest without a WepyHDF5
        # file view. We exclude the discontinuities on purpose because
        # we want to keep the paren child relationships through the
        # warps for visualization and will just change the shapes of
        # the nodes to communicate the warping
        parent_forest = ParentForest(parent_table=self._parent_table)

        # The only ones that should be none are the roots which will
        # be 1.0 / N
        n_roots = len(self._parent_table[0])
        root_weight = 1.0 / n_roots

        # compute the size of the nodes to plot, which is the
        # free_energy. To compute this we flatten the list of list of
        # weights to a 1-d array then unravel it back out

        # put the root weights at the beginning of this array, and
        # account for it in the flattening
        flattened_weights = [root_weight for _ in self._walker_weights[0]]
        cycles_n_walkers = [len(self._walker_weights[0])]
        for cycle_weights in self._walker_weights:
            cycles_n_walkers.append(len(cycle_weights))
            flattened_weights.extend(cycle_weights)

        # now compute the free energy
        flattened_free_energies = free_energy(np.array(flattened_weights))

        # and ravel them back into a list of lists
        free_energies = []
        last_index = 0
        for cycle_n_walkers in cycles_n_walkers:
            free_energies.append(flattened_free_energies[last_index:last_index + cycle_n_walkers])
            last_index += cycle_n_walkers

        # we want to set the colors based on the progress, first we
        # scale all the progress values to a 1-100 scale
        if self.max_progress_value is not None:
            norm_ratio = self.MAX_PROGRESS_NORM / self.max_progress_value

        # otherwise we use the maximum value from the full progress
        # values dataset so far
        else:
            max_progress_value = max([max(row) for row in self._walkers_progress])
            # then use that for the ratio
            norm_ratio = self.MAX_PROGRESS_NORM / max_progress_value

        # now that we have ratios for normalizing values we apply that
        # and then use the lookup table for the color bar to get the
        # RGB color values
        colors = []
        for progress_row in self._walkers_progress:
            color_row = [self.colormap(progress * norm_ratio, bytes=True)
                         for progress in progress_row]
            colors.append(color_row)

        # put these into the parent forest graph as node attributes,
        # so we can get them out as a lookup table by node id for
        # assigning to nodes in the layout. We skip the first row of
        # free energies for the roots in this step
        parent_forest.set_attrs_by_array('free_energy', free_energies[1:])
        parent_forest.set_attrs_by_array('color', colors)

        # get the free energies out as a dictionary, flattening it to
        # a single value. Set the nodes with None for their FE to a
        # small number.

        # now we make the graph which will contain the layout
        # information
        layout_forest = LayoutGraph(parent_forest.graph)


        # get the free energy attributes out using the root weight for
        # the nodes with no fe defined

        # start it with the root weights
        node_fes = {(-1, i) : free_energy for i, free_energy in enumerate(free_energies[0])}
        for node_id, fe_arr in layout_forest.get_node_attributes('free_energy').items():

            if fe_arr is None:
                node_fes[node_id] = 0.0
            else:
                node_fes[node_id] = fe_arr

        # the default progress for the root ones is 0 and so the color
        # is also 0
        node_colors = {}
        for node_id, color_arr in layout_forest.get_node_attributes('color').items():

            if color_arr is None:
                # make a black color for these
                node_colors[node_id] = tuple(int(255) for a in range(4))
            else:
                node_colors[node_id] = color_arr

        # make a dictionary for the shapes of the nodes
        node_shapes = {}
        # set all the nodes to a default 'disc'
        for node in layout_forest.viz_graph.nodes:
            node_shapes[node] = self.DEFAULT_NODE_SHAPE
        # then set all the discontinuous ones
        for discontinuous_node in self._discontinuous_nodes:
            node_shapes[discontinuous_node] = self.DISCONTINUOUS_NODE_SHAPE

        # also set the color to black
        for discontinuous_node in self._discontinuous_nodes:
            node_colors[discontinuous_node] = tuple(int(0) for a in range(4))


        # we are going to output to the gexf format so we use the
        # pertinent methods

        # we set the sizes
        layout_forest.set_node_gexf_sizes(node_fes)

        # and set the colors based on the progresses

        layout_forest.set_node_gexf_colors_rgba(node_colors)

        # then set the shape of the nodes based on whether they were
        # warped or not
        layout_forest.set_node_gexf_shape(node_shapes)


        # now we get to the part where we make the layout (positioning
        # of nodes), so we parametrize a layout engine
        tree_layout = ResamplingTreeLayout(row_spacing=self.row_spacing,
                                           step_spacing=self.step_spacing,
                                           node_radius=self.default_node_radius)


        node_coords = tree_layout.layout(parent_forest,
                                         node_radii=node_fes)

        layout_forest.set_node_gexf_positions(node_coords)


        # now we can write this graph to the gexf format to the file
        nx.write_gexf(layout_forest.viz_graph, self.gexf_restree_path)
