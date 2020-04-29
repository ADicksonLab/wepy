"""Classes providing a contig and branching contigs abstractions over
the underlying WepyHDF5 simulation data store.

Routines
--------

ContigTree
Contig
"""


import itertools as it
from copy import copy
from operator import attrgetter
from collections import deque

import networkx as nx
import numpy as np
from matplotlib import cm

from geomm.free_energy import free_energy as calc_free_energy

from wepy.analysis.parents import (
    DISCONTINUITY_VALUE,
    parent_panel, net_parent_table,
    ancestors, sliding_window,
    parent_cycle_discontinuities,
    ParentForest
)

from wepy.analysis.network_layouts.tree import ResamplingTreeLayout
from wepy.analysis.network_layouts.layout_graph import LayoutGraph

# optional dependencies
try:
    import pandas as pd
except ModuleNotFoundError:
    warn("pandas is not installed and that functionality will not work", RuntimeWarning)


# the groups of run records
RESAMPLING = 'resampling'
"""Record key for resampling records."""
RESAMPLER = 'resampler'
"""Record key for resampler records."""
WARPING = 'warping'
"""Record key for warping records."""
PROGRESS = 'progress'
"""Record key for progress records."""
BC = 'boundary_conditions'
"""Record key for boundary condition records."""

class BaseContigTree():
    """A base class for the contigtree which doesn't contain a WepyHDF5
    object. Useful for serialization of the object and can then be
    reattached later to a WepyHDF5.

    """

    RESAMPLING_PANEL_KEY = 'resampling_steps'
    """Key for resampling panel node attributes in the tree graph."""
    PARENTS_KEY = 'parent_idxs'
    """Key for parent indices node attributes in the tree graph."""
    DISCONTINUITY_KEY = 'discontinuities'
    """Key for discontinuity node attributes in the tree graph."""

    def __init__(self, wepy_h5,
                 continuations=Ellipsis,
                 runs=Ellipsis,
                 boundary_condition_class=None,
                 decision_class=None):
        """The only required argument is an WepyHDF5 object from which to draw
        data.

        If `continuations` is given it specifies the (continuing_run_idx,
        continued_run_idx) continuations the contig tree will use. This
        overrides anything in the WepyHDF5 file. Only use this if you know
        what you are doing. If `continuations` is `Ellipsis` the
        continuations from the `WepyHDF5` will be used.

        If `runs` is not `Ellipsis` only these runs will be included in
        the `ContigTree` and the continuations relvant to this subset will
        be used.

        If a valid `decision_class` is given walker lineages will be
        generated. This class is largely useless without this.

        The `boundary_condition_class` is only used in the presence of a
        `decision_class`. A valid `boundary_condition_class` will detect
        the discontinuities in walker trajectories and will automatically
        apply them to divide up logical trajectories. Otherwise, if
        discontinuous boundary condition warping events in data are
        present logical trajectories will have discontinuities in them.


        Parameters
        ----------
        wepy_h5 : closed WepyHDF5 object

        continuations : Ellipsis (use continuations in wepy_h5) or
                        list of tuple of int (run_idx, run_idx)

            The continuations to apply to the runs in this contig tree.
             (Default value = Ellipsis)

        runs : Ellipsis (use all runs in wepy_h5) or
               list of idx

            The indices of the runs to use in the contig tree
             (Default value = Ellipsis)

        boundary_condition_class : class implementing BoundaryCondition
                                   interface
             (Default value = None)


        decision_class : class implementing Decision interface
             (Default value = None)


        Warnings
        --------

        Only set `continuations` if you know what you are doing.

        A `decision_class` must be given to be able to detect cloning and
        merging and get parental lineages.

        Make sure to give the `boundary_condition_class` if there was
        discontinuous warping events in the simulation or else you will
        get erroneous contiguous trajectories.
        """

        was_closed = False
        if wepy_h5.closed:
            was_closed = True
            wepy_h5.open()


        self._graph = nx.DiGraph()

        self._boundary_condition_class=boundary_condition_class
        self._decision_class = decision_class

        # we can optionally specify which continuations to use when
        # creating the contig tree instead of defaulting to the whole file
        self._continuations = set()
        self._run_idxs = set()

        # if specific runs were specified we add them right away, they
        # should be unique
        if runs is Ellipsis:
            self._run_idxs.update(wepy_h5.run_idxs)
        elif runs is not None:
            self._run_idxs.update(runs)

        # the continuations also give extra runs to incorporate into
        # this contig tree

        # if it is Ellipsis (...) then we include all runs and all the continuations
        if continuations is Ellipsis:
            self._run_idxs.update(wepy_h5.run_idxs)
            self._continuations.update([(a,b) for a, b in wepy_h5.continuations])

        # otherwise we make the tree based on the runs in the
        # continuations
        elif continuations is not None:
            # the unique run_idxs
            self._run_idxs.update(it.chain(*self._continuations))

            # the continuations themselves
            self._continuations.update([(a,b) for a, b in continuations])


        # using the wepy_h5 create a tree of the cycles
        self._create_tree(wepy_h5)

        self._set_resampling_panels(wepy_h5)

        if self._decision_class is not None:
            self._set_parents(self._decision_class)

            if self._boundary_condition_class is not None:
                self._set_discontinuities(wepy_h5, self._boundary_condition_class)

        # set the spanning contigs as a mapping of the index to the
        # span trace
        self._spans = {span_idx : span_trace
                       for span_idx, span_trace
                       in enumerate(self.spanning_contig_traces())}

        if was_closed:
            wepy_h5.close()

    @property
    def graph(self):
        """The underlying networkx.DiGraph object. """
        return self._graph

    @property
    def decision_class(self):
        """The decision class used to determine parental lineages. """
        return self._decision_class

    @property
    def boundary_condition_class(self):
        """The boundary condition class is used to determine discontinuities in lineages. """
        return self._boundary_condition_class

    @property
    def span_traces(self):
        """Dictionary mapping the spand indices to their run traces."""
        return self._spans

    def span_contig(self, span_idx):
        """Generates a contig object for the specified spanning contig."""

        contig = self.make_contig(self.span_traces[span_idx])

        return contig

    def _create_tree(self, wepy_h5):
        """Generate the tree of cycles from the WepyHDF5 object/file. """

        # first go through each run without continuations
        for run_idx in self._run_idxs:
            n_cycles = wepy_h5.num_run_cycles(run_idx)

            # make all the nodes for this run
            nodes = [(run_idx, step_idx) for step_idx in range(n_cycles)]
            self.graph.add_nodes_from(nodes)

            # the same for the edges
            edge_node_idxs = list(zip(range(1, n_cycles), range(n_cycles - 1)))

            edges = [(nodes[a], nodes[b]) for a, b in edge_node_idxs]
            self.graph.add_edges_from(edges)

        # after we have added all the nodes and edges for the run
        # subgraphs we need to connect them together with the
        # information in the contig tree.
        for edge_source, edge_target in self._continuations:

            # for the source node (the restart run) we use the run_idx
            # from the edge source node and the index of the first
            # cycle
            source_node = (edge_source, 0)

            # for the target node (the run being continued) we use the
            # run_idx from the edge_target and the last cycle index in
            # the run
            target_node = (edge_target, wepy_h5.num_run_cycles(edge_target)-1)

            # make the edge
            edge = (source_node, target_node)

            # add this connector edge to the network
            self.graph.add_edge(*edge)

    def _set_resampling_panels(self, wepy_h5):
        """Generates resampling panels for each cycle and sets them as node attributes. """

        # then get the resampling tables for each cycle and put them
        # as attributes to the appropriate nodes
        for run_idx in self.run_idxs:

            run_resampling_panel = wepy_h5.run_resampling_panel(run_idx)

            # add each cycle of this panel to the network by adding
            # them in as nodes with the resampling steps first
            for step_idx, step in enumerate(run_resampling_panel):
                node = (run_idx, step_idx)
                self.graph.nodes[node][self.RESAMPLING_PANEL_KEY] = step

    def _set_discontinuities(self, wepy_h5, boundary_conditions_class):
        """Given the boundary condition class sets node attributes for where
        there are discontinuities in the parental lineages.

        Parameters
        ----------
        boundary_conditions_class : class implementing BoundaryCondition interface

        """

        # initialize the attributes for discontinuities to 0s for no
        # discontinuities
        for node in self.graph.nodes:
            n_walkers = len(self.graph.node[node][self.PARENTS_KEY])
            self.graph.node[node][self.DISCONTINUITY_KEY] = [0 for i in range(n_walkers)]

        #
        for run_idx in self.run_idxs:

            # get the warping records for this run
            warping_records = wepy_h5.warping_records([run_idx])

            # just the indices for checking stuff later
            warp_cycle_idxs = set([rec[0] for rec in warping_records])

            # go through the nodes
            for node in self.graph.nodes:
                node_run_idx = node[0]
                node_cycle_idx = node[1]

                # for a node which is in this run and has warp records
                if (node_run_idx == run_idx) and (node_cycle_idx in warp_cycle_idxs):

                    # if there is then we want to apply the
                    # warping records for this cycle to the
                    # discontinuities for this cycle
                    cycle_warp_records = [rec for rec in warping_records
                                          if (rec[0] == node_cycle_idx)]

                    # go through each record and test if it is a
                    # discontinuous warp
                    for rec in cycle_warp_records:

                        # index of the trajectory this warp effected
                        rec_traj_idx = rec[1]

                        # if it is discontinuous we need to mark that,
                        # otherwise do nothing
                        if boundary_conditions_class.warping_discontinuity(rec):

                            self.graph.node[node][self.DISCONTINUITY_KEY][rec_traj_idx] = -1

    def _set_parents(self, decision_class):
        """Determines the net parents for each cycle and sets them in-place to
        the cycle tree given.

        Parameters
        ----------
        decision_class : class implementing Decision interface

        """

        # just go through each node individually in the tree
        for node in self.graph.nodes:
            # get the records for each step in this node
            node_recs = self.graph.node[node][self.RESAMPLING_PANEL_KEY]

            # get the node parent table by using the parent panel method
            # on the node records
            node_parent_panel = parent_panel(decision_class, [node_recs])
            # then get the net parents from this parent panel, and slice
            # out the only entry from it
            node_parents = net_parent_table(node_parent_panel)[0]

            # put this back into the self
            self.graph.nodes[node][self.PARENTS_KEY] = node_parents

    @property
    def run_idxs(self):
        """Indices of runs in WepyHDF5 used in this contig tree. """
        return self._run_idxs

    @property
    def continuations(self):
        """The continuations that are used in this contig tree over the runs. """
        return self._continuations


    @staticmethod
    def contig_trace_to_run_trace(contig_trace, contig_walker_trace):
        """Combine a contig trace and a walker trace to get the equivalent run trace.

        The contig_walker_trace cycle_idxs must be a subset of the
        frame indices given by the contig_trace.


        Parameters
        ----------
        contig_trace : list of tuples of ints (run_idx, cycle_idx)

        contig_walker_trace : list of tuples of ints (traj_idx, cycle_idx)

        Returns
        -------

        run_trace : list of tuples of ints (run_idx, traj_idx, cycle_idx)

        """


        trace = []
        for traj_idx, contig_cycle_idx in contig_walker_trace:

            try:
                run_idx, run_cycle_idx = contig_trace[contig_cycle_idx]
            except IndexError:
                raise ValueError("Invalid cycle_idx in the contig_walker_trace. "
                                 "Must be an index over the frames given by contig_trace.")

            frame = (run_idx, traj_idx, run_cycle_idx)
            trace.append(frame)

        return trace


    def walker_trace_to_run_trace(self, contig_walker_trace):
        """Combine a walker trace to get the equivalent run trace for this contig.

        The contig_walker_trace cycle_idxs must be a subset of the
        frame indices given by the contig_trace.


        Parameters
        ----------

        contig_walker_trace : list of tuples of ints (traj_idx, cycle_idx)

        Returns
        -------

        run_trace : list of tuples of ints (run_idx, traj_idx, cycle_idx)

        See Also
        --------
        Contig.contig_trace_to_run_trace : calls this static method

        """

        return self.contig_trace_to_run_trace(self.contig_trace, contig_walker_trace)


    def run_trace_to_contig_trace(self, run_trace):
        """

        Assumes that the run trace goes along a valid contig.

        Parameters
        ----------

        run_trace : list of tuples of ints (run_idx, traj_idx, cycle_idx)

        Returns
        -------

        contig_walker_trace : list of tuples of ints (traj_idx, contig_cycle_idx)


        """

        contig_walker_trace = []
        for run_idx, traj_idx, cycle_idx in run_trace:

            # get the contig cycle index given the branch we are on
            contig_cycle_idx = self.contig_cycle_idx(run_idx, cycle_idx)

            contig_walker_trace.append((traj_idx, contig_cycle_idx))

        return contig_walker_trace


    def contig_cycle_idx(self, run_idx, cycle_idx):
        """ Convert an in-run cycle index to an in-contig cyle_idx.

        Parameters
        ----------
        run_idx : int
            Index of a run in the contig tree

        cycle_idx : int
            Index of a cycle index within a run

        Returns
        -------

        contig_cycle_idx : int
            The cycle idx in the contig

        """

        # make the contig trace
        contig_trace = self.get_branch_trace(run_idx, cycle_idx)

        # get the length and subtract one for the index
        return len(contig_trace) - 1


    def get_branch_trace(self, run_idx, cycle_idx, start_contig_idx=0):
        """Get a contig trace for a branch of the contig tree from an end
        point back to a set point (defaults to root of contig tree).

        Parameters
        ----------
        run_idx : int

        cycle_idx : int
            An in-contig cycle index

        start_contig_idx : int, optional
            The in-contig cycle index where the "root" of the branch will start
             (Default value = 0)

        Returns
        -------

        contig_trace : list of tuples of ints (run_idx, cycle_idx)

        """

        assert start_contig_idx >= 0, "start_contig_idx must be a valid index"

        # initialize the current node
        curr_node = (run_idx, cycle_idx)

        # make a trace of this contig
        contig_trace = [curr_node]

        # stop the loop when we reach the root of the cycle tree
        at_root = False
        while not at_root:

            # first we get the cycle node previous to this one because it
            # has the parents for the current node
            parent_nodes = list(self.graph.adj[curr_node].keys())

            # if there is no parent then this is the root of the
            # cycle_tree and we should stop after this step of the loop
            if len(parent_nodes) == 0:
                at_root = True

            # or else we put the node into the contig trace
            else:
                parent_node = parent_nodes[0]
                contig_trace.insert(0, parent_node)

                # if there are any parents there should only be one, since it is a tree
                prev_node = parent_nodes[0]

                # then update the current node to the previous one
                curr_node = prev_node

        return contig_trace

    def trace_parent_table(self, contig_trace, discontinuities=True):
        """Given a contig trace returns a parent table for that contig.

        Parameters
        ----------
        contig_trace : list of tuples (run_idx, cycle_idx)

        discontinuities : bool
           Whether or not to include discontinuities in the table.

        Returns
        -------
        parent_table : list of list of int

        """

        parent_table = []
        for run_idx, cycle_idx in contig_trace:

            parent_row = self.graph.node[(run_idx, cycle_idx)][self.PARENTS_KEY]

            if discontinuities:

                discs = self.graph.node[(run_idx, cycle_idx)][self.DISCONTINUITY_KEY]

                parent_row = parent_cycle_discontinuities(parent_row, discs)

            parent_table.append(parent_row)

        return parent_table


    @classmethod
    def _tree_leaves(cls, root, tree):
        """Given the root node ID and the tree as a networkX DiGraph returns
        the leaves of the tree.

        Must give it the reversed tree because it is recursive.

        Parameters
        ----------
        root : node_id

        tree : networkx.DiGraph
            The reversed tree from the contigtree

        Returns
        -------
        leaves: list of node_id
            The leaf node IDs of the tree.

        """

        # traverse the tree away from the root node until a branch point
        # is found then iterate over the subtrees from there and
        # recursively call this function on them
        branch_child_nodes = []
        curr_node = root
        leaves = []
        leaf_found = False
        while (len(branch_child_nodes) == 0) and (not leaf_found):

            # get the child nodes
            child_nodes = list(tree.adj[curr_node].keys())

            # if there is more than one child node, the current node is a
            # branch node
            if len(child_nodes) > 1:

                # we will use the branch child nodes as the roots of the
                # next recursion level
                branch_child_nodes = child_nodes

            # if there are no children then this is a leaf node
            elif len(child_nodes) == 0:
                # set the current node as the only leaf
                leaves = [curr_node]

                # and break out of the loop
                leaf_found = True

            # otherwise reset the current node
            else:
                # there will only be one child node
                curr_node = child_nodes[0]

        # this will run if any child nodes were found to find more leaves,
        # which won't happen when the loop ended upon finding a leaf node
        for branch_child_node in branch_child_nodes:
            branch_leaves = cls._tree_leaves(branch_child_node, tree)
            leaves.extend(branch_leaves)

        return leaves

    def _subtree_leaves(self, root):
        """Given a root defining a subtree on the full tree returns the leaves
        of that subtree.

        Parameters
        ----------
        root : node_id

        Returns
        -------
        leaves: list of node_id
            The leaf node IDs of the tree.

        """

        # get the subtree given the root
        subtree = self.get_subtree(root)

        # get the reversed directions of the cycle tree as a view, we
        # don't need a copy
        rev_tree = subtree.reverse(copy=False)

        # then we use the adjacencies to find the last node in the network
        # using a recursive algorithm
        leaves = self._tree_leaves(root, rev_tree)

        return leaves

    def leaves(self):
        """All of the leaves of this contig tree.

        Returns
        -------
        leaves: list of node_id
            The leaf node IDs of the tree.

        """
        leaves = []
        for root in self.roots():
            subtree_leaves = self._subtree_leaves(root)
            leaves.extend(subtree_leaves)

        return leaves

    def root_leaves(self):
        """Return a dictionary mapping the roots to their leaves."""

        root_leaves = {}
        for root in self.roots():
            root_leaves[root] = self._subtree_leaves(root)

        return root_leaves

    def _subtree_root(self, node):
        """Given a node find the root of the tree it is on

        Parameters
        ----------
        node : node_id

        Returns
        -------
        root : node_id

        """

        curr_node = node

        # we also get the adjacent nodes for this node
        adj_nodes = list(self.graph.adj[curr_node].keys())

        # there should only be one node in the dict
        assert len(adj_nodes) <= 1, "There should be at most 1 edge"

        # then we use this node as the starting point to move back to the
        # root, we end when there is no adjacent node
        while len(adj_nodes) > 0:

            # we take another step backwards, and choose the node in the
            # adjacency
            adj_nodes = list(self.graph.adj[curr_node])

            # there should only be 1 or none nodes
            assert len(adj_nodes) <= 1, "There should be at most 1 edge"

            # and reset the current node
            try:
                curr_node = adj_nodes[0]
            except IndexError:
                # this happens when this is the last node, inelegant apologies
                pass

        return curr_node

    def roots(self):
        """Returns all of the roots in this contig tree (which is technically
        a forest and can have multiple roots).

        Returns
        -------
        root : list of node_id

        """

        subtree_roots = []
        for subtree in self.subtrees():
            # use a node from the subtree to get the root
            node = next(subtree.adjacency())[0]
            subtree_root = self._subtree_root(node)
            subtree_roots.append(subtree_root)

        return subtree_roots

    def subtrees(self):
        """Returns all of the subtrees (with unique roots) in this contig tree
        (which is technically a forest and can have multiple
        roots).

        Returns
        -------
        subtrees : list of networkx.DiGraph trees

        """

        subtree_nxs = []
        for component_nodes in nx.weakly_connected_components(self.graph):

            # actually get the subtree from the main tree
            subtree = self.graph.subgraph(component_nodes)

            subtree_nxs.append(subtree)

        return subtree_nxs

    def get_subtree(self, node):
        """Given a node defining a subtree root return that subtree.

        Parameters
        ----------
        node : node_id

        Returns
        -------
        subtree : networkx.DiGraph tree

        """

        # get all the subtrees
        subtrees = self.subtrees()

        # see which tree the node is in
        for subtree in subtrees:

            # if the node is in it this is the subtree it is in so
            # just return it
            if node in subtree:
                return subtree

    def contig_sliding_windows(self, contig_trace, window_length):
        """Given a contig trace get the sliding windows of length
        'window_length' as contig walker traces.

        Parameters
        ----------
        contig_trace : list of tuples of ints (run_idx, cycle_idx)
            Trace defining a contig in the contig tree.
        window_length : int
            The length of the sliding windows to return.

        Returns
        -------

        windows : list of list of tuples of ints (traj_idx, cycle_idx)
            List of contig walker traces

        """

        # make a parent table for the contig trace
        parent_table = self.trace_parent_table(contig_trace)

        # this gives you windows of trace elements (traj_idx, cycle_idx)
        windows = sliding_window(parent_table, window_length)

        return windows

    def sliding_contig_windows(self, window_length):
        """Given a 'window_length' return all the windows over the contig tree
        as contig traces.

        Parameters
        ----------
        window_length : int

        Returns
        -------
        contig_windows : list of list of tuples of ints (run_idx, cycle_idx)
            List of contig traces

        """

        assert window_length > 1, "window length must be greater than one"

        # we can deal with each tree in this forest of trees separately,
        # that is runs that are not connected
        contig_windows = []
        for root in self.roots():

            # get the contig windows for the individual tree
            subtree_contig_windows = self._subtree_sliding_contig_windows(root, window_length)

            contig_windows.extend(subtree_contig_windows)

        return contig_windows

    def _subtree_sliding_contig_windows(self, subtree_root, window_length):
        """Get all the sliding windows of length 'window_length' from the
        subtree defined by the subtree root as run traces.

        Parameters
        ----------
        subtree_root : node_id

        window_length : int

        Returns
        -------
        subtree_windows : list of list of tuples of ints (run_idx, cycle_idx)
            List of the contig tree windows as contig traces

        """

        # to generate all the sliding windows over a connected cycle tree
        # it is useful to think of it as a braid, since within this tree
        # there is a forest of trees which are the lineages of the
        # walkers. To simplify things we can first generate window traces
        # over the cycle tree (ignoring the fine structure of the walkers),
        # which is a very similar process as to the original process of
        # the sliding windows over trees, and then treat each window trace
        # as it's own parent table, which can then be treated using the
        # original sliding window algorithm. As long as the correct contig
        # traces are generated this will be no problem, and all that is
        # left is to translate the cycle idxs properly such that the
        # windows generated are inter-run traces i.e. lists of tuples of
        # the form (run_idx, traj_idx, cycle_idx) where traj_idx and
        # cycle_idx are internal to the run specified by run_idx.

        # so we want to construct a list of contig windows, which are
        # traces with components (run_idx, cycle_idx)
        contig_windows = []

        # to do this we start at the leaves of the cycle tree, but first
        # we have to find them. The cycle tree should be directed with
        # edges pointing towards parents. Since we are working with a
        # single tree, we have a single root and we can just reverse the
        # directions of the edges and recursively walk the tree until we
        # find nodes with no adjacent edges

        # first we need to find the leaves for this subtree
        leaves = self._subtree_leaves(subtree_root)

        # now we use these leaves to move backwards in the tree with the
        # window length to get contiguous segments
        contig_windows = []

        # starting with the leaf nodes we generate contig trace windows
        # until the last nodes are the same as other windows from other
        # leaves, i.e. until a branch point has been arrived at between 2
        # or more leaf branches. To do this we start at the leaf of the
        # longest spanning contig and make windows until the endpoint is
        # no longer the largest contig cycle index. Then we alternate
        # between them until we find they have converged

        # initialize the list of active branches all going back to the
        # root
        branch_contigs = [self.get_branch_trace(*leaf) for leaf in leaves]

        done = False
        while not done:

            # make a window for the largest endpoint, no need to break
            # ties since the next iteration will get it
            contig_lengths = [len(contig) for contig in branch_contigs]
            longest_branch_idx = np.argmax(contig_lengths)

            # if the branch is not long enough for the window we end this
            # process
            if window_length > len(branch_contigs[longest_branch_idx]):
                done = True

            # otherwise we get the next window and do the other processing
            else:
                # get the next window for this branch
                window = branch_contigs[longest_branch_idx][-window_length:]

                contig_windows.append(window)

                # pop the last element off of this branch contig
                last_node = branch_contigs[longest_branch_idx].pop()

                # if there are any other branches of the same length that have
                # this as their last node then we have reached a branch point
                # and that other branch must be eliminated
                for branch_idx, branch_contig in enumerate(branch_contigs):

                    # compare the last node in contig and if it is the same as
                    # the node that was just used as a window end
                    if branch_contig[-1] == last_node:
                        # this branch is the same so we just get rid of it
                        _ = branch_contigs.pop(branch_idx)

        return contig_windows

    def sliding_windows(self, window_length):
        """Returns all the sliding windows over walker trajectories as run
        traces for a given window length.

        Parameters
        ----------
        window_length : int

        Returns
        -------
        windows : list of list of tuples of ints (run_idx, traj_idx, cycle_idx)
            List of run traces.

        """

        # get all of the contig traces for these trees
        contig_traces = self.sliding_contig_windows(window_length)

        # for each of these we generate all of the actual frame sliding windows
        windows = []
        for contig_trace in contig_traces:

            # these are windows of trace element (traj_idx, cycle_idx)
            # all over this contig
            contig_windows = self.contig_sliding_windows(contig_trace, window_length)

            # convert those traces to the corresponding (run_idx, traj_idx, cycle_idx)
            # trace
            for contig_window in contig_windows:
                run_trace_window = self.contig_trace_to_run_trace(contig_trace, contig_window)
                windows.append(run_trace_window)

        return windows

    @classmethod
    def _rec_spanning_paths(cls, edges, root):
        """Given a set of directed edges (source, target) and a root node id
        of a tree imposed over the edges, returns all the paths over
        that tree which span from the root to a leaf.

        This is a recursive function and has pretty bad performance
        for nontrivial simulations.

        Parameters
        ----------
        edges : (node_id, node_id)

        root : node_id

        Returns
        -------

        spanning_paths : list of edges

        """

        # nodes targetting this root
        root_sources = []

        # go through all the edges and find those with this
        # node as their target
        for edge_source, edge_target in edges:

            # check if the target_node we are looking for matches
            # the edge target node
            if root == edge_target:

                # if this root is a target of the source add it to the
                # list of edges targetting this root
                root_sources.append(edge_source)

        # from the list of source nodes targetting this root we choose
        # the lowest index one, so we sort them and iterate through
        # finding the paths starting from it recursively
        root_paths = []
        root_sources.sort()
        for new_root in root_sources:

            # add these paths for this new root to the paths for the
            # current root
            root_paths.extend(cls._rec_spanning_paths(edges, new_root))

        # if there are no more sources to this root it is a leaf node and
        # we terminate recursion, by not entering the loop above, however
        # we manually generate an empty list for a path so that we return
        # this "root" node as a leaf, for default.
        if len(root_paths) < 1:
            root_paths = [[]]

        final_root_paths = []
        for root_path in root_paths:
            final_root_paths.append([root] + root_path)

        return final_root_paths

    # @classmethod
    # def _find_root_sources(cls, edges, root):

    #     # nodes targetting this root
    #     root_sources = []

    #     # go through all the edges and find those with this
    #     # node as their target
    #     for edge_source, edge_target in edges:

    #         # check if the target_node we are looking for matches
    #         # the edge target node
    #         if root == edge_target:

    #             # if this root is a target of the source add it to the
    #             # list of edges targetting this root
    #             root_sources.append(edge_source)

    #     return root_sources

    def _spanning_paths(self, root):
        """

        Parameters
        ----------
        root : node_id

        Returns
        -------

        spanning_paths : list of list of edges

        """

        # get the subtree given the root
        subtree = self.get_subtree(root)

        # get the leaves manually since we already have the subtree
        leaves = self._tree_leaves(root, subtree.reverse(copy=False))

        # then starting at each leaf just write out the path from it
        # to the root
        leaf_paths = {}
        for leaf in leaves:

            leaf_path = deque([leaf])
            curr_node = leaf
            root_found = False
            while not root_found:

                # get the next node, if there is one
                parent_nodes = list(subtree.adj[curr_node].keys())

                if len(parent_nodes) == 0:
                    root_found = True
                elif len(parent_nodes) == 1:
                    next_node = parent_nodes.pop()
                    leaf_path.appendleft(next_node)
                    curr_node = next_node
                else:
                    raise ValueError("Multiple parents, illegal tree")

            leaf_paths[leaf] = list(leaf_path)

        return leaf_paths


    def spanning_contig_traces(self):
        """Returns a list of all possible spanning contigs given the
        continuations present in this file. Spanning contigs are paths
        through a tree that must start from a root node and end at a
        leaf node.

        This algorithm always returns them in a canonical order (as
        long as the runs are not rearranged after being added). This
        means that the indices here are the indices of the contigs.

        Returns
        -------
        spanning_contig_traces : list of list of tuples of ints (run_idx, cycle_idx)
            List of all spanning contigs which are contig traces.

        """

        spanning_contig_traces = []
        # roots should be sorted already, so we just iterate over them
        for root, root_spanning_contigs in self._root_spanning_contig_traces().items():

            spanning_contig_traces.extend(root_spanning_contigs)

        return spanning_contig_traces

    def _root_spanning_contig_traces(self):
        """Returns a list of all possible spanning contigs given the
        continuations present in this file. Spanning contigs are paths
        through a tree that must start from a root node and end at a
        leaf node.

        This algorithm always returns them in a canonical order (as
        long as the runs are not rearranged after being added). This
        means that the indices here are the indices of the contigs.

        Returns
        -------
        spanning_contig_traces : dict of root_id to list of tuples of ints (run_idx, cycle_idx)
            Dictionary mapping the root ids to all spanning contigs
            for it which are contig traces.

        """

        spanning_contig_traces = {}
        # roots should be sorted already, so we just iterate over them
        for root in self.roots():

            # get the spanning paths by passing the continuation edges
            # and this root to this recursive static method
            root_spanning_contigs = [span for leaf, span in self._spanning_paths(root).items()]

            spanning_contig_traces[root] = root_spanning_contigs

        return spanning_contig_traces


    @classmethod
    def _contig_trace_to_contig_runs(cls, contig_trace):
        """ Convert a contig trace to a list of runs.

        Parameters
        ----------
        contig_trace : list of tuples of ints (run_idx, cycle_idx)

        Returns
        -------
        contig_runs : list of int

        """

        contig_runs = []
        for run_idx, cycle_idx in contig_trace:

            if not run_idx in contig_runs:
                contig_runs.append(run_idx)
            else:
                pass

        return contig_runs

    @classmethod
    def _contig_runs_to_continuations(cls, contig_runs):
        """Helper function to convert a list of run indices defining a contig
        to continuations.

        Parameters
        ----------
        contig_runs : list of int

        Returns
        -------
        continuations : list of tuple of int (run_idx, run_idx)

        """

        continuations = []
        for i in range(len(contig_runs) - 1, 0, -1):
            continuations.append([contig_runs[i], contig_runs[i-1]])

        return continuations

    @classmethod
    def _continuations_to_contig_runs(cls, continuations):
        """Helper function that converts a list of continuations to a list of
        the runs in the order of the contigs defined by the continuations.

        Parameters
        ----------
        continuations : list of tuple of int (run_idx, run_idx)

        Returns
        -------
        contig_runs : list of int

        """

        if len(continuations) == 0:
            return []

        continuations = list(copy(continuations))

        # for the runs in the continuations figure out which are the
        # ends and which end they are
        continued_runs = [continued_run for next_run, continued_run in continuations]
        next_runs = [next_run for next_run, continued_run in continuations]

        # the intersection of these are runs in the middle of the
        # contig. Runs that are in the continued runs but not in the
        # next runs are the base of the chain
        base_run = set(continued_runs).difference(next_runs)

        # double check you don't find multiple bases or none (a loop)
        assert len(base_run) == 1, "None or multiple base runs in this contig spec"
        base_run = base_run.pop()

        # do the same for the end run
        end_run = set(next_runs).difference(continued_runs)
        assert len(end_run) == 1, "None or multiple end runs in this contig spec"
        end_run = end_run.pop()

        # now go through the runs starting at the base and ending at
        # the end and build up the runs
        continued_run = base_run
        # get the next run from this base run
        next_run = [next_run for next_run, cont_run in continuations
                    if cont_run == continued_run][0]

        # start the listing of the runs in order
        contig_runs = [base_run, next_run]

        # now iterate through continuations until we get to the end
        while next_run != end_run:
            # get the next continuation pair
            continued_run = next_run
            next_run = [next_run for next_run, cont_run in continuations
                        if cont_run == continued_run][0]
            contig_runs.append(next_run)

        # should have a completed contig_runs listing now

        # since this is only valid if the continuations don't form a
        # tree check that we didn't put the same number in twice,
        # which this would indicate, fail if true
        assert len(set(contig_runs)) == len(contig_runs), \
            "this is not a single contig"

        return contig_runs


    def exit_point_trajectories(self):
        """Return full run traces for every warping event."""

        # this operation is done over every spanning contig since we
        # need a parent table
        span_traces = []
        for span_trace in self.spanning_contig_traces():

            # make the contig
            span_contig = self.make_contig(span_trace)

            span_traces.extend(span_contig.exit_point_trajectories())

        return span_traces


class ContigTree(BaseContigTree):
    """Wraps a WepyHDF5 object and gives access to logical trajectories.

    The contig tree is technically a forest (a collection of trees)
    and can have multiple roots.

    """

    def __init__(self, wepy_h5,
                 base_contigtree=None,
                 continuations=Ellipsis,
                 runs=Ellipsis,
                 boundary_condition_class=None,
                 decision_class=None):

        self.closed = True

        # if we pass a base contigtree use that one instead of building one manually
        if base_contigtree is not None:
            assert isinstance(base_contigtree, BaseContigTree)

            self._set_base_contigtree_to_self(base_contigtree)

        else:

            new_contigtree = BaseContigTree(wepy_h5,
                                            continuations=continuations,
                                            runs=runs,
                                            boundary_condition_class=boundary_condition_class,
                                            decision_class=decision_class)

            self._set_base_contigtree_to_self(new_contigtree)

        self._wepy_h5 = wepy_h5

    def _set_base_contigtree_to_self(self, base_contigtree):

        self._base_contigtree = base_contigtree

        # then make references to this for the attributes we need
        self._graph = self._base_contigtree._graph
        self._boundary_condition_class = self._base_contigtree._boundary_condition_class
        self._decision_class = self._base_contigtree._decision_class
        self._continuations = self._base_contigtree._continuations
        self._run_idxs = self._base_contigtree._run_idxs
        self._spans = self._base_contigtree._spans

    def open(self, mode=None):
        if self.closed:
            self.wepy_h5.open(mode=mode)
            self.closed = False
        else:
            raise IOError("This file is already open")

    def close(self):
        self.wepy_h5.close()
        self.closed = True

    def __enter__(self):
        self.wepy_h5.__enter__()
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.wepy_h5.__exit__(exc_type, exc_value, exc_tb)
        self.close()


    @property
    def base_contigtree(self):
        return self._base_contigtree

    @property
    def wepy_h5(self):
        """The WepyHDF5 source object for which the contig tree is being constructed. """
        return self._wepy_h5

    # TODO deprecate, commenting to catch errors if this breaks
    # stuff. THis shouldn't be something we want to expose since it
    # defines a contig as a sequence of runs, which it needn't be. Nor
    # do I think this code is even correct, so...

    # def contig_to_run_trace(self, contig, contig_trace):
    #     """Convert a run listing (contig) and a contig trace to a run trace.

    #     Parameters
    #     ----------
    #     contig : list of int

    #     contig_trace : list of tuples of ints (run_idx, cycle_idx)

    #     Returns
    #     -------

    #     run_trace : list of tuples of ints (run_idx, traj_idx, cycle_idx)

    #     """

    #     # go through the contig and get the lengths of the runs that
    #     # are its components, and slice that many trace elements and
    #     # build up the new trace
    #     runs_trace = []
    #     cum_n_frames = 0
    #     for run_idx in contig:

    #         # number of frames in this run
    #         n_frames = self.wepy_h5.num_run_cycles(run_idx)

    #         # get the contig trace elements for this run
    #         contig_trace_elements = contig_trace[cum_n_frames : n_frames + cum_n_frames]

    #         # convert the cycle_idxs to the run indexing and add a run_idx to each
    #         run_trace_elements = [(run_idx, traj_idx, contig_cycle_idx - cum_n_frames)
    #                               for traj_idx, contig_cycle_idx in contig_trace_elements]

    #         # add these to the trace
    #         runs_trace.extend(run_trace_elements)

    #         # then increase the cumulative n_frames for the next run
    #         cum_n_frames += n_frames

    #     return run_trace_elements


    # TODO: optimize this, we don't need to recalculate everything
    # each time to implement this
    def make_contig(self, contig_trace):
        """Create a Contig object given a contig trace.

        Parameters
        ----------
        contig_trace : list of tuples of ints (run_idx, cycle_idx)

        Returns
        -------
        contig : Contig object

        """

        # get the runs and continuations for just this contig

        # get the contig as the sequence of run idxs
        contig_runs = self._contig_trace_to_contig_runs(contig_trace)

        # then convert to continuations
        continuations = self._contig_runs_to_continuations(contig_runs)

        return Contig(self.wepy_h5,
                      runs=contig_runs,
                      continuations=continuations,
                      boundary_condition_class=self.boundary_condition_class,
                      decision_class=self.decision_class)


    def warp_trace(self):
        """Get the trace for all unique warping events from all contigs."""

        with self:
            big_trace = []
            for contig_idx in self.span_traces.keys():
                contig = self.span_contig(contig_idx)

                big_trace.extend(contig.warp_trace())

        # then cast to a set to get the unique ones
        return list(set(big_trace))

    def resampling_trace(self, decision_id):
        """Return full run traces for every specified type of resampling
        event.

        Parameters
        ----------

        decision_id : int
            The string ID of the decision you want to match on and get
            lineages for.

        """

        with self:
            big_trace = []
            for contig_idx in self.span_traces.keys():
                contig = self.span_contig(contig_idx)

                big_trace.extend(contig.resampling_trace(decision_id))

        # then cast to a set to get the unique ones
        return list(set(big_trace))

    def final_trace(self):
        """Return a trace of all the walkers at the end of the contig."""

        with self:
            big_trace = []
            for contig_idx in self.span_traces.keys():
                contig = self.span_contig(contig_idx)

                big_trace.extend(contig.final_trace())

        # then cast to a set to get the unique ones
        return list(set(big_trace))


    def lineages(self, trace):
        """Get the ancestry lineage for each element of the trace as a run
        trace."""

        lines = []
        # for each element of the trace we need to get it's lineage
        for w_run_idx, w_walker_idx, w_cycle_idx in trace:

            # get a span contig that it is a part of
            for span_idx, span_trace in self.span_traces.items():

                # if this frame is from this span trace
                if (w_run_idx, w_cycle_idx,) in span_trace:

                    # then we process to get the lineages with this span
                    contig = self.span_contig(span_idx)

                    # lineages for a single frame
                    lines.append(contig.lineages([(w_cycle_idx, w_walker_idx)])[0])

                    # any span trace will do so we finish at this point
                    break


        return lines

class Contig(ContigTree):
    """Wraps a WepyHDF5 object and gives access to logical trajectories
    from a single contig.

    This class is very similar to the ContigTree class and inherits
    all of those methods. It adds methods for getting records and
    other data about a single contig.

    """

    def __init__(self, wepy_h5,
                 **kwargs):

        # uses superclass docstring exactly, this constructor just
        # generates some extra attributes

        # use the superclass initialization
        super().__init__(wepy_h5, **kwargs)

        # check that the result is a single contig
        spanning_contig_traces = self.spanning_contig_traces()
        assert len(spanning_contig_traces) == 1, \
            "continuations given do not form a single contig"

        # if so we add some useful attributes valid for only a
        # standalone contig

        # the contig_trace
        self._contig_trace = spanning_contig_traces[0]

        # TODO this should not be part of the API in the future since
        # we don't want to have the run_idxs alone be how contigs are
        # defined (since we want contigs to be able to go to different
        # runs from the middle of other runs), in any case that is how
        # it is implemented now and there is no reason to change it
        # now, so we keep it hidden

        # get the number of runs in that trace
        trace_run_idxs = set(run_idx for run_idx, _ in self._contig_trace)

        # the run idxs of the contig for more than one
        if len(trace_run_idxs) > 1:
            self._contig_run_idxs = self._continuations_to_contig_runs(self.continuations)

        # if there is only 1 run we set it like this
        else:
            self._contig_run_idxs = list(self.run_idxs)

    def contig_fields(self, fields):
        """Returns trajectory field data for the specified fields.

        Parameters
        ----------
        fields : list of str
            The field keys you want to retrieve data for.

        Returns
        -------
        fields_dict : dict of str: numpy.ndarray

        """

        return self.wepy_h5.get_contig_trace_fields(self.contig_trace, fields)

    @property
    def contig_trace(self):
        """Returns the contig trace corresponding to this contig.

        Returns
        -------
        contig_trace :: list of tuples of ints (run_idx, cycle_idx)

        """

        return self._contig_trace

    @property
    def num_cycles(self):
        """The number of cycles in this contig.

        Returns
        -------
        num_cycles : int

        """
        return len(self.contig_trace)

    # TODO: may need to be implemented without using the wepy_h5 in the BaseContigTree
    def num_walkers(self, cycle_idx):
        """Get the number of walkers at a given cycle in the contig.

        Parameters
        ----------
        cycle_idx : int

        Returns
        -------
        n_walkers : int

        """

        # get the run idx and in-run cycle idx for this cycle_idx
        run_idx, run_cycle_idx = self.contig_trace[cycle_idx]

        # then get the number of walkers for that run and cycle
        n_walkers = self.wepy_h5.num_walkers(run_idx, run_cycle_idx)

        return n_walkers

    def records(self, record_key):
        """Returns the records for the given key.

        Parameters
        ----------
        record_key : str

        Returns
        -------
        records : list of namedtuple records

        """
        return self.wepy_h5.run_contig_records(self._contig_run_idxs, record_key)

    def records_dataframe(self, record_key):
        """Returns the records as a pandas.DataFrame for the given key.

        Parameters
        ----------
        record_key : str

        Returns
        -------
        records_df : pandas.DataFrame

        """
        return self.wepy_h5.run_contig_records_dataframe(self._contig_run_idxs, record_key)

    # resampling
    def resampling_records(self):
        """Returns the resampling records.

        Returns
        -------
        resampling_records : list of namedtuple records

        """

        return self.records(RESAMPLING)

    def resampling_records_dataframe(self):
        """Returns the resampling records as a pandas.DataFrame.

        Returns
        -------
        resampling_df : pandas.DataFrame

        """

        return pd.DataFrame(self.resampling_records())

    # resampler records
    def resampler_records(self):
        """Returns the resampler records.

        Returns
        -------
        resampler_records : list of namedtuple records

        """


        return self.records(RESAMPLER)

    def resampler_records_dataframe(self):
        """Returns the resampler records as a pandas.DataFrame.

        Returns
        -------
        resampler_df : pandas.DataFrame

        """


        return pd.DataFrame(self.resampler_records())

    # warping
    def warping_records(self):
        """Returns the warping records.

        Returns
        -------
        warping_records : list of namedtuple records

        """


        return self.records(WARPING)

    def warping_records_dataframe(self):
        """Returns the warping records as a pandas.DataFrame.

        Returns
        -------
        warping_df : pandas.DataFrame

        """


        return pd.DataFrame(self.warping_records())

    # boundary conditions
    def bc_records(self):
        """Returns the boundary conditions records.

        Returns
        -------
        bc_records : list of namedtuple records

        """


        return self.records(BC)

    def bc_records_dataframe(self):
        """Returns the boundary conditions records as a pandas.DataFrame.

        Returns
        -------
        bc_df : pandas.DataFrame

        """


        return pd.DataFrame(self.bc_records())

    # progress
    def progress_records(self):
        """Returns the progress records.

        Returns
        -------
        progress_records : list of namedtuple records

        """


        return self.records(PROGRESS)

    def progress_records_dataframe(self):
        """Returns the progress records as a pandas.DataFrame.

        Returns
        -------
        progress_df : pandas.DataFrame

        """


        return pd.DataFrame(self.progress_records())


    # resampling panel
    def resampling_panel(self):
        """Returns the full resampling panel for this contig.

        Returns
        -------
        resampling_panel : list of list of list of namedtuple records

        """

        return self.wepy_h5.contig_resampling_panel(self._contig_run_idxs)

    def parent_table(self, discontinuities=True):
        """Returns the full parent table for this contig.

        Notes
        -----

        This requires the decision class to be given to the Contig at
        construction.

        Warnings
        --------

        If the simulation was run with boundary conditions that result
        in discontinuous warping events and that class is not provided
        at construction time to this class these discontinuities will
        not be taken into account and not added to this parent table.


        Returns
        -------
        parent_table : list of list of int

        """

        return self.trace_parent_table(self.contig_trace,
                                       discontinuities=discontinuities)

    def lineages_contig(self, contig_trace, discontinuities=True):

        # get the parent table for this contig
        parent_table = self.parent_table(discontinuities=discontinuities)

        lineages = []
        for walker_idx, cycle_idx in contig_trace:
            lineage = ancestors(parent_table, cycle_idx, walker_idx)

            lineages.append(lineage)

        return lineages

    def lineages(self, contig_trace, discontinuities=True):
        """Get the ancestry lineage for each element of the trace as a run
        trace."""

        return [self.walker_trace_to_run_trace(trace)
                for trace in self.lineages_contig(contig_trace,
                                                  discontinuities=discontinuities)]


    def warp_contig_trace(self):
        """Return a trace that gives all of the walkers that were warped."""

        trace = []
        for warping_record in self.warping_records():

            cycle_idx = warping_record[0]
            walker_idx = warping_record[1]

            rec = (walker_idx, cycle_idx)

            trace.append(rec)

        return trace

    def resampling_contig_trace(self, decision_id):
        """Return full run traces for every specified type of resampling
        event.

        Parameters
        ----------

        decision_id : int
            The integer ID of the decision you want to match on and get
            lineages for. This is the integer value of the decision
            enumeration value.

        """

        trace = []

        # go through the resampling records and filter only the
        # records that match the decision_id
        for rec in self.resampling_records():

            if rec.decision_id == decision_id:

                trace.append((rec.walker_idx, rec.cycle_idx))

        return trace

    def final_contig_trace(self):

        # this is just the last cycle index
        last_cycle_idx = self.num_cycles - 1

        # make a contig trace for each of the last walkers
        trace = [(walker_idx, last_cycle_idx)
                 for walker_idx in range(self.num_walkers(last_cycle_idx))]

        return trace


    def warp_trace(self):
        """Return a run trace that gives all of the walkers that were warped."""

        trace = self.warp_contig_trace()

        run_trace = self.contig_trace_to_run_trace(self.contig_trace, trace)

        return run_trace

    def resampling_trace(self, decision_id):
        """Return full run traces for every specified type of resampling
        event.

        Parameters
        ----------

        decision_id : int
            The string ID of the decision you want to match on and get
            lineages for.

        """


        trace = self.resampling_contig_trace(decision_id)

        run_trace = self.contig_trace_to_run_trace(self.contig_trace, trace)

        return run_trace


    def final_trace(self):
        """Return a trace of all the walkers at the end of the contig."""

        trace = self.final_contig_trace()

        run_trace = self.contig_trace_to_run_trace(self.contig_trace, trace)

        return run_trace

    def resampling_tree_layout_graph(self,
                                     bc_class=None,
                                     progress_key=None,
                                     node_shape='disc',
                                     discontinuous_node_shape='square',
                                     colormap_name='plasma',
                                     node_radius=None,
                                     row_spacing=None,
                                     step_spacing=None,
                                     central_axis=None,
    ):

        ### The data we need for making the resampling tree

        ## parent table, don't include discontinuities, we will handle
        ## that on our own
        parent_forest = ParentForest(self)

        ## walker weights
        with self:
            walker_weights = self.wepy_h5.get_contig_trace_fields(
                self.contig_trace,
                ['weights']
            )['weights']

        ### Optional data

        ## Warping records
        if bc_class is not None:
            with self:
                warping_records = self.warping_records()


        ## walker progress

        if progress_key is not None:
            with self:
                progress_records = self.progress_records()

            walkers_progress = []
            progress_getter = attrgetter(progress_key)
            for prog_rec in progress_records:
                walkers_progress.append(progress_getter(prog_rec))

            del progress_records

        ### Layout node properties

        ## Discontinuities

        if bc_class is not None:
            # tabulate the discontinuities
            discontinuous_nodes = []
            for warping_record in warping_records:

                # if this record classifies as discontinuous
                if bc_class.warping_discontinuity(warping_record):

                    # then we save it as one of the nodes that is discontinuous
                    disc_node_id = (warping_record.cycle_idx, warping_record.walker_idx)
                    discontinuous_nodes.append(disc_node_id)


        ## Free Energies

        # The only ones that should be none are the roots which will
        # be 1.0 / N
        n_roots = len(parent_forest.roots)
        root_weight = 1.0 / n_roots

        # compute the size of the nodes to plot, which is the
        # free_energy. To compute this we flatten the list of list of
        # weights to a 1-d array then unravel it back out

        # put the root weights at the beginning of this array, and
        # account for it in the flattening
        flattened_weights = [root_weight for _ in walker_weights[0]]
        cycles_n_walkers = [len(walker_weights[0])]

        for cycle_weights in walker_weights:
            cycles_n_walkers.append(len(cycle_weights))
            flattened_weights.extend(list(cycle_weights.reshape( (len(cycle_weights),) )))

        # now compute the free energy
        flattened_free_energies = calc_free_energy(np.array(flattened_weights))

        # and ravel them back into a list of lists
        free_energies = []
        last_index = 0
        for cycle_n_walkers in cycles_n_walkers:
            free_energies.append(list(
                flattened_free_energies[last_index:last_index + cycle_n_walkers]))
            last_index += cycle_n_walkers

        # put these into the parent forest graph as node attributes,
        # so we can get them out as a lookup table by node id for
        # assigning to nodes in the layout. We skip the first row of
        # free energies for the roots in this step
        parent_forest.set_attrs_by_array('free_energy', free_energies[1:])

        ## Progress colors
        colormap = cm.get_cmap(name=colormap_name)

        # if a progress key was given use that
        if progress_key is not None:

            # get the maximum progress value in the data
            max_progress_value = max([max(row) for row in walkers_progress])

            # then use that for the ratio
            norm_ratio = 1.0 / max_progress_value

            # now that we have ratios for normalizing values we apply that
            # and then use the lookup table for the color bar to get the
            # RGB color values
            colors = []
            for progress_row in walkers_progress:
                color_row = [colormap(progress * norm_ratio, bytes=True)
                             for progress in progress_row]
                colors.append(color_row)

        # otherwise set them to the free energy
        else:

            # get the maximum progress value in the data
            max_fe_value = max([max(row) for row in free_energies[1:]])

            # then use that for the ratio
            norm_ratio = 1.0 / max_fe_value

            # now that we have ratios for normalizing values we apply that
            # and then use the lookup table for the color bar to get the
            # RGB color values
            colors = []
            for progress_row in free_energies[1:]:
                color_row = [colormap(progress * norm_ratio, bytes=True)
                             for progress in progress_row]
                colors.append(color_row)

        parent_forest.set_attrs_by_array('color', colors)


        ### Per node properties

        # get the free energies out as a dictionary, flattening it to
        # a single value. Set the nodes with None for their FE to a
        # small number.

        # now we make the graph which will contain the layout
        # information
        layout_forest = LayoutGraph(parent_forest.graph)

        ## Free energy and size

        # get the free energy attributes out using the root weight for
        # the nodes with no fe defined

        # start it with the root weights
        node_fes = {(-1, i) : free_energy for i, free_energy in enumerate(free_energies[0])}
        for node_id, fe_arr in layout_forest.get_node_attributes('free_energy').items():

            if fe_arr is None:
                node_fes[node_id] = 0.0
            else:
                node_fes[node_id] = fe_arr

        ## Node colors

        # the default progress for the root ones is 0 and so the color
        # is also 0
        node_colors = {}
        for node_id, color_arr in layout_forest.get_node_attributes('color').items():

            if color_arr is None:
                # make a black color for these
                node_colors[node_id] = tuple(int(255) for a in range(4))
            else:
                node_colors[node_id] = color_arr

        # set discontinuous nodes to black
        if bc_class is not None:

            for discontinuous_node in discontinuous_nodes:
                node_colors[discontinuous_node] = tuple(int(0) for a in range(4))


        ## Node Shapes

        # make a dictionary for the shapes of the nodes
        node_shapes = {}

        # set all the nodes to a default 'disc'
        for node in layout_forest.viz_graph.nodes:
            node_shapes[node] = node_shape

        # then set all the discontinuous ones
        if bc_class is not None:
            for discontinuous_node in discontinuous_nodes:
                node_shapes[discontinuous_node] = discontinuous_node_shape


        ## Node positions

        # get the options correctly
        restree_layout_opts = {
            'node_radius' : node_radius,
            'row_spacing' : row_spacing,
            'step_spacing' : step_spacing,
            'central_axis' : central_axis,
        }

        # filter out the None values, this will ensure the defaults
        # are used
        restree_layout_opts = {field : value
                               for field, value in restree_layout_opts.items()
                               if value is not None}

        # now we get to the part where we make the layout (positioning
        # of nodes), so we parametrize a layout engine
        tree_layout = ResamplingTreeLayout(**restree_layout_opts)

        node_coords = tree_layout.layout(parent_forest,
                                         node_radii=node_fes)

        ### Set Layout values

        # we are going to output to the gexf format so we use the
        # pertinent methods

        # we set the sizes
        layout_forest.set_node_gexf_sizes(node_fes)

        # and set the colors based on the progresses

        layout_forest.set_node_gexf_colors_rgba(node_colors)

        # then set the shape of the nodes based on whether they were
        # warped or not
        layout_forest.set_node_gexf_shape(node_shapes)

        # positions
        layout_forest.set_node_gexf_positions(node_coords)

        return layout_forest

