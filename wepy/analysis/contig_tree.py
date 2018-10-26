import itertools as it
from copy import copy

import networkx as nx
import numpy as np

from wepy.analysis.parents import DISCONTINUITY_VALUE, \
                                  parent_panel, net_parent_table,\
                                  ancestors, sliding_window

# optional dependencies
try:
    import pandas as pd
except ModuleNotFoundError:
    warn("pandas is not installed and that functionality will not work", RuntimeWarning)


# the groups of run records
RESAMPLING = 'resampling'
RESAMPLER = 'resampler'
WARPING = 'warping'
PROGRESS = 'progress'
BC = 'boundary_conditions'

class ContigTree():

    RESAMPLING_PANEL_KEY = 'resampling_steps'
    PARENTS_KEY = 'parent_idxs'
    DISCONTINUITY_KEY = 'discontinuities'


    def __init__(self, wepy_h5,
                 continuations=Ellipsis,
                 runs=Ellipsis,
                 boundary_condition_class=None,
                 decision_class=None):

        self._graph = nx.DiGraph()

        self._wepy_h5 = wepy_h5

        self._boundary_condition_class=boundary_condition_class
        self._decision_class = decision_class

        # we can optionally specify which continuations to use when
        # creating the contig tree instead of defaulting to the whole file
        self._continuations = set()
        self._run_idxs = set()

        # if specific runs were specified we add them right away, they
        # should be unique
        if runs is Ellipsis:
            self._run_idxs.update(self.wepy_h5.run_idxs)
        elif runs is not None:
            self._run_idxs.update(runs)

        # the continuations also give extra runs to incorporate into
        # this contig tree

        # if it is Ellipsis (...) then we include all runs and all the continuations
        if continuations is Ellipsis:
            self._run_idxs.update(self.wepy_h5.run_idxs)
            self._continuations.update([(a,b) for a, b in self.wepy_h5.continuations])

        # otherwise we make the tree based on the runs in the
        # continuations
        elif continuations is not None:
            # the unique run_idxs
            self._run_idxs.update(it.chain(*self._continuations))

            # the continuations themselves
            self._continuations.update([(a,b) for a, b in continuations])


        # using the wepy_h5 create a tree of the cycles
        self._create_tree()

        self._set_resampling_panels()

        if self._decision_class is not None:
            self._set_parents(self._decision_class)

            if self._boundary_condition_class is not None:
                self._set_discontinuities(self._boundary_condition_class)

    @property
    def graph(self):
        return self._graph

    @property
    def decision_class(self):
        return self._decision_class

    @property
    def boundary_condition_class(self):
        return self._boundary_condition_class

    def _create_tree(self):

        # first go through each run without continuations
        for run_idx in self._run_idxs:
            n_cycles = self.wepy_h5.run_n_cycles(run_idx)

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
            target_node = (edge_target, self.wepy_h5.run_n_cycles(edge_target)-1)

            # make the edge
            edge = (source_node, target_node)

            # add this connector edge to the network
            self.graph.add_edge(*edge)

    def _set_resampling_panels(self):

        # then get the resampling tables for each cycle and put them
        # as attributes to the appropriate nodes
        for run_idx in self.run_idxs:

            run_resampling_panel = self.wepy_h5.run_resampling_panel(run_idx)

            # add each cycle of this panel to the network by adding
            # them in as nodes with the resampling steps first
            for step_idx, step in enumerate(run_resampling_panel):
                node = (run_idx, step_idx)
                self.graph.nodes[node][self.RESAMPLING_PANEL_KEY] = step

    def _set_discontinuities(self, boundary_conditions_class):

        # initialize the attributes for discontinuities to 0s for no
        # discontinuities
        for node in self.graph.nodes:
            n_walkers = len(self.graph.node[node][self.PARENTS_KEY])
            self.graph.node[node][self.DISCONTINUITY_KEY] = [0 for i in range(n_walkers)]

        #
        for run_idx in self.run_idxs:

            # get the warping records for this run
            warping_records = self.wepy_h5.warping_records([run_idx])

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
        the cycle tree given."""

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
        return self._run_idxs

    @property
    def continuations(self):
        return self._continuations

    @property
    def wepy_h5(self):
        return self._wepy_h5

    def contig_trace_to_run_trace(self, contig_trace, contig_walker_trace):
        """Given a trace of a contig with elements (run_idx, cycle_idx) and
        walker based trace of elements (traj_idx, cycle_idx) over that
        contig get the trace of elements (run_idx, traj_idx, cycle_idx)

        """

        trace = []

        for frame_idx, contig_el in enumerate(contig_trace):

            run_idx, cycle_idx = contig_el
            traj_idx = contig_walker_trace[frame_idx][0]
            frame = (run_idx, traj_idx, cycle_idx)
            trace.append(frame)

        return trace


    def contig_to_run_trace(self, contig, contig_trace):
        """Convert a trace of elements (traj_idx, cycle_idx) over the contig
        trace given over this contig tree and return a trace over the
        runs with elements (run_idx, traj_idx, cycle_idx).

        """

        # go through the contig and get the lengths of the runs that
        # are its components, and slice that many trace elements and
        # build up the new trace
        runs_trace = []
        cum_n_frames = 0
        for run_idx in contig:

            # number of frames in this run
            n_frames = self.wepy_h5.run_n_frames(run_idx)

            # get the contig trace elements for this run
            contig_trace_elements = contig_trace[cum_n_frames : n_frames + cum_n_frames]

            # convert the cycle_idxs to the run indexing and add a run_idx to each
            run_trace_elements = [(run_idx, traj_idx, contig_cycle_idx - cum_n_frames)
                                  for traj_idx, contig_cycle_idx in contig_trace_elements]

            # add these to the trace
            runs_trace.extend(run_trace_elements)

            # then increase the cumulative n_frames for the next run
            cum_n_frames += n_frames

        return run_trace_elements

    def contig_cycle_idx(self, run_idx, cycle_idx):

        """Get the contig cycle idx for a (run_idx, cycle_idx) pair."""

        # make the contig trace
        contig_trace = self.get_branch_trace(run_idx, cycle_idx)

        # get the length and subtract one for the index
        return len(contig_trace) - 1


    def get_branch_trace(self, run_idx, cycle_idx, start_contig_idx=0):
        """Given an identifier of (run_idx, cycle_idx) from the contig tree
        and a starting contig index generate a contig trace of
        (run_idx, cycle_idx) indices for that contig. Which is a
        branch of the tree hence the name.

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

    def trace_parent_table(self, contig_trace):
        """Given a contig trace returns a parent table for that contig.

        """

        parent_table = []
        for run_idx, cycle_idx in contig_trace:
            parent_idxs = self.graph.node[(run_idx, cycle_idx)][self.PARENTS_KEY]
            parent_table.append(parent_idxs)

        return parent_table


    @classmethod
    def _tree_leaves(cls, root, tree):

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
        """All of the leaves of the contig forest"""
        leaves = []
        for root in self.roots():
            subtree_leaves = self._subtree_leaves(root)
            leaves.extend(subtree_leaves)

        return leaves

    def _subtree_root(self, node):
        """ Given a node find the root of the tree it is on"""

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

        subtree_roots = []
        for subtree in self.subtrees():
            # use a node from the subtree to get the root
            node = next(subtree.adjacency())[0]
            subtree_root = self._subtree_root(node)
            subtree_roots.append(subtree_root)

        return subtree_roots

    def subtrees(self):

        subtree_nxs = []
        for component_nodes in nx.weakly_connected_components(self.graph):

            # actually get the subtree from the main tree
            subtree = self.graph.subgraph(component_nodes)

            subtree_nxs.append(subtree)

        return subtree_nxs

    def get_subtree(self, node):

        # get all the subtrees
        subtrees = self.subtrees()

        # see which tree the node is in
        for subtree in subtrees:

            # if the node is in it this is the subtree it is in so
            # just return it
            if node in subtree:
                return subtree

    def contig_sliding_windows(self, contig_trace, window_length):
        """Given a contig trace (run_idx, cycle_idx) get the sliding windows
        over it (traj_idx, cycle_idx)."""

        # make a parent table for the contig trace
        parent_table = self.trace_parent_table(contig_trace)

        # this gives you windows of trace elements (traj_idx, cycle_idx)
        windows = sliding_window(parent_table, window_length)

        return windows

    def sliding_contig_windows(self, window_length):

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
        """All the sliding windows (run_idx, traj_idx, cycle_idx) for all
        contig windows in the contig tree"""

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
    def _spanning_paths(cls, edges, root):

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
            root_paths.extend(cls._spanning_paths(edges, new_root))

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

    def spanning_contig_traces(self):
        """Returns a list of all possible spanning contigs given the
        continuations present in this file. Contigs are a list of runs
        in the order that makes a continuous set of data. Spanning
        contigs are always as long as possible, thus all must start
        from a root and end at a leaf node.

        This algorithm always returns them in a canonical order (as
        long as the runs are not rearranged after being added). This
        means that the indices here are the indices of the contigs.

        Contigs can in general are any such path drawn from what we
        call the "contig tree" which is the tree (or forest of trees)
        generated by the directed edges of the 'continuations'. They
        needn't be spanning from root to leaf.

        """

        spanning_contig_traces = []
        # roots should be sorted already, so we just iterate over them
        for root in self.roots():

            # get the spanning paths by passing the continuation edges
            # and this root to this recursive static method
            root_spanning_contigs = self._spanning_paths(self.graph.edges, root)

            spanning_contig_traces.extend(root_spanning_contigs)

        return spanning_contig_traces

    @classmethod
    def _contig_trace_to_contig_runs(cls, contig_trace):

        contig_runs = []
        for run_idx, cycle_idx in contig_trace:

            if not run_idx in contig_runs:
                contig_runs.append(run_idx)
            else:
                pass

        return contig_runs

    @classmethod
    def _contig_runs_to_continuations(cls, contig_runs):

        continuations = []
        for i in range(len(contig_runs) - 1, 0, -1):
            continuations.append([contig_runs[i], contig_runs[i-1]])

        return continuations

    @classmethod
    def _continuations_to_contig_runs(cls, continuations):

        if len(continuations) == 0:
            return []

        continuations = list(copy(continuations))
        continuations.sort()

        contig_runs = []
        for next_run, continued_run in continuations:
            contig_runs.append(continued_run)

        contig_runs.append(continuations[-1][0])


        # since this is only valid if the continuations don't form a
        # tree check that we didn't put the same number in twice,
        # which this would indicate, fail if true
        assert len(set(contig_runs)) == len(contig_runs), \
            "this is not a single contig"

        return contig_runs

    def make_contig(self, contig_trace):

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


class Contig(ContigTree):

    def __init__(self, wepy_h5,
                 **kwargs):

        # use the superclass initialization
        super().__init__(wepy_h5, **kwargs)

        # check that the result is a single contig
        assert len(self.spanning_contig_traces()) == 1, \
            "continuations given do not form a single contig"

        # if so we add some useful attributes valid for only a
        # standalone contig

        # the contig_trace
        self._contig_trace = self.spanning_contig_traces()[0]

        # TODO this should not be part of the API in the future since
        # we don't want to have the run_idxs alone be how contigs are
        # defined (since we want contigs to be able to go to different
        # runs from the middle of other runs), in any case that is how
        # it is implemented now and there is no reason to change it
        # now, so we keep it hidden

        # the run idxs of the contig
        if len(self.contig_trace) > 1:
            self._contig_run_idxs = self._continuations_to_contig_runs(self.continuations)
        # if there is only 1 run we set it like this
        else:
            self._contig_run_idxs = list(self.run_idxs)


    @property
    def contig_trace(self):
        return self._contig_trace

    @property
    def n_cycles(self):
        return len(self.contig_trace)


    def contig_fields(self, fields):

        return self.wepy_h5.get_contig_trace_fields(self.contig_trace, fields)

    def records(self, record_key):
        return self.wepy_h5.contig_records(self._contig_run_idxs, record_key)

    def records_dataframe(self, record_key):
        return self.wepy_h5.contig_records_dataframe(self._contig_run_idxs, record_key)

    # resampling
    def resampling_records(self):

        return self.records(RESAMPLING)

    def resampling_records_dataframe(self):

        return pd.DataFrame(self.resampling_records())

    # resampler records
    def resampler_records(self):

        return self.records(RESAMPLER)

    def resampler_records_dataframe(self):

        return pd.DataFrame(self.resampler_records())

    # warping
    def warping_records(self):

        return self.records(WARPING)

    def warping_records_dataframe(self):

        return pd.DataFrame(self.warping_records())

    # boundary conditions
    def bc_records(self):

        return self.records(BC)

    def bc_records_dataframe(self):

        return pd.DataFrame(self.bc_records())

    # progress
    def progress_records(self):

        return self.records(PROGRESS)

    def progress_records_dataframe(self):

        return pd.DataFrame(self.progress_records())


    # resampling panel
    def resampling_panel(self):

        return self.wepy_h5.contig_resampling_panel(self._contig_run_idxs)


    def parent_table(self):

        return self.trace_parent_table(self.contig_trace)
