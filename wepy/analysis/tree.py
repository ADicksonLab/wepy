import networkx as nx
import numpy as np

def parent_panel(decision_class, resampling_panel):

    parent_panel = []
    for cycle_idx, cycle in enumerate(resampling_panel):

        # each stage in the resampling for that cycle
        # make a stage parent table
        parent_table = []

        # now iterate through the rest of the stages
        for step in cycle:

            # get the parents idxs for the children of this step
            step_parents = decision_class.parents(step)

            # for the full stage table save all the intermediate parents
            parent_table.append(step_parents)

        # for the full parent panel
        parent_panel.append(parent_table)

    return parent_panel

def net_parent_table(parent_panel):

    net_parent_table = []

    # each cycle
    for cycle_idx, step_parent_table in enumerate(parent_panel):
        # for the net table we only want the end results,
        # we start at the last cycle and look at its parent
        step_net_parents = []
        n_steps = len(step_parent_table)
        for walker_idx, parent_idx in enumerate(step_parent_table[-1]):
            # initialize the root_parent_idx which will be updated
            root_parent_idx = parent_idx

            # if no resampling skip the loop and just return the idx
            if n_steps > 0:
                # go back through the steps getting the parent at each step
                for prev_step_idx in range(n_steps):
                    prev_step_parents = step_parent_table[-(prev_step_idx+1)]
                    root_parent_idx = prev_step_parents[root_parent_idx]

            # when this is done we should have the index of the root parent,
            # save this as the net parent index
            step_net_parents.append(root_parent_idx)

        # for this step save the net parents
        net_parent_table.append(step_net_parents)

    return net_parent_table

def ancestors(parent_table, cycle_idx, walker_idx, ancestor_cycle=0):
    """Given a parent table, step_idx, and walker idx returns the
    ancestor at a given cycle of the walker.

    Input:

    parent: table (n_cycles x n_walkers numpy array): It describes
    how the walkers merged and cloned during the WExplore simulation .

    cycle_idx:

    walker_idx:

    ancestor_cycle:


    Output:

    ancestors: A list of 2x1 tuples indicating the
                   walker and cycle parents

    """

    lineage = [(walker_idx, cycle_idx)]

    previous_walker = walker_idx

    for curr_cycle_idx in range(cycle_idx-1, ancestor_cycle-1, -1):
            previous_walker = parent_table[curr_cycle_idx][previous_walker]

            # check for discontinuities, e.g. warping events
            if previous_walker == -1:
                # there are no more continuous ancestors for this
                # walker so we cannot return ancestors back to the
                # requested cycle just return the ancestors to this
                # point
                break

            previous_point = (previous_walker, curr_cycle_idx)
            lineage.insert(0, previous_point)

    return lineage

def sliding_window(parent_table, window_length):
    """Returns traces (lists of frames across a run) on a sliding window
    on the branching structure of a run of a WepyHDF5 file. There is
    no particular order guaranteed.

    """

    # assert parent_table.dtype == np.int, \
    #     "parent table values must be integers, not {}".format(parent_table.dtype)

    assert window_length > 1, "window length must be greater than one"

    windows = []
    # we make a range iterator which goes from the last cycle to the
    # cycle which would be the end of the first possible sliding window
    for cycle_idx in range(len(parent_table)-1, window_length-2, -1):
        # then iterate for each walker at this cycle
        for walker_idx in range(len(parent_table[0])):

            # then get the ancestors according to the sliding window
            window = ancestors(parent_table, cycle_idx, walker_idx,
                               ancestor_cycle=cycle_idx-(window_length-1))

            # if the window is too short because the lineage has a
            # discontinuity in it skip to the next window
            if len(window) < window_length:
                continue

            windows.append(window)

    return windows

## contig trees for of parent panels

def cycle_tree_parent_table(decision_class, cycle_tree):
    """Determines the net parents for each cycle and sets them in-place to
    the cycle tree given."""

    # just go through each node individually in the tree
    for node in cycle_tree:
        # get the records for each step in this node
        node_recs = cycle_tree.nodes[node]['resampling_steps']

        # get the node parent table by using the parent panel method
        # on the node records
        node_parent_panel = parent_panel(decision_class, [node_recs])
        # then get the net parents from this parent panel, and slice
        # out the only entry from it
        node_parents = net_parent_table(node_parent_panel)[0]

        # put this back into the cycle_tree
        cycle_tree.nodes[node]['parent_idxs'] = node_parents

    return cycle_tree

def cycle_tree_contig_trace(cycle_tree, run_idx, cycle_idx, start_contig_idx=0):
    """Given a single cycle tree and a (run_idx, cycle_idx) in that tree
    and a starting contig index generate a contig trace of (run_idx,
    cycle_idx) indices for that contig.

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
        parent_nodes = list(cycle_tree.adj[curr_node].keys())

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


def contig_sliding_window(cycle_tree, contig_trace, window_length):
    """Given a contig trace (run_idx, cycle_idx) get the sliding windows
    over it (run_idx, traj_idx, cycle_idx)."""

    # make a parent table for the contig trace
    parent_table = contig_trace_parent_table(cycle_tree, contig_trace)

    windows = sliding_window(parent_table, window_length)

    return windows


def contig_cycle(cycle_tree, run_idx, cycle_idx):
    """Get the contig cycle idx for a (run_idx, cycle_idx) pair."""

    # make the contig trace
    contig_trace = cycle_tree_contig_trace(cycle_tree, run_idx, cycle_idx)

    # get the length and subtract one for the index
    return len(contig_trace) - 1

def _tree_leaves(root, tree):

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
        branch_leaves = _tree_leaves(branch_child_node, tree)
        leaves.extend(branch_leaves)

    return leaves

def cycle_tree_leaves(root, cycle_tree):

    # get the reversed directions of the cycle tree as a view, we
    # don't need a copy
    rev_tree = cycle_tree.reverse(copy=False)

    # then we use the adjacencies to find the last node in the network
    # using a recursive algorithm
    leaves = _tree_leaves(root, rev_tree)

    return leaves

def cycle_tree_root(cycle_tree):


    # just pick a random node, the first one
    curr_node = list(cycle_tree.adjacency())[0][0]

    # we also get the adjacent nodes for this node
    adj_nodes = list(cycle_tree.adj[curr_node].keys())

    # there should only be one node in the dict
    assert len(adj_nodes) <= 1, "There should be at most 1 edge"

    # then we use this node as the starting point to move back to the
    # root, we end when there is no adjacent node
    while len(adj_nodes) > 0:

        # we take another step backwards, and choose the node in the
        # adjacency
        adj_nodes = list(cycle_tree.adj[curr_node])

        # there should only be 1 or none nodes
        assert len(adj_nodes) <= 1, "There should be at most 1 edge"

        # and reset the current node
        try:
            curr_node = adj_nodes[0]
        except IndexError:
            # this happens when this is the last node, inelegant apologies
            pass

    return curr_node

def contig_trace_parent_table(cycle_tree, contig_trace):
    """Given a cycle tree with parents and a contig trace returns a parent
    table for that contig."""

    parent_table = []
    for run_idx, cycle_idx in contig_trace:
        parent_idxs = cycle_tree.node[(run_idx, cycle_idx)]['parent_idxs']
        parent_table.append(parent_idxs)

    return parent_table

def ancestors_from_tree(cycle_tree, run_idx, cycle_idx, walker_idx, ancestor_node=None):

    raise NotImplementedError

    # make the contig trace for this to the root
    contig_trace = cycle_tree_contig_trace(cycle_tree, run_idx, cycle_idx)

    # first we generate the parent table given the walker we want
    # parents from
    parent_table = parent_table_from_tree(cycle_tree, run_idx, cycle_idx)

    # get the cycle index within the parent table (contig) and not the
    # one given for within the run
    contig_cycle_idx = run_cycle_to_spanning_contig_cycle(cycle_tree, run_idx, cycle_idx)

    # the same for the index of the ancestor node if it is given
    if ancestor_node is not None:
        ancestor_contig_cycle_idx = run_cycle_to_spanning_contig_cycle(cycle_tree, *ancestor_node)
    else:
        ancestor_contig_cycle_idx = 0

    # then we just perform a normal ancestors function on that table
    # using a corrected cycle index from the contig
    return ancestors(parent_table, contig_cycle_idx, walker_idx, ancestor_contig_cycle_idx)

def sliding_contig_windows(cycle_tree, window_length):

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

    # first we need to find the root and leaves for this tree
    root = cycle_tree_root(cycle_tree)
    leaves = cycle_tree_leaves(root, cycle_tree)

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

    # initialize the list of active branches
    branch_contigs = [cycle_tree_contig_trace(cycle_tree, *leaf) for leaf in leaves]

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

def forest_sliding_contig_windows(cycle_forest, window_length):

    assert window_length > 1, "window length must be greater than one"

    # we can deal with each tree in this forest of trees separately,
    # that is runs that are not connected
    forest_contig_windows = []
    for component_nodes in nx.weakly_connected_components(cycle_forest):

        # actually get the subtree from the main tree
        subtree = cycle_forest.subgraph(component_nodes)

        # get the contig windows for the individual tree
        subtree_contig_windows = sliding_contig_windows(subtree, window_length)

        forest_contig_windows.extend(subtree_contig_windows)

    return forest_contig_windows

def contig_tree_sliding_windows(cycle_tree, window_length):

    # get all of the contig traces for these trees
    contig_traces = forest_sliding_contig_windows(cycle_tree, window_length)

    # for each of these we generate all of the actual frame sliding windows
    windows = []
    for contig_trace in contig_traces:
        contig_windows = contig_sliding_window(cycle_tree, contig_trace, window_length)
        windows.extend(contig_windows)

    return windows
