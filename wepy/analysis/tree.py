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


def parent_graph(parent_table):

    graph = nx.Graph()

    n_walkers = parent_table.shape[1]

    for step_idx, parent_idxs in enumerate(parent_table):

        # the first step is useless
        if step_idx == 0:
            continue

        # make edge between each walker of this step to the previous step
        for curr_walker_idx in range(n_walkers):

            # make an edge between the parent of this walker and this walker
            edge = ((step_idx, curr_walker_idx), (step_idx - 1, parent_idxs[curr_walker_idx]))

            graph.add_edge(*edge)

    return graph

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

    for curr_cycle_idx in range(cycle_idx, ancestor_cycle, -1):
            previous_walker = parent_table[curr_cycle_idx][previous_walker]

            # check for discontinuities, e.g. warping events
            if previous_walker == -1:
                # there are no more continuous ancestors for this
                # walker so we cannot return ancestors back to the
                # requested cycle just return the ancestors to this
                # point
                break

            previous_point = (previous_walker, curr_cycle_idx - 1)
            lineage.insert(0, previous_point)

    return lineage

def parent_table_from_tree(cycle_tree, run_idx, cycle_idx):
    # to make this easy we just first generate a parent table from the
    # tree that is relevant for the query
    parent_table = []

    # initialize the current node
    curr_node = (run_idx, cycle_idx)

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

        # if there are any parents there should only be one, since it is a tree
        prev_node = parent_nodes[0]

        # get the parents from the previous node
        parents = cycle_tree.nodes[prev_node]['parent_idxs']

        # add these parents to the parent table at the beginning
        parent_table.insert(0, parents)

        # then update the current node to the previous one
        curr_node = prev_node

    return parent_table

def run_cycle_to_spanning_contig_cycle(cycle_tree, run_idx, cycle_idx):
    # initialize the current node
    curr_node = (run_idx, cycle_idx)

    # the cycle idxs for the whole contig, this will tally the number
    # of movements back through the tree
    contig_cycle_idx = 0

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

        # if there are any parents there should only be one, since it is a tree
        prev_node = parent_nodes[0]

        # add to the count of cycles
        contig_cycle_idx += 1

        # then update the current node to the previous one
        curr_node = prev_node

    return contig_cycle_idx


def ancestors_from_tree(cycle_tree, run_idx, cycle_idx, walker_idx, ancestor_node=None):

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

def ancestor_table(parent_table, ancestor_cycle=0):
    """Given a parent table and a cycle index, will return a table that
    gives the index of the walker that was the ancestor to the walker.

    The table wil be of size (n_cycles - ancestor_cycle, n_walkers),
    because only walkers after the ancestor_cycle will be assigned
    values. Walkers at the ancestor cycle index will be assigned their own
    index.

    Defaults to the zeroth cycle which is the initial walkers.


    This function will create a table that tracks the ultimate
    parent of the walkers. Each row is a timestep, and the
    value in the table corresponds to the original walker
    that the walker was associated from at the start of the
    MD simulation.

    Inputs:

    parent_table (2D numpy array): A table that shows how walkers from different time
    steps are related to each other.

    Outputs:

    ancestor_table (2D numpy array): A table of the ancestor walker index
    for walkers after the ancestor cycle idx.

    """

    n_steps = parent_table.shape[0]
    n_walkers = parent_table.shape[1]
    ancestor_mat = np.zeros((n_steps - ancestor_cycle, n_walkers))

    for step_idx in range(ancestor_cycle, n_steps):

        for walker_idx in range(n_walkers):

            # if this is the ancestor cycel we set the ancestor idx to itself
            if step_idx == ancestor_cycle:
                ancestor_mat[step_idx, walker_idx] = walker_idx

            else:
                ancestor_mat[step_idx, walker_idx] = \
                                        ancestor_mat[step_idx - 1,
                                                        parent_table[step_idx, walker_idx]]

    return ancestor_mat



def sliding_window(parent_table, window_length):
    """Returns traces (lists of frames across a run) on a sliding window
    on the branching structure of a run of a WepyHDF5 file. There is
    no particular order guaranteed.

    """

    assert parent_table.dtype == np.int, \
        "parent table values must be integers, not {}".format(parent_table.dtype)

    assert window_length > 1, "window length must be greater than one"

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

            yield window

def cycle_tree_leaves(root, cycle_tree):

    # get the reversed directions of the cycle tree as a view, we
    # don't need a copy
    rev_tree = cycle_tree.reverse(copy=False)

    # then

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

def cycle_tree_sliding_windows(cycle_tree, window_length):

    # to generate all the sliding windows over a connected cycle tree
    # it is useful to think of it as a braid, since within this tree
    # there is a forest of trees which are the lineages of the
    # walkers. To simplify things we can first generate window traces
    # over the cycle tree (ignoring the fin structure of the walkers),
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

    # first we need to find the root


def cycle_forest_sliding_windows(cycle_forest, window_length):

    assert window_length > 1, "window length must be greater than one"

    # we can deal with each tree in this forest of trees separately,
    # that is runs that are not connected
    forest_windows = []
    for component_nodes in nx.weakly_connected_components(cycle_forest):

        # actually get the subtree from the main tree
        cycle_tree = cycle_forest.subgraph(component_nodes)

        # then we can get the windows on this connected tree
        subtree_windows = cycle_forest_sliding_windows(cycle_tree, window_length)

        forest_windows.extend(subtree_windows)

    return forest_windows
