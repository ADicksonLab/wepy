import networkx as nx
import numpy as np

def ancestors(parent_matrix, cycle_idx, walker_idx, ancestor_cycle=0):
    """Given a parent matrix, step_idx, and walker idx returns the
    ancestor at a given cycle of the walker.

    Input:

    parent: matrix (n_cycles x n_walkers numpy array): It describes
    how the walkers merged and cloned during the WExplore simulation .

    cycle_idx:

    walker_idx:

    ancestor_cycle:


    Output:

    ancestors: A list of 2x1 tuples indicating the
                   walker and cycle parents

    """

    ancestors = [(walker_idx, cycle_idx)]

    previous_walker = walker_idx

    for curr_cycle_idx in range(cycle_idx, ancestor_cycle, -1):
            previous_walker = parent_matrix[curr_cycle_idx][previous_walker]

            # check for discontinuities, e.g. warping events
            if previous_walker == -1:
                # there are no more continuous ancestors for this
                # walker so we cannot return ancestors back to the
                # requested cycle just return the ancestors to this
                # point
                break

            previous_point = (previous_walker, curr_cycle_idx - 1)
            ancestors.insert(0, previous_point)

    return ancestors

def ancestor_matrix(parent_matrix, ancestor_cycle=0):
    """Given a parent matrix and a cycle index, will return a matrix that
    gives the index of the walker that was the ancestor to the walker.

    The matrix wil be of size (n_cycles - ancestor_cycle, n_walkers),
    because only walkers after the ancestor_cycle will be assigned
    values. Walkers at the ancestor cycle index will be assigned their own
    index.

    Defaults to the zeroth cycle which is the initial walkers.


    This function will create a matrix that tracks the ultimate
    parent of the walkers. Each row is a timestep, and the
    value in the matrix corresponds to the original walker
    that the walker was associated from at the start of the
    MD simulation.

    Inputs:

    parent_matrix (2D numpy array): A matrix that shows how walkers from different time
    steps are related to each other.

    Outputs:

    ancestor_matrix (2D numpy array): A matrix of the ancestor walker index
    for walkers after the ancestor cycle idx.

    """

    n_steps = parent_matrix.shape[0]
    n_walkers = parent_matrix.shape[1]
    ancestor_mat = np.zeros((n_steps - ancestor_cycle, n_walkers))

    for step_idx in range(ancestor_cycle, n_steps):

        for walker_idx in range(n_walkers):

            # if this is the ancestor cycel we set the ancestor idx to itself
            if step_idx == ancestor_cycle:
                ancestor_mat[step_idx, walker_idx] = walker_idx

            else:
                ancestor_mat[step_idx, walker_idx] = \
                                        ancestor_mat[step_idx - 1,
                                                        parent_matrix[step_idx, walker_idx]]

    return ancestor_mat

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


def parent_graph(parent_matrix):

    graph = nx.Graph()

    n_walkers = parent_matrix.shape[1]

    for step_idx, parent_idxs in enumerate(parent_matrix):

        # the first step is useless
        if step_idx == 0:
            continue

        # make edge between each walker of this step to the previous step
        for curr_walker_idx in range(n_walkers):

            # make an edge between the parent of this walker and this walker
            edge = ((step_idx, curr_walker_idx), (step_idx - 1, parent_idxs[curr_walker_idx]))

            graph.add_edge(*edge)

    return graph


def sliding_window(parent_matrix, window_length):
    """Returns traces (lists of frames across a run) on a sliding window
    on the branching structure of a run of a WepyHDF5 file. There is
    no particular order guaranteed.

    """

    assert parent_matrix.dtype == np.int, \
        "parent matrix values must be integers, not {}".format(parent_matrix.dtype)

    assert window_length > 1, "window length must be greater than one"

    # we make a range iterator which goes from the last cycle to the
    # cycle which would be the end of the first possible sliding window
    for cycle_idx in range(len(parent_matrix)-1, window_length-2, -1):

        # then iterate for each walker at this cycle
        for walker_idx in range(len(parent_matrix[0])):

            # then get the ancestors according to the sliding window
            window = ancestors(parent_matrix, cycle_idx, walker_idx,
                               ancestor_cycle=cycle_idx-(window_length-1))

            # if the window is too short because the lineage has a
            # discontinuity in it skip to the next window
            if len(window) < window_length:
                continue

            yield window
