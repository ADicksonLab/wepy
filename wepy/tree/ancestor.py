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
                  cycle and walker parents

    """

    ancestors = [(cycle_idx, walker_idx)]

    previous_walker = walker_idx

    for cycle_idx in range(cycle_idx, ancestor_cycle , -1):
            previous_walker = parent_matrix[cycle_idx, previous_walker]
            previous_point = (cycle_idx - 1, previous_walker)
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

    acestor_matrix (2D numpy array): A matrix of the ancestor walker index
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

            # edge = (step_idx - 1) * len(positional_data_storage[0]) +
            #            parent_matrix[step_idx, beads_1],
            #            ((step_idx) * len(positional_data_storage[0]) + beads_1)


    return graph
