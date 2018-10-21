
import numpy as np

def _overlaps(positions, node_idx, node_radius):

        # get the nodes that this one overlaps with
        overlaps = []
        for other_node in positions:

            # you can't overlap yourself
            if node_idx == other_node:
                overlaps.append(False)
            else:
                # check if there is an overlap between nodes
                if np.abs(positions[node_idx] - positions[other_node]) < node_radius:
                    overlaps.append(True)
                else:
                    overlaps.append(False)

        return overlaps

def simple_next_gen(parents_x, children_parent_idxs,
                     fanning_factor, node_radius):

    # initialize the childrens x row list with the positions of their
    # parent
    children_x = [parents_x[parent_idx] for parent_idx in children_parent_idx]


    # to place the children such that the edges never overlap we first
    # layout the nodes within their parent group, then we treat the
    # parent groups as single larger nodes and lay them out together.

    # This part of the code looks for the same numbers in the vector,
    # and changes the distance by 0.05 increments, to get the branch
    # effect. First value goes to the left of the center, the next
    # value goes to the right of the center.
    for node_idx in range(len(x)):


        # 
        counter_1 = 0
        counter_2 = 0
        while any(_overlaps(children_x, node_idx, node_radius)):
            counter_2 += 1
            counter_1 += 1
            if counter_1 % 2 == 0:
                children_x[node_idx] = children_x[node_idx] + counter_2 * fanning_factor
            else:
                children_x[node_idx] = children_x[node_idx] - counter_2 * fanning_factor

    return children_x

def initial_parent_distribution(n_nodes,
                                spacing_factor=0.01, node_radius=3):

    return np.linspace(spacing_factor * (-n_nodes + 1),
                       spacing_factor * (n_nodes - 1),
                       n_nodes)


def add_generation(new_children_parents,
                   parent_table, old_positions,
                   node_radius=3, fanning_factor=1.5):

    n_new_children = len(new_children_parents)
    cycle_idx = len(old_positions)

    # make a new array for the parent table and positions with an
    # extra row
    new_parent_table = np.zeros((parent_table.shape[0] + 1,
                                 parent_table.shape[1]))

    new_positions = np.zeros((old_positions.shape[0] + 1,
                              old_positions.shape[1], old_positions.shape[2]))

    # then set the old positions to what they were
    new_parent_table[0:-1] = parent_table
    new_positions[0:-1] = old_positions

    # set the new parents row for these children into this new table
    new_parent_table[-1] = new_children_parents

    # now we need to get the positions for the next generation

    # get the x coordinates of the last generation (parents of new children)
    parents_x = old_positions[-1, 0]

    # generate the x positions for the children
    children_x = simple_next_gen(parents_x, new_children_parents,
                                 fanning_factor, node_radius)

    # set this into the new positions table
    new_positions[-1, 0] = children_x

    # then generate the y values for this (the cycle/generation index)
    children_y = np.array([cycle_idx for i in range(n_new_children)])

    new_positions[-1, 1] = children_y


    return new_parent_table, new_positions


def simple(parent_table,
           spacing_factor=0.01,
           node_radius=3, fanning_factor=1.5):

    n_timesteps = parent_table.shape[0]
    n_walkers = parent_table.shape[1]

    # initialize the positions to zeros
    node_positions = np.zeros((n_timesteps+1, n_walkers, 3))

    # initialize the first generation (cycle) node positions
    first_gen_positions = np.linspace(spacing_factor * (-n_walkers + 1),
                                             spacing_factor * (n_walkers - 1),
                                             n_walkers)

    # save them as full coordinates for visualization
    node_positions[0] = np.array([[x, 0.0, 0.0] for x in node_positions])

    # propagate nodes for the second step, this will seed the process
    # which can be done iteratively following this
    last_gen_positions = first_gen_positions
    curr_gen_positions = simple_next_gen(node_positions, parent_table[0],
                                             fanning_factor=fanning_factor,
                                             node_radius=node_radius)

    # # add in the other dimensions and save for output and visualization
    # cycles_node_positions[1] = np.array([np.array([x, 1.0, 0.0])
    #                                      for x in node_positions])

    # propagate and minimize the rest of the step nodes
    for step_idx in range(n_timesteps):

        curr_gen_positions = simple_next_gen(last_gen_positions,
                                             parent_table[step_idx],
                                             fanning_factor, node_radius)


        # generate the starting positions for the next generation nodes, 
        node_positions[step_idx] = np.array([np.array([x, float(step_idx), 0.0])
                                                    for x in ])


        # continue with propagating positions (along x) for the next steps
        node_positions_previous = node_positions

    return cycles_node_positions


def steepest_descent():
    pass
