from copy import copy

import numpy as np
import networkx as nx

DISCONTINUITY_VALUE = -1

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

def parent_table_discontinuities(boundary_condition_class, parent_table, warping_records):
    """Given a parent table and warping records returns a new parent table
    with the discontinuous warping events for parents set to -1"""

    # Make a copy of the parent table
    new_parent_table = copy(parent_table)

    # Find the number of walkers and cycles
    n_walker = np.shape(parent_table)[1]

    for rec_idx, warp_record in enumerate(warping_records):

        cycle_idx = warp_record[0]
        parent_idx = warp_record[1]

        # Check to see if any walkers in the current step
        # originated from this warped walker
        for walker_idx in range(n_walker):

            # if it's parent is the walker in this warping event
            # we also need to check to see if that warping event
            # was a discontinuous warping event
            if parent_table[cycle_idx][walker_idx] == parent_idx:

                # just check by using the method from the boundary
                # condition class used
                if boundary_condition_class.warping_discontinuity(warp_record):

                    # set the element in the parent table to the
                    # discontinuity value if it is
                    new_parent_table[cycle_idx][walker_idx] = DISCONTINUITY_VALUE

    return new_parent_table

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


class ParentForest(nx.DiGraph):
    def __init__(self, parent_table):
        super().__init__()

        self._n_steps = parent_table.shape[0]
        self._n_walkers = parent_table.shape[1]

        # make the roots of each tree in the parent graph, the step is
        # 0
        self._roots = [(0, i) for i in range(self.n_walkers)]



        # set these as nodes
        self.add_nodes_from(self.roots)

        # go through the parent matrix and make edges from the parents
        # to children nodes
        for step_idx, parent_idxs in enumerate(parent_table):
            step_idx += 1

            # make edge between each walker of this step to the previous step
            for curr_walker_idx in range(self.n_walkers):

                # get the parent index
                parent_idx = parent_idxs[curr_walker_idx]

                # if it is a -1 indicating a discontinuity we add an
                # attribute indicating this is a discontinuity
                discontinuity = False
                if parent_idx == -1:
                    discontinuity = True

                parent_node = (step_idx - 1, parent_idx)
                child_node = (step_idx, curr_walker_idx)

                # make an edge between the parent of this walker and this walker
                edge = (parent_node, child_node)


                self.add_edge(*edge, discontinuous=discontinuity)

    @property
    def roots(self):
        return self._roots

    @property
    def trees(self):
        trees_by_size = [self.subgraph(c) for c in nx.weakly_connected_components(self)]
        trees = []
        for root in self.roots:
            root_tree = [tree for tree in trees_by_size if root in tree][0]
            trees.append(root_tree)
        return trees

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def n_walkers(self):
        return self._n_walkers

    def step(self, step_idx):
        """ Get the nodes at the step (level of the tree)."""

        step_nodes = []
        for node in self.nodes:
            if node[0] == step_idx:
                step_nodes.append(node)

        step_nodes.sort()
        return step_nodes

    def steps(self):
        node_steps = []
        for step_idx in range(self.n_steps + 1):
            node_steps.append(self.step(step_idx))

        return node_steps

    def walker(self, walker_idx):
        """ Get the nodes for this walker."""

        walker_nodes = []
        for node in self.nodes:
            if node[1] == walker_idx:
                walker_nodes.append(node)

        walker_nodes.sort()
        return walker_nodes

    def walkers(self):
        node_walkers = []
        for walker_idx in range(self.n_walkers):
            node_walkers.append(self.walker(walker_idx))

        return node_walkers

    def set_step_attrs(self, key, values):
        """Set attributes on a stepwise basis, i.e. expects a array/list that
        is n_steps long and has the appropriate number of values for
        the number of walkers at each step

        """
        for step in self.steps():
            for node in step:
                self.nodes[node][key] = values[node[0]][node[1]]

    def set_step_viz_properties(self, positions):
        pass

    def set_step_positions(self, positions):
        for step in self.steps():
            for node in step:
                coord = positions[node[0]][node[1]]
                viz_dict = {'position' : {'x' : float(coord[0]),
                                           'y' : float(coord[1]),
                                           'z' : float(coord[2])}
                           }

                self.nodes[node]['viz'] = viz_dict

    def set_step_colors(self, color_values):
        pass

    def set_step_sizes(self, sizes):
        pass
