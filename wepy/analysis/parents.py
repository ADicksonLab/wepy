from copy import copy

import numpy as np
import networkx as nx

DISCONTINUITY_VALUE = -1

def resampling_panel(resampling_records, is_sorted=False):
    """Converts a simple collection of resampling records into a list of
    elements corresponding to cycles. It is like doing a pivot on
    the step indices into an extra dimension. Hence it can be
    thought of as a list of tables indexed by the cycle, hence the
    name panel.

    """

    res_panel = []

    # if the records are not sorted this must be done:
    if not is_sorted:
        resampling_records.sort()

    # iterate through the resampling records
    rec_it = iter(resampling_records)
    last_cycle_idx = None
    cycle_recs = []
    stop = False
    while not stop:

        # iterate through records until either there is none left or
        # until you get to the next cycle
        cycle_stop = False
        while not cycle_stop:
            try:
                rec = next(rec_it)
            except StopIteration:
                # this is the last record of all the records
                stop = True
                # this is the last record for the last cycle as well
                cycle_stop = True
                # alias for the current cycle
                curr_cycle_recs = cycle_recs
            else:

                # the cycles in the records may not start at 0, but
                # they are in order so we initialize the last
                # cycle_idx so we know when in the records we have
                # gotten to the next cycle of records
                if last_cycle_idx is None:
                    last_cycle_idx = rec.cycle_idx

                # if the resampling record retrieved is from the next
                # cycle we finish the last cycle
                if rec.cycle_idx > last_cycle_idx:
                    cycle_stop = True
                    # save the current cycle as a special
                    # list which we will iterate through
                    # to reduce down to the bare
                    # resampling record
                    curr_cycle_recs = cycle_recs

                    # start a new cycle_recs for the record
                    # we just got
                    cycle_recs = [rec]
                    last_cycle_idx += 1

            if not cycle_stop:
                cycle_recs.append(rec)

            else:

                # we need to break up the records in the cycle into steps
                cycle_table = []

                # temporary container for the step we are working on
                step_recs = []
                step_idx = 0
                step_stop = False
                cycle_it = iter(curr_cycle_recs)
                while not step_stop:
                    try:
                        cycle_rec = next(cycle_it)
                    # stop the step if this is the last record for the cycle
                    except StopIteration:
                        step_stop = True
                        # alias for the current step
                        curr_step_recs = step_recs

                    # or if the next stop index has been obtained
                    else:

                        if cycle_rec.step_idx > step_idx:
                            step_stop = True
                            # save the current step as a special
                            # list which we will iterate through
                            # to reduce down to the bare
                            # resampling record
                            curr_step_recs = step_recs

                            # start a new step_recs for the record
                            # we just got
                            step_recs = [cycle_rec]
                            step_idx += 1


                    if not step_stop:
                        step_recs.append(cycle_rec)
                    else:
                        # go through the walkers for this step since it is completed
                        step_row = [None for _ in range(len(curr_step_recs))]
                        for walker_rec in curr_step_recs:

                            # collect data from the record
                            walker_idx = walker_rec.walker_idx
                            decision_id = walker_rec.decision_id
                            instruction = walker_rec.target_idxs

                            # set the resampling record for the walker in the step records
                            step_row[walker_idx] = (decision_id, instruction)


                        # add the records for this step to the cycle table
                        cycle_table.append(step_row)

        # add the table for this cycles records to the parent panel
        res_panel.append(cycle_table)

    return res_panel


def parent_panel(decision_class, resampling_panel):

    parent_panel_in = []
    for cycle in resampling_panel:

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
        parent_panel_in.append(parent_table)

    return parent_panel_in

def net_parent_table(parent_panel):

    net_parent_table_in = []

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
        net_parent_table_in.append(step_net_parents)

    return net_parent_table_in

def parent_table_discontinuities(boundary_condition_class, parent_table, warping_records):
    """Given a parent table and warping records returns a new parent table
    with the discontinuous warping events for parents set to -1"""

    # Make a copy of the parent table
    new_parent_table = copy(parent_table)

    for warp_record in warping_records:

        cycle_idx = warp_record[0]
        parent_idx = warp_record[1]

        n_walkers = len(parent_table[cycle_idx])


        # Check to see if any walkers in the current step
        # originated from this warped walker
        for walker_idx in range(n_walkers):

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


class ParentForest():

    WEIGHT = 'weight'
    FREE_ENERGY = 'free_energy'

    ROOT_CYCLE_IDX = -1
    DISCONTINUITY_VALUE = -1

    def __init__(self, contig=None,
                 parent_table=None):

        if (contig is not None) and (parent_table is not None):
            raise ValueError("Pass in either a contig or a parent table but not both")

        if (contig is None) and (parent_table is None):
            raise ValueError("Must provide either a contig or a parent table")

        self._contig = contig

        # if the contig was given, generate a parent table
        if self.contig is not None:

            # from that contig make a parent table
            self._parent_table = self.contig.parent_table()

        # otherwise use the one given
        else:
            self._parent_table = parent_table

        self._graph = nx.DiGraph()

        self._n_steps = len(self.parent_table)

        # TODO this should be somehow removed so that we can
        # generalize to a variable number of walkers
        self._n_walkers = len(self.parent_table[0])

        # make the roots of each tree in the parent graph, because
        # this is outside the indexing of the steps we use a special
        # index
        self._roots = [(self.ROOT_CYCLE_IDX, i) for i in range(self.n_walkers)]

        # set these as nodes
        self.graph.add_nodes_from(self.roots)

        # go through the parent matrix and make edges from the parents
        # to children nodes
        for step_idx, parent_idxs in enumerate(self.parent_table):

            # make edge between each walker of this step to the previous step
            edges, edges_attrs = self._make_child_parent_edges(step_idx, parent_idxs)

            # then put them into this graph
            for i, edge in enumerate(edges):

                if edge is not None:

                    # if there is a discontinuity in the edge we only
                    # add the node
                    if self.DISCONTINUITY_VALUE == edge[1]:
                        self.graph.add_node(edge[0])
                    else:
                        self.graph.add_edge(*edge, **edges_attrs[i])

    def _make_child_parent_edges(self, step_idx, parent_idxs):

        edges = []
        edges_attrs = []
        for curr_walker_idx, parent_idx in enumerate(parent_idxs):

            # if the parent is the discontinuity value we set the
            # parent node in the edge as the discontinuity value
            if parent_idx == self.DISCONTINUITY_VALUE:
                parent_node = self.DISCONTINUITY_VALUE
                child_node = (step_idx, curr_walker_idx)
            else:
                # otherwise we make the edge with the parent and child as
                # normal
                parent_node = (step_idx - 1, parent_idx)
                child_node = (step_idx, curr_walker_idx)

            # make an edge between the parent of this walker and this walker
            edge = (parent_node, child_node)

            edges.append(edge)

            # nothing to do but I already wrote it this way and may be
            # useful later
            edge_attrs = {}

            edges_attrs.append(edge_attrs)

        return edges, edges_attrs

    @property
    def contig(self):
        return self._contig

    @property
    def parent_table(self):
        return self._parent_table

    @property
    def graph(self):
        return self._graph

    @property
    def roots(self):
        return self._roots

    @property
    def trees(self):
        trees_by_size = [self.graph.subgraph(c) for c in nx.weakly_connected_components(self.graph)]
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
        for node in self.graph.nodes:
            if node[0] == step_idx:
                step_nodes.append(node)

        step_nodes.sort()
        return step_nodes

    def steps(self):
        node_steps = []
        for step_idx in range(self.n_steps):
            node_steps.append(self.step(step_idx))

        return node_steps

    def walker(self, walker_idx):
        """ Get the nodes for this walker."""

        walker_nodes = []
        for node in self.graph.nodes:
            if node[1] == walker_idx:
                walker_nodes.append(node)

        walker_nodes.sort()
        return walker_nodes

    def walkers(self):
        node_walkers = []
        for walker_idx in range(self.n_walkers):
            node_walkers.append(self.walker(walker_idx))

        return node_walkers

    def set_node_attributes(self, attribute_key, node_attribute_dict):

        for node_id, value in node_attribute_dict.items():
            self.graph.nodes[node_id][attribute_key] = value

    def set_attrs_by_array(self, key, values):

        """Set attributes on a stepwise basis, i.e. expects a array/list that
        is n_steps long and has the appropriate number of values for
        the number of walkers at each step

        """
        for step in self.steps():
            for node in step:
                try:
                    self.graph.nodes[node][key] = values[node[0]][node[1]]
                except IndexError:
                    import ipdb; ipdb.set_trace()
