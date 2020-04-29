"""Routines for converting resampling data to parental lineages.

The core mechanism of weighted ensemble (WE) is to resample cohorts of
parallel simulations. Principally this means 'cloning' and 'merging'
different walkers which gives rise to branching 'family' tree
structures.

The routines in this module are concerned with utilizing raw
resampling records data from a resampler and extracting these lineages
in useful easy to query structures.

Cloning and merging is performed on a cohort of walkers at every
cycle. A walker has both a state and weight. The state is related to
the dynamical state such as the positions, velocities, etc. that
describe the state of a simulation. The weight corresponds to the
probability normalized with the other walkers of the cohort.

An n-clone constitutes copying a state n times to n walkers with w/n
weights where w is the weight of the cloned walker. The cloned walker
is said to be the parent of the new child walkers.

An k-merge constitutes combining k walkers into a single walker with a
weight that is the sum of all k walkers. The state of this walker is
chosen by sampling from the discrete distribution of the k
walkers. The walker that has it's state chosen to persist is referred
to as the 'kept' walker and the other walkers are said to be
'squashed', as they are compressed into a single walker, preserving
their weight but not their state. Squashed walkers are leaves of the
tree and leave no children. The kept walker and walkers for which no
cloning or merging was performed will have a single child.

Routines
--------

resampling_panel : Compiles an unordered collection of resampling records
    into a structured array of records.

parent_panel : Using a parental relationship reduce the records in a
    resampling panel to a structured array of parent indices.

net_parent_table : Reduce the full parent panel (with multiple steps
    per cycle) to the net results of resampling per cycle.

parent_table_discontinuties : Using an interpretation of warping
    records assigns a special value in a parent table for
    discontinuous warping events.

ancestors : Generate the lineage trace of a given walker back through
    history. This is used to get logically contiguous trajectories from
    walker slot trajectories (not necessarily contiguous in storage).

sliding_window : Generate non-redundant sliding window traces over the
    parent forest (tree) imposed over a parent table.

ParentForest : Class that imposes the forest (tree) structure over the
    parent table. Valid for a single contig in the contig tree (forest).
"""


from copy import copy
import itertools as it

import numpy as np
import networkx as nx

DISCONTINUITY_VALUE = -1
"""Special value used to determine if a parent-child relationship has
discontinuous dynamical continuity. Functions in this module uses this
to set this value.

"""

def resampling_panel(resampling_records, is_sorted=False):
    """Converts an unordered collection of resampling records into a
    structured array (lists) corresponding to cycles and resampling
    steps within cycles.

    It is like doing a pivot on the step indices into an extra
    dimension. Hence it can be thought of as a list of tables indexed
    by the cycle, hence the name panel.

    Parameters
    ----------
    resampling_records : list of nametuple records
        A list of resampling records.

    is_sorted : bool
        If this is True it will be assumed that the resampling_records
        are presorted, otherwise they will be sorted.

    Returns
    -------
    resampling_panel : list of list of list of namedtuple records
        The panel (list of tables) of resampling records in order
        (cycle, step, walker)

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
    """Using the parental interpretation of resampling records given by
    the decision_class, convert resampling records in a resampling
    panel to parent indices.

    Parameters
    ----------
    decision_class : class implementing Decision interface
        The class that interprets resampling records for parental relationships.
    resampling_panel : list of list of list of namedtuple records
        Panel of resampling records.

    Returns
    -------
    parent_panel : list of list of list of int
        A structured list of the same for as the resampling panel,
        with parent indices swapped for resampling records.

    """

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
    """Reduces a full parent panel to get parent indices on a cycle basis.

    The full parent panel has parent indices for every step in each
    cycle. This computes the net parent relationships for each cycle,
    thus reducing the list of tables (panel) to a single table. A
    table need not have the same length rows, i.e. a ragged array,
    since there can be different numbers of walkers each cycle.

    Parameters
    ----------
    parent_panel : list of list of list of int
        The full panel of parent relationships.

    Returns
    -------
    parent_table : list of list of int
        Net parent relationships for each cycle.

    """

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
    with the discontinuous warping events for parents set to a special
    value (-1).

    Parameters
    ----------
    boundary_condition_class : class implementing BoundaryCondition interface
        The boundary condition class that interprets warping
        records for if they are discontinuous or not.
    parent_table : list of list of int

    warping_records : list of namedtuple records
        The unordered collection of warping events from the simulation.

    Returns
    -------
    parent_table : list of list of int
        Same shape as input parent_table but with discontinuous
        relationships inserted as -1.

    """

    # Make a copy of the parent table
    new_parent_table = copy(parent_table)

    for warp_record in warping_records:

        cycle_idx = warp_record[0]
        parent_idx = warp_record[1]

        n_walkers = len(parent_table[cycle_idx])


        discs = [False for _ in range(n_walkers)]
        # Check to see if any walkers in the current step
        # originated from this warped walker
        for walker_idx in range(n_walkers):

            # this walker was warped discontinuously make a record true for itq
            if boundary_condition_class.warping_discontinuity(warp_record):
                discs[walker_idx] = True

        disc_parent_row = parent_cycle_discontinuities(parent_table[cycle_idx],
                                                       discs)

        new_parent_table[cycle_idx] = disc_parent_row


    return new_parent_table


def parent_cycle_discontinuities(parent_idxs, discontinuities):

    parent_row = copy(parent_idxs)
    for walker_idx, disc in enumerate(discontinuities):

        # if there was a discontinuity in this walker, we need to
        # check for which children it had and apply the discontinuity
        # to them
        if disc:
            for child_idx in range(len(parent_idxs)):

                parent_idx = parent_idxs[child_idx]

                if parent_idx == walker_idx:
                    parent_row[child_idx] = DISCONTINUITY_VALUE

    return parent_row

def ancestors(parent_table, cycle_idx, walker_idx, ancestor_cycle=0):
    """Returns the lineage of ancestors as walker indices leading up to
    the given walker.

    Parameters
    ----------
    parent_table : list of list of int
    cycle_idx : int
        Cycle of walker to query.
    walker_idx : int
        Walker index in to query along with cycle_idx.
    ancestor_cycle : int
        Index of cycle in history to go back to. Must be less than cycle_idx.

    Returns
    -------
    ancestor_trace : list of tuples of ints (traj_idx, cycle_idx)
        Contig walker trace of the ancestors leading up to the queried walker.
        The contig is sequence of cycles in the parent table.

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
    """Return contig walker traces of sliding windows of given length over
    the parent forest imposed over the contig given by the parent table.

    Windows are given in no particular order and are nonredundant for
    trees.

    Parameters
    ----------
    parent_table : list of list of int
        Parent table defining parent relationships and contig.
    window_length : int
        Length of window to use. Must be greater than 1.

    Returns
    -------
    windows : list of list of tuples of ints (traj_idx, cycle_idx)
        List of contig walker traces.
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
    """A tree abstraction to a contig representing the family trees of walkers.

    Uses a directed graph (networkx.DiGraph) to represent parent-child
    relationships; i.e. for edge (A, B) node A is a parent of node B.

    """


    WEIGHT = 'weight'
    """Key for weight node attribute."""
    FREE_ENERGY = 'free_energy'
    """Key for free energy node attribute."""

    ROOT_CYCLE_IDX = -1
    """Special value for the root nodes cycle indices which the initial
     conditions of the simulation

    """
    DISCONTINUITY_VALUE = -1
    """Special value used to determine if a parent-child relationship has
    discontinuous dynamical continuity. This class tests for this
    value in the parent table.
    """

    CONTINUITY_VALUE = 0

    def __init__(self, contig=None,
                 parent_table=None,
    ):
        """Constructs a parent forest from either a Contig object or parent table.

        Either a contig or parent_table must be given but not both.

        The underlying data structure used is a parent table. However,
        if a contig is given a reference to it will be kept.

        Arguments
        ---------
        contig : Conting object, optional conditional on parent_table

        parent_table : list of list of int, optional conditional on contig
            Must not contain the discontinuity values. If you want to
            include metadata on discontinuities use the contig input
            which is preferrable.

        Raises
        ------
        ValueError
            If neither parent_table nor contig is given, or if both are given.

        """


        if (contig is not None) and (parent_table is not None):
            raise ValueError("Pass in either a contig or a parent table but not both")

        if (contig is None) and (parent_table is None):
            raise ValueError("Must provide either a contig or a parent table")

        self._contig = contig

        # if the contig was given, generate a parent table
        if self._contig is not None:

            # from that contig make a parent table
            self._parent_table = self.contig.parent_table(discontinuities=False)

        # otherwise use the one given
        else:
            assert not self.DISCONTINUITY_VALUE in it.chain(*parent_table), \
                "Discontinuity values in parent table are not allowed."

            self._parent_table = parent_table

        self._graph = nx.DiGraph()

        # make the roots of each tree in the parent graph, because
        # this is outside the indexing of the steps we use a special
        # index
        self._roots = [(self.ROOT_CYCLE_IDX, i)
                       for i in range(len(self.parent_table[0]))]

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
                    self.graph.add_edge(*edge, **edges_attrs[i])

    def _make_child_parent_edges(self, step_idx, parent_idxs):
        """Generate edge_ids and edge attributes for an array of parent indices.

        Parameters
        ----------
        step_idx : int
            Index of step, just sets the value to put into the node_ids
        parent_idxs : list of int
            For element i in this list, the value j is the index of
            the child in slot i in step_idx+1. Thus j must be
            between 0 and len(parent_idxs)-1.

        Returns
        -------
        edges : list of 2-tuple of node_id
        edge_attributes : list of dict

        """

        edges = []
        edges_attrs = []
        for curr_walker_idx, parent_idx in enumerate(parent_idxs):

            # if the parent is the discontinuity value we set the
            # parent node in the edge as the discontinuity value
            disc = self.CONTINUITY_VALUE
            if parent_idx == self.DISCONTINUITY_VALUE:
                disc = self.DISCONTINUITY_VALUE

            # otherwise we make the edge with the parent and child as
            # normal
            parent_node = (step_idx - 1, parent_idx)
            child_node = (step_idx, curr_walker_idx)

            # make an edge between the parent of this walker and this walker
            edge = (parent_node, child_node)

            edges.append(edge)

            # nothing to do but I already wrote it this way and may be
            # useful later
            edge_attrs = {'discontinuity' : disc}

            edges_attrs.append(edge_attrs)

        return edges, edges_attrs

    @property
    def contig(self):
        """Underlying contig if given during construction."""
        if self._contig is None:
            raise AttributeError("No contig set.")
        else:
            return self._contig

    @property
    def parent_table(self):
        """Underlying parent table data structure."""
        return self._parent_table

    @property
    def graph(self):
        """Underlying networkx.DiGraph object."""
        return self._graph

    @property
    def roots(self):
        """Returns the roots of all the trees in this forest."""
        return self._roots

    @property
    def trees(self):
        """Returns a list of the subtrees from each root in this forest. In no particular order"""
        trees_by_size = [self.graph.subgraph(c) for c in nx.weakly_connected_components(self.graph)]
        trees = []
        for root in self.roots:
            root_tree = [tree for tree in trees_by_size if root in tree][0]
            trees.append(root_tree)
        return trees

    @property
    def n_steps(self):
        """Number of steps of resampling in the parent forest."""

        return len(self.parent_table)

    def step(self, step_idx):
        """Get the nodes at the step (level of the tree).

        Parameters
        ----------
        step_idx : int

        Returns
        -------
        nodes : list of node_id
        """

        step_nodes = []
        for i, node in enumerate(self.graph.nodes):
            if node[0] == step_idx:
                step_nodes.append(node)

        step_nodes.sort()
        return step_nodes

    def steps(self):
        """Returns the nodes ordered by the step and walker indices.

        Returns
        -------
        node_steps : list of list of node_id
        """
        node_steps = []
        for step_idx in range(self.n_steps):
            node_steps.append(self.step(step_idx))

        return node_steps

    def walker(self, walker_idx):
        """Get the nodes for this walker for the whole tree.

        Parameters
        ----------
        walker_idx : int

        Returns
        -------
        nodes : list of node_id

        """

        walker_nodes = []
        for node in self.graph.nodes:
            if node[1] == walker_idx:
                walker_nodes.append(node)

        walker_nodes.sort()
        return walker_nodes

    # TODO DEPRECATE: we shouldn't assume constant number of walkers
    # def walkers(self):
    #     """ """
    #     node_walkers = []
    #     for walker_idx in range(self.n_walkers):
    #         node_walkers.append(self.walker(walker_idx))

    #     return node_walkers

    def set_node_attributes(self, attribute_key, node_attribute_dict):
        """Set attributes for all nodes for a single key.

        Parameters
        ----------
        attribute_key : str
            Key to set for a node attribute.
        node_attribute_dict : dict of node_id: value
            Dictionary mapping nodes to the values that will be set
            for the attribute_key.

        """

        for node_id, value in node_attribute_dict.items():
            self.graph.nodes[node_id][attribute_key] = value

    def set_attrs_by_array(self, attribute_key, values):
        """Set node attributes on a stepwise basis using structural indices.

        Expects a array/list that is n_steps long and has the
        appropriate number of values for the number of walkers at each
        step.

        Parameters
        ----------
        attribute_key : str
            Key to set for a node attribute.
        values : array_like of dim (n_steps, n_walkers) or list of
                 list of values.
            Either an array_like if there is a constant number of
            walkers or a list of lists of the values.
        """
        for step in self.steps():
            for node in step:
                self.graph.nodes[node][attribute_key] = values[node[0]][node[1]]
