"""Module that allows for imposing a kinetically connected network
structure of weighted ensemble simulation data.
"""

from collections import defaultdict
from copy import deepcopy

import numpy as np
import networkx as nx

from wepy.analysis.transitions import transition_counts, counts_d_to_matrix, \
                                      normalize_counts

try:
    import pandas as pd
except ModuleNotFoundError:
    print("Pandas is not installe, that functionality won't work")

class MacroStateNetworkError(Exception):
    """Errors specific to MacroStateNetwork requirements."""
    pass

class BaseMacroStateNetwork():
    """A base class for the MacroStateNetwork which doesn't contain a
    WepyHDF5 object. Useful for serialization of the object and can
    then be reattached later to a WepyHDF5.

    """

    ASSIGNMENTS = 'assignments'
    """Key for the microstates that are assigned to a macrostate."""

    def __init__(self,
                 contig_tree,
                 assg_field_key=None,
                 assignments=None,
                 transition_lag_time=2):
        """Create a network of macrostates from the simulation microstates
        using a field in the trajectory data or precomputed assignments.

        Either 'assg_field_key' or 'assignments' must be given, but not
        both.

        The lag time is default set to 2, which is the natural connection
        between microstates. The lag time can be increased to vary the
        kinetic accuracy of transition probabilities generated through
        Markov State Modelling.

        Alternatively the MacroStateNetwork can also be though of as just
        a way of mapping macrostate properties to the underlying
        microstate data.

        Arguments
        ---------
        contig_tree : ContigTree object

        assg_field_key : str, conditionally optional on 'assignments'
            The field in the WepyHDF5 dataset you want to generate macrostates for.

        assignments : list of list of array_like of dim (n_traj_frames, observable_shape[0], ...),
                      conditionally optional on 'assg_field_key'

            List of assignments for all frames in each run, where each
            element of the outer list is for a run, the elements of
            these lists are lists for each trajectory which are
            arraylikes of shape (n_traj, observable_shape[0], ...).
        """

        self._graph = nx.DiGraph()

        assert not (assg_field_key is None and assignments is None), \
            "either assg_field_key or assignments must be given"

        assert assg_field_key is not None or assignments is not None, \
            "one of assg_field_key or assignments must be given"

        self._base_contig_tree = contig_tree.base_contigtree

        self._assg_field_key = assg_field_key

        # initialize the groups dictionary
        self._node_groups = {}

        # initialize the list of the observables
        self._observables = []

        # initialize the list of available layouts
        self._layouts = []

        # the temporary assignments dictionary
        self._node_assignments = None
        # and temporary raw assignments
        self._assignments = None

        with contig_tree:


            # map the keys to their lists of assignments, depending on
            # whether or not we are using a field from the HDF5 traj or
            # assignments provided separately
            if assg_field_key is not None:
                assert type(assg_field_key) == str, "assignment key must be a string"

                self._key_init(contig_tree)
            else:
                self._assignments_init(assignments)

            # once we have made th dictionary add the nodes to the network
            # and reassign the assignments to the nodes
            self._node_idxs = {}
            for node_idx, assg_item in enumerate(self._node_assignments.items()):
                assg_key, assigs = assg_item
                # count the number of samples (assigs) and use this as a field as well
                num_samples = len(assigs)

                # save the nodes with attributes, we save the node_id
                # as the assg_key, because of certain formats only
                # typing the attributes, and we want to avoid data
                # loss, through these formats (which should be avoided
                # as durable stores of them though)
                self._graph.add_node(assg_key,
                                     node_id=assg_key,
                                     node_idx=node_idx,
                                     assignments=assigs,
                                     num_samples=num_samples)
                self._node_idxs[assg_key] = node_idx

            # also initialize the reverse which is memoized if needed
            self._node_idx_to_id_dict = None

            # now count the transitions between the states and set those
            # as the edges between nodes

            # first get the sliding window transitions from the contig
            # tree, once we set edges for a tree we don't really want to
            # have multiple sets of transitions on the same network so we
            # don't provide the method to add different assignments
            self._transition_lag_time = None
            self._probmat = None
            self._countsmat = None
            if transition_lag_time is not None:


                # get the weights for the walkers so we can compute
                # the weighted transition counts
                weights = [[] for run_idx in contig_tree.wepy_h5.run_idxs]
                for idx_tup, traj_fields_d in contig_tree.wepy_h5.iter_trajs_fields(
                        ['weights'],
                        idxs=True):

                    run_idx, traj_idx = idx_tup

                    weights[run_idx].append(np.ravel(traj_fields_d['weights']))

                # set the lag time attribute
                self._transition_lag_time = transition_lag_time

                # get the transitions
                transitions = []
                for window in contig_tree.sliding_windows(self._transition_lag_time):

                    transition = [window[0], window[-1]]

                    # convert the window trace on the contig to a trace
                    # over the runs
                    transitions.append(transition)

                # then get the counts for those edges
                counts_d = transition_counts(
                    self._assignments,
                    transitions,
                    weights=weights)

                # create the edges and set the counts into them
                for edge, trans_counts in counts_d.items():
                    self._graph.add_edge(*edge, counts=trans_counts)

                # then we also want to get the transition probabilities so
                # we get the counts matrix and compute the probabilities
                # we first have to replace the keys of the counts of the
                # node_ids with the node_idxs
                node_id_to_idx_dict = self.node_id_to_idx_dict()
                self._countsmat = counts_d_to_matrix(
                                    {(node_id_to_idx_dict[edge[0]],
                                      node_id_to_idx_dict[edge[1]]) : counts
                                     for edge, counts in counts_d.items()})

                self._probmat = normalize_counts(self._countsmat)

                # then we add these attributes to the edges in the network
                node_idx_to_id_dict = self.node_id_to_idx_dict()
                for i_id, j_id in self._graph.edges:
                    # i and j are the node idxs so we need to get the
                    # actual node_ids of them
                    i_idx = node_idx_to_id_dict[i_id]
                    j_idx = node_idx_to_id_dict[j_id]

                    # convert to a normal float and set it as an explicitly named attribute
                    self._graph.edges[i_id, j_id]['transition_probability'] = \
                                                                float(self._probmat[i_idx, j_idx])

                    # we also set the general purpose default weight of
                    # the edge to be this.
                    self._graph.edges[i_id, j_id]['Weight'] = \
                                                    float(self._probmat[i_idx, j_idx])


        # then get rid of the assignments dictionary, this information
        # can be accessed from the network
        del self._node_assignments
        del self._assignments

    def _key_init(self, contig_tree):
        """Initialize the assignments structures given the field key to use.

        Parameters
        ----------

        """

        wepy_h5 = contig_tree.wepy_h5

        # blank assignments
        assignments = [[[] for traj_idx in range(wepy_h5.num_run_trajs(run_idx))]
                             for run_idx in wepy_h5.run_idxs]

        test_field = wepy_h5.get_traj_field(
            wepy_h5.run_idxs[0],
            wepy_h5.run_traj_idxs(0)[0],
            self.assg_field_key,
        )

        # WARN: assg_field shapes can come wrapped with an extra
        # dimension. We handle both cases. Test the first traj and see
        # how it is
        unwrap = False

        if len(test_field.shape) == 2 and test_field.shape[1] == 1:
            # then we raise flag to unwrap them
            unwrap = True

        elif len(test_field.shape) == 1:
            # then it is unwrapped and we don't need to do anything,
            # just assert the flag to not unwrap
            unwrap = False

        else:

            raise ValueError(f"Wrong shape for an assignment type observable: {test_field.shape}")


        # the raw assignments
        curr_run_idx = -1
        for idx_tup, fields_d in wepy_h5.iter_trajs_fields(
                                         [self.assg_field_key], idxs=True):

            run_idx = idx_tup[0]
            traj_idx = idx_tup[1]


            assg_field = fields_d[self.assg_field_key]

            # if we need to we unwrap the assignements scalar values
            # if they need it
            if unwrap:
                assg_field = np.ravel(assg_field)

            assignments[run_idx][traj_idx].extend(assg_field)

        # then just call the assignments constructor to do it the same
        # way
        self._assignments_init(assignments)

    def _assignments_init(self, assignments):
        """Given the assignments structure sets up the other necessary
        structures.

        Parameters
        ----------
        assignments : list of list of array_like of dim (n_traj_frames, observable_shape[0], ...),
                      conditionally optional on 'assg_field_key'

            List of assignments for all frames in each run, where each
            element of the outer list is for a run, the elements of
            these lists are lists for each trajectory which are
            arraylikes of shape (n_traj, observable_shape[0], ...).

        """

        # set the type for the assignment field
        self._assg_field_type = type(assignments[0])

        # set the raw assignments to the temporary attribute
        self._assignments = assignments

        # this is the dictionary mapping node_id -> the (run_idx, traj_idx, cycle_idx) frames
        self._node_assignments = defaultdict(list)

        for run_idx, run in enumerate(assignments):
            for traj_idx, traj in enumerate(run):
                for frame_idx, assignment in enumerate(traj):
                    self._node_assignments[assignment].append( (run_idx, traj_idx, frame_idx) )

    def node_id_to_idx(self, assg_key):
        """Convert a node_id (which is the assignment value) to a canonical index.

        Parameters
        ----------
        assg_key : node_id

        Returns
        -------
        node_idx : int

        """
        return self.node_id_to_idx_dict()[assg_key]

    def node_idx_to_id(self, node_idx):
        """Convert a node index to its node id.

        Parameters
        ----------
        node_idx : int

        Returns
        -------
        node_id : node_id


        """
        return self.node_idx_to_id_dict()[node_idx]

    def node_id_to_idx_dict(self):
        """Generate a full mapping of node_ids to node_idxs."""
        return self._node_idxs

    def node_idx_to_id_dict(self):
        """Generate a full mapping of node_idxs to node_ids."""


        if self._node_idx_to_id_dict is None:
            rev = {node_idx : node_id for node_id, node_idx in self._node_idxs.items()}
            self._node_idx_to_id_dict = rev
        else:
            rev = self._node_idx_to_id_dict

        # just reverse the dictionary and return
        return rev


    @property
    def graph(self):
        """The networkx.DiGraph of the macrostate network."""
        return self._graph

    @property
    def num_states(self):
        """The number of states in the network."""
        return len(self.graph)

    @property
    def node_ids(self):
        """A list of the node_ids."""
        return list(self.graph.nodes)

    @property
    def contig_tree(self):
        """The underlying ContigTree"""
        return self._base_contig_tree

    @property
    def assg_field_key(self):
        """The string key of the field used to make macro states from the WepyHDF5 dataset.

        Raises
        ------
        MacroStateNetworkError
            If this wasn't used to construct the MacroStateNetwork.

        """
        if self._assg_field_key is None:
            raise MacroStateNetworkError("Assignments were manually defined, no key.")
        else:
            return self._assg_field_key

    @property
    def countsmat(self):
        """Return the transition counts matrix of the network.

        Raises
        ------
        MacroStateNetworkError
            If no lag time was given.

        """

        if self._countsmat is None:
            raise MacroStateNetworkError("transition counts matrix not calculated")
        else:
            return self._countsmat

    @property
    def probmat(self):
        """Return the transition probability matrix of the network.

        Raises
        ------
        MacroStateNetworkError
            If no lag time was given.

        """

        if self._probmat is None:
            raise MacroStateNetworkError("transition probability matrix not set")
        else:
            return self._probmat

    def get_node_attributes(self, node_id):
        """Returns the node attributes of the macrostate.

        Parameters
        ----------
        node_id : node_id

        Returns
        -------
        macrostate_attrs : dict

        """
        return self.graph.nodes[node_id]

    def get_node_attribute(self, node_id, attribute_key):
        """Return the value for a specific node and attribute.

        Parameters
        ----------
        node_id : node_id

        attribute_key : str

        Returns
        -------
        node_attribute

        """
        return self.get_node_attributes(node_id)[attribute_key]

    def get_nodes_attribute(self, attribute_key):
        """Get a dictionary mapping nodes to a specific attribute. """

        nodes_attr = {}
        for node_id in self.graph.nodes:
            nodes_attr[node_id] = self.graph.nodes[node_id][attribute_key]

        return nodes_attr

    def node_assignments(self, node_id):
        """Return the microstates assigned to this macrostate as a run trace.

        Parameters
        ----------
        node_id : node_id

        Returns
        -------
        node_assignments : list of tuples of ints (run_idx, traj_idx, cycle_idx)
            Run trace of the nodes assigned to this macrostate.

        """
        return self.get_node_attribute(node_id, self.ASSIGNMENTS)


    def set_nodes_attribute(self, key, values_dict):
        """Set node attributes for the key and values for each node.

        Parameters
        ----------
        key : str

        values_dict : dict of node_id: values

        """
        for node_id, value in values_dict.items():
            self.graph.nodes[node_id][key] = value

    @property
    def node_groups(self):
        return self._node_groups

    def set_node_group(self, group_name, node_ids):

        # push these values to the nodes themselves, overwriting if
        # necessary
        self._set_group_nodes_attribute(group_name, node_ids)

        # then update the group mapping with this
        self._node_groups[group_name] = node_ids


    def _set_group_nodes_attribute(self, group_name, group_node_ids):

        # the key for the attribute of the group goes in a little
        # namespace prefixed with _group
        group_key = '_groups/{}'.format(group_name)

        # make the mapping
        values_map = {node_id : True if node_id in group_node_ids else False
                      for node_id in self.graph.nodes}

        # then set them
        self.set_nodes_attribute(group_key, values_map)


    @property
    def observables(self):
        """The list of available observables."""
        return self._observables

    def node_observables(self, node_id):
        """Dictionary of observables for each node_id."""

        node_obs = {}
        for obs_name in self.observables:
            obs_key = '_observables/{}'.format(obs_name)
            node_obs[obs_name] = self.get_nodes_attributes(node_id, obs_key)

        return node_obs

    def set_nodes_observable(self, observable_name, node_values):

        # the key for the attribute of the observable goes in a little
        # namespace prefixed with _observable
        observable_key = '_observables/{}'.format(observable_name)

        self.set_nodes_attribute(observable_key, node_values)

        # then add to the list of available observables
        self._observables.append(observable_name)

    @property
    def layouts(self):
        return self._layouts

    def node_layouts(self, node_id):
        """Dictionary of layouts for each node_id."""

        node_layouts = {}
        for layout_name in self.layouts:
            layout_key = '_layouts/{}'.format(layout_name)
            node_layouts[obs_name] = self.get_nodes_attributes(node_id, layout_key)

        return node_layouts

    def set_nodes_layout(self, layout_name, node_values):

        # the key for the attribute of the observable goes in a little
        # namespace prefixed with _observable
        layout_key = '_layouts/{}'.format(layout_name)

        self.set_nodes_attribute(layout_key, node_values)

        # then add to the list of available observables
        if layout_name not in self._layouts:
            self._layouts.append(layout_name)


    def write_gexf(self, filepath, exclude_fields=None, layout=None):
        """Writes a graph file in the gexf format of the network.

        Parameters
        ----------
        filepath : str

        """

        layout_key = None
        if layout is not None:
            layout_key = '_layouts/{}'.format(layout)
            if layout not in self.layouts:
                raise ValueError("Layout not found, use None for no layout")


        if exclude_fields is None:
            exclude_fields = [self.ASSIGNMENTS]
        else:
            exclude_fields.append(self.ASSIGNMENTS)
            exclude_fields = list(set(exclude_fields))

        # exclude the layouts, we will set the viz manually for the layout
        exclude_fields.extend(['_layouts/{}'.format(layout_name)
                               for layout_name in self.layouts])


        # to do this we need to get rid of the assignments in the
        # nodes though since this is not really supported or good to
        # store in a gexf file which is more for visualization as an
        # XML format, so we copy and modify then write the copy
        gexf_graph = deepcopy(self._graph)
        for node in gexf_graph:
            # remove requested fields
            for field in exclude_fields:
                del gexf_graph.nodes[node][field]

            # also remove the fields which are not valid gexf types
            fields = list(gexf_graph.nodes[node].keys())
            for field in fields:
                if type(gexf_graph.nodes[node][field]) not in nx.readwrite.gexf.GEXF.xml_type:
                    del gexf_graph.nodes[node][field]

            if layout_key is not None:

                # set the layout as viz attributes to this
                gexf_graph.nodes[node]['viz'] = self._graph.nodes[node][layout_key]

        nx.write_gexf(gexf_graph, filepath)


    def nodes_to_records(self):

        keys = ['num_samples', 'Weight']

        # add all the groups to the keys
        keys.extend(['_groups/{}'.format(key) for key in self.node_groups.keys()])

        # add the observables
        keys.extend(['_observables/{}'.format(obs) for obs in self.observables])

        recs = []
        for node_id in self.graph.nodes:
            rec = {'node_id' : node_id}
            for key in keys:
                rec[key] = self.get_node_attribute(node_id, key)

            recs.append(rec)

        return recs

    def nodes_to_dataframe(self):
        """Make a dataframe of the nodes and their attributes.

        Not all attributes will be added as they are not relevant to a
        table style representation anyhow.

        The columns will be:

        - node_id
        - node_idx
        - num samples
        - Weight (if available)
        - groups (as booleans) which is anything in the '_groups' namespace
        - observables : anything in the '_observables' namespace and
          will assume to be scalars

        """

        # TODO: set the column order
        # col_order = []

        return pd.DataFrame(self.nodes_to_records())


    def edges_to_records(self):
        """Make a dataframe of the nodes and their attributes.

        Not all attributes will be added as they are not relevant to a
        table style representation anyhow.

        The columns will be:

        - edge_id
        - source
        - target
        - type (directed or undirected)
        - transition counts (if available)
        - transition probability (if available)

        """


        keys = ['counts', 'transition_probability']

        recs = []
        for edge_id in self.graph.edges:
            rec = {'edge_id' : edge_id}
            for key in keys:
                rec[key] = self.graph.edges[edge_id][key]

            recs.append(rec)

        return recs



    def edges_to_dataframe(self):
        """Make a dataframe of the nodes and their attributes.

        Not all attributes will be added as they are not relevant to a
        table style representation anyhow.

        The columns will be:

        - transition counts (if available)
        - transition probability (if available)

        """

        return pd.DataFrame(self.edges_to_records())

    def node_map(self, func, map_func=map):
        """Map a function over the nodes.

        The function should take as its first argument a node_id and
        the second argument a dictionary of the node attributes. This
        will not give access to the underlying trajectory data in the
        HDF5, to do this use the 'node_fields_map' function.

        Extra args not supported use 'functools.partial' to make
        functions with arguments for all data.


        Parameters
        ----------
        func : callable
            The function to map over the nodes.

        map_func : callable
            The mapping function, implementing the `map` interface

        Returns
        -------

        node_values : dict of node_id : values
            The mapping of node_ids to the values computed by the mapped func.

        """

        # wrap the function so that we can pass through the node_id
        def func_wrapper(args):
            node_id, node_attrs = args
            return node_id, func(node_attrs)

        # zip the node_ids with the node attributes as an iterator
        node_attr_it = ((node_id,
                         {**self.get_node_attributes(node_id), 'node_id' : node_id})
                        for node_id in self.graph.nodes
                       )

        return {node_id : value for node_id, value
                in map_func(func_wrapper, node_attr_it)}

class MacroStateNetwork():
    """Provides an abstraction over weighted ensemble data in the form of
    a kinetically connected network.

    The MacroStateNetwork refers to any grouping of the so called
    "micro" states that were observed during simulation,
    i.e. trajectory frames, and not necessarily in the usual sense
    used in statistical mechanics. Although it is the perfect vehicle
    for working with such macrostates.

    Because walker trajectories in weighted ensemble there is a
    natural way to generate the edges between the macrostate nodes in
    the network. These edges are determined automatically and a lag
    time can also be specified, which is useful in the creation of
    Markov State Models.
    """


    def __init__(self, contig_tree,
                 base_network=None,
                 assg_field_key=None,
                 assignments=None,
                 transition_lag_time=2):

        self.closed = True
        self._contig_tree = contig_tree
        self._wepy_h5 = self._contig_tree.wepy_h5

        # if we pass a base network use that one instead of building
        # one manually
        if base_network is not None:
            assert isinstance(base_network, BaseMacroStateNetwork)

            self._set_base_network_to_self(base_network)

        else:
            new_network = BaseMacroStateNetwork(contig_tree,
                                                assg_field_key=assg_field_key,
                                                assignments=assignments,
                                                transition_lag_time=transition_lag_time)

            self._set_base_network_to_self(new_network)



    def _set_base_network_to_self(self, base_network):

        self._base_network = base_network

        # then make references to this for the attributes we need

        # attributes
        self._graph = self._base_network._graph
        self._assg_field_key = self._base_network._assg_field_key
        self._node_idxs = self._base_network._node_idxs
        self._node_idx_to_id_dict = self._base_network._node_idx_to_id_dict
        self._transition_lag_time = self._base_network._transition_lag_time
        self._probmat = self._base_network._probmat
        self._countsmat = self._base_network._countsmat

        # functions
        self.node_id_to_idx = self._base_network.node_id_to_idx
        self.node_idx_to_id = self._base_network.node_idx_to_id
        self.node_id_to_idx_dict = self._base_network.node_id_to_idx_dict
        self.node_idx_to_id_dict = self._base_network.node_idx_to_id_dict

        self.get_node_attributes = self._base_network.get_node_attributes
        self.get_node_attribute = self._base_network.get_node_attribute
        self.get_nodes_attribute = self._base_network.get_nodes_attribute
        self.node_assignments = self._base_network.node_assignments
        self.set_nodes_attribute = self._base_network.set_nodes_attribute

        self.node_groups = self._base_network.node_groups
        self.set_node_group = self._base_network.set_node_group
        self._set_group_nodes_attribute = self._base_network._set_group_nodes_attribute
        self.observables = self._base_network.observables
        self.node_observables = self._base_network.node_observables
        self.set_nodes_observable = self._base_network.set_nodes_observable

        self.nodes_to_records =  self._base_network.nodes_to_records
        self.nodes_to_dataframe = self._base_network.nodes_to_dataframe
        self.edges_to_records = self._base_network.edges_to_records
        self.edges_to_dataframe = self._base_network.edges_to_dataframe
        self.node_map = self._base_network.node_map

        self.write_gexf = self._base_network.write_gexf

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


    # from the Base class

    @property
    def graph(self):
        """The networkx.DiGraph of the macrostate network."""
        return self._graph

    @property
    def num_states(self):
        """The number of states in the network."""
        return len(self.graph)

    @property
    def node_ids(self):
        """A list of the node_ids."""
        return list(self.graph.nodes)

    @property
    def assg_field_key(self):
        """The string key of the field used to make macro states from the WepyHDF5 dataset.

        Raises
        ------
        MacroStateNetworkError
            If this wasn't used to construct the MacroStateNetwork.

        """
        if self._assg_field_key is None:
            raise MacroStateNetworkError("Assignments were manually defined, no key.")
        else:
            return self._assg_field_key

    @property
    def countsmat(self):
        """Return the transition counts matrix of the network.

        Raises
        ------
        MacroStateNetworkError
            If no lag time was given.

        """

        if self._countsmat is None:
            raise MacroStateNetworkError("transition counts matrix not calculated")
        else:
            return self._countsmat

    @property
    def probmat(self):
        """Return the transition probability matrix of the network.

        Raises
        ------
        MacroStateNetworkError
            If no lag time was given.

        """

        if self._probmat is None:
            raise MacroStateNetworkError("transition probability matrix not set")
        else:
            return self._probmat






    # unique to the HDF5 holding one
    @property
    def base_network(self):
        return self._base_network

    @property
    def wepy_h5(self):
        """The WepyHDF5 source object for which the contig tree is being constructed. """
        return self._wepy_h5

    def state_to_mdtraj(self, node_id, alt_rep=None):
        """Generate an mdtraj.Trajectory object from a macrostate.

        By default uses the "main_rep" in the WepyHDF5
        object. Alternative representations of the topology can be
        specified.

        Parameters
        ----------
        node_id : node_id

        alt_rep : str
             (Default value = None)

        Returns
        -------
        traj : mdtraj.Trajectory

        """
        with self.wepy_h5:
            return self.wepy_h5.trace_to_mdtraj(self.base_network.node_assignments(node_id),
                                                alt_rep=alt_rep)

    def state_to_traj_fields(self, node_id, alt_rep=None):

        return self.states_to_traj_fields([node_id], alt_rep=alt_rep)

    def states_to_traj_fields(self, node_ids, alt_rep=None):


        node_assignments = []
        for node_id in node_ids:
            node_assignments.extend(self.base_network.node_assignments(node_id))

        with self.wepy_h5:

            # get the right fields
            rep_path = self.wepy_h5._choose_rep_path(alt_rep)
            fields = [rep_path, 'box_vectors']

            return self.wepy_h5.get_trace_fields(node_assignments,
                                                 fields)


    def get_node_fields(self, node_id, fields):
        """Return the trajectory fields for all the microstates in the
        specified macrostate.

        Parameters
        ----------
        node_id : node_id

        fields : list of str
            Field name to retrieve.

        Returns
        -------

        fields : dict of str: array_like
           A dictionary mapping the names of the fields to an array of the field.
           Like fields of a trace.

        """
        node_trace = self.base_network.node_assignments(node_id)

        # use the node_trace to get the weights from the HDF5
        with self.wepy_h5:
            fields_d = self.wepy_h5.get_trace_fields(node_trace, fields)

        return fields_d

    def iter_nodes_fields(self, fields):
        """Iterate over all nodes and return the field values for all the
        microstates for each.

        Parameters
        ----------
        fields : list of str

        Returns
        -------
        nodes_fields : dict of node_id: (dict of field: array_like)
            A dictionary with an entry for each node.
            Each node has it's own dictionary of node fields for each microstate.

        """

        nodes_d = {}
        for node_id in self.graph.nodes:
            fields_d = self.base_network.get_node_fields(node_id, fields)
            nodes_d[node_id] = fields_d

        return nodes_d

    def microstate_weights(self):
        """Returns the weights of each microstate on the basis of macrostates.

        Returns
        -------
        microstate_weights : dict of node_id: ndarray

        """

        node_weights = {}
        for node_id in self.graph.nodes:
            # get the trace of the frames in the node
            node_trace = self.base_network.node_assignments(node_id)

            # use the node_trace to get the weights from the HDF5
            with self.wepy_h5:
                trace_weights = self.wepy_h5.get_trace_fields(node_trace, ['weights'])['weights']

            node_weights[node_id] = trace_weights

        return node_weights

    def macrostate_weights(self):
        """Compute the total weight of each macrostate.

        Returns
        -------
        macrostate_weights : dict of node_id: float

        """

        macrostate_weights = {}
        microstate_weights = self.microstate_weights()
        for node_id, weights in microstate_weights.items():
            macrostate_weights[node_id] = float(sum(weights)[0])

        return macrostate_weights

    def set_macrostate_weights(self):
        """Compute the macrostate weights and set them as node attributes
        'Weight'."""
        self.base_network.set_nodes_attribute('Weight', self.macrostate_weights())

    def node_fields_map(self, func, fields, map_func=map):
        """Map a function over the nodes and microstate fields.

        The function should take as its arguments:

        1. node_id
        2. dictionary of all the node attributes
        3. fields dictionary mapping traj field names. (The output of
        `MacroStateNetwork.get_node_fields`)

        This *will* give access to the underlying trajectory data in
        the HDF5 which can be requested with the `fields`
        argument. The behaviour is very similar to the
        `WepyHDF5.compute_observable` function with the added input
        data to the mapped function being all of the macrostate node
        attributes.

        Extra args not supported use 'functools.partial' to make
        functions with arguments for all data.


        Parameters
        ----------
        func : callable
            The function to map over the nodes.

        fields : iterable of str
           The microstate (trajectory) fields to provide to the mapped function.

        map_func : callable
            The mapping function, implementing the `map` interface

        Returns
        -------

        node_values : dict of node_id : values
            The mapping of node_ids to the values computed by the mapped func.


        Returns
        -------

        node_values : dict of node_id : values
            Dictionary mapping nodes to the computed values from the
            mapped function.

        """

        # wrap the function so that we can pass through the node_id
        def func_wrapper(args):
            node_id, node_attrs, node_fields = args
            return node_id, func(node_attrs, node_fields)

        # zip the node_ids with the node attributes as an iterator
        node_attr_fields_it = (
            (node_id,
             {**self.get_node_attributes(node_id), 'node_id' : node_id},
             self.get_node_fields(node_id, fields),
            )
                     for node_id in self.graph.nodes)

        return {node_id : value for node_id, value
                in map_func(func_wrapper, node_attr_fields_it)}
