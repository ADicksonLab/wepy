
class LayoutGraph():
    """Base class for networks that will be given layouts for visualization."""

    # graphviz colors are given by the RGB or RGBA hex string for them
    GRAPHVIZ_COLOR = 'color'
    GRAPHVIZ_POSITION = 'pos'
    GRAPHVIZ_SHAPE = 'shape'
    GRAPHVIZ_SIZE = 'size'

    GEXF_VIZ = 'viz'
    GEXF_VIZ_COLOR = 'color'
    GEXF_VIZ_COLOR_RED = 'r'
    GEXF_VIZ_COLOR_GREEN = 'g'
    GEXF_VIZ_COLOR_BLUE = 'b'
    GEXF_VIZ_COLOR_ALPHA = 'a'

    GEXF_VIZ_POSITION = 'position'
    GEXF_VIZ_POSITION_X = 'x'
    GEXF_VIZ_POSITION_Y = 'y'
    GEXF_VIZ_POSITION_Z = 'z'

    GEXF_VIZ_SIZE = 'size'
    GEXF_VIZ_SHAPE = 'shape'

    GEXF_EDGE_THICKNESS = 'thickness'
    GEXF_EDGE_SHAPE = 'shape'


    COLOR = GRAPHVIZ_COLOR
    POSITION = GRAPHVIZ_POSITION
    SHAPE = GRAPHVIZ_SHAPE
    SIZE = GRAPHVIZ_SIZE

    def __init__(self, graph):

        self._graph = graph

        # we copy the graph to another graph without the attributes so
        # we avoid name collisions between data and visualization metadata
        self._viz_graph = type(self.graph)()
        self._viz_graph.add_nodes_from(self.graph.nodes)
        self._viz_graph.add_edges_from(self.graph.edges)


        # add a "viz" attribute for all nodes for visualization
        # puproses in gexf format
        for node in self.viz_graph.nodes:
            self.viz_graph.node[node][self.GEXF_VIZ] = {}

    @property
    def graph(self):
        return self._graph

    @property
    def viz_graph(self):
        return self._viz_graph


    def get_node_attributes(self, attribute_key):
        """Gets attributes from the data graph and returns as a dictionary
        which is compatible with the set_node_viz_attributes methods.

        For instance could be used to get a 'weight' value and set as the color.
        """

        node_attr_dict = {}
        for node_id in self.graph.nodes:
            node_d = self.graph.node[node_id]
            try:
                value = node_d[attribute_key]

            # if the node doesn't have it set a None instead
            except KeyError:
                value = None

            node_attr_dict[node_id] = value

        return node_attr_dict


    # for setting visualization values of the network in a format agnostic way
    def set_node_viz_attributes(self, attribute_key, node_attribute_dict):

        for node_id, value in node_attribute_dict.items():
            self.viz_graph.nodes[node_id][attribute_key] = value

    def set_node_positions(self, node_positions_dict):

        self.set_node_viz_attributes(self.POSITION, node_positions_dict)

    def set_node_colors(self, node_colors_dict):

        self.set_node_viz_attributes(self.COLOR, node_colors_dict)

    def set_node_sizes(self, node_sizes_dict):

        self.set_node_viz_attributes(self.SIZE, node_sizes_dict)

    def set_node_shapes(self, node_shapes_dict):

        self.set_node_viz_attributes(self.SHAPE, node_shapes_dict)


    # getters
    def get_node_viz_attributes(self, attribute_key):

        node_attr_dict = {}
        for node_id in self.viz_graph.nodes:
            node_attr_dict[node_id] = self.graph[node_id][attribute_key]

        return node_attr_dict

    def get_node_positions(self):

        self.get_node_viz_attributes(self.POSITION)

    def get_node_colors(self):

        self.get_node_viz_attributes(self.COLOR)

    def get_node_sizes(self):

        self.get_node_viz_attributes(self.SIZE)

    def get_node_shapes(self):

        self.get_node_viz_attributes(self.SHAPE)



    @classmethod
    def RGBA_to_hex(cls, color_vec):

        assert not any([True if (color <= 255 and color >=0) else False for color in color_vec]),\
            "invalid color values, must be between 0 and 255"

        return '#' + ''.join(['{:02x}' for _ in color_vec]).format(*[color for color in color_vec])


    # methods for setting graphviz visualization attributes
    @classmethod
    def feature_vector_to_graphviz_position(cls, coord_vec):

        return ','.join(['{}' for _ in coord_vec]).format(*[i for i in coord_vec])


    @classmethod
    def feature_vector_to_graphviz_color_RGB(cls, color_vec):

        assert len(color_vec) == 3, "only 3 values for RGB allowed"

        return cls.RGBA_to_hex(color_vec)

    @classmethod
    def feature_vector_to_graphviz_color_RGBA(cls, color_vec):

        assert len(color_vec) == 4, "only 4 values for RGBA allowed"

        return cls.RGBA_to_hex(color_vec)


    # methods for setting graphviz attributes for visualization

    def set_node_graphviz(self, viz_key, node_dict):

        for node_id, value in node_dict.items():

            self.viz_graph.nodes[node_id][viz_key] = value

    def set_node_graphviz_positions(self, node_positions_dict):

        # convert the node positions values to a valid dictionary and
        # set the positions viz for all the nodes
        self.set_node_graphviz(self.GRAPHVIZ_POSITION,
                          {node_id : self.feature_vector_to_graphviz_position(coord)
                           for node_id, coord in node_positions_dict.items()})

        # we also pin the nodes here
        self.set_node_graphviz('pin',
                               {node_id : 'true' for node_id in node_positions_dict.keys()})

    def set_node_graphviz_colors_rgb(self, node_colors_dict):

        # convert the node colors values to a valid dictionary and
        # set the colors viz for all the nodes
        self.set_node_graphviz(self.GRAPHVIZ_COLOR,
                               {node_id : self.feature_vector_to_graphviz_color_RGB(coord)
                                        for node_id, coord in node_colors_dict.items()})

    def set_node_graphviz_colors_rgba(self, node_colors_dict):

        # convert the node colors values to a valid dictionary and
        # set the colors viz for all the nodes
        self.set_node_graphviz(self.GRAPHVIZ_COLOR,
                               {node_id : self.feature_vector_to_graphviz_color_RGBA_dict(coord)
                                        for node_id, coord in node_colors_dict.items()})

    def set_node_graphviz_sizes(self, node_sizes_dict):

        # convert the node colors values to a valid dictionary and
        # set the colors viz for all the nodes
        self.set_node_graphviz(self.GRAPHVIZ_SIZE, node_sizes_dict)

    def set_node_graphviz_shape(self, node_sizes_dict):

        # convert the node colors values to a valid dictionary and
        # set the colors viz for all the nodes
        self.set_node_graphviz(self.GRAPHVIZ_SHAPE, node_sizes_dict)


    # methods for setting gexf visualization attributes
    @classmethod
    def feature_vector_to_gexf_viz_position(cls, coord_vec):

        return {cls.GEXF_VIZ_POSITION_X : float(coord_vec[0]),
                cls.GEXF_VIZ_POSITION_Y : float(coord_vec[1]),
                cls.GEXF_VIZ_POSITION_Z : float(coord_vec[2])}

    @classmethod
    def feature_vector_to_gexf_viz_color_RGB(cls, color_vec):

        return {cls.GEXF_VIZ_COLOR_RED : int(color_vec[0]),
                cls.GEXF_VIZ_COLOR_GREEN : int(color_vec[1]),
                cls.GEXF_VIZ_COLOR_BLUE : int(color_vec[2])}

    @classmethod
    def feature_vector_to_gexf_viz_color_RGBA(cls, color_vec):

        return {cls.GEXF_VIZ_COLOR_RED : int(color_vec[0]),
                cls.GEXF_VIZ_COLOR_GREEN : int(color_vec[1]),
                cls.GEXF_VIZ_COLOR_BLUE : int(color_vec[2]),
                cls.GEXF_VIZ_COLOR_ALPHA : int(color_vec[3])}

    def set_node_gexf_viz(self, viz_key, node_dict):

        for node_id, value in node_dict.items():

            self.viz_graph.node[node_id][self.GEXF_VIZ][viz_key] = value

    def set_node_gexf_positions(self, node_positions_dict):

        # convert the node positions values to a valid dictionary and
        # set the positions viz for all the nodes
        self.set_node_gexf_viz(self.GEXF_VIZ_POSITION,
                          {node_id : self.feature_vector_to_gexf_viz_position(coord)
                           for node_id, coord in node_positions_dict.items()})

    def set_node_gexf_colors_rgb(self, node_colors_dict):

        # convert the node colors values to a valid dictionary and
        # set the colors viz for all the nodes
        self.set_node_gexf_viz(self.GEXF_VIZ_COLOR,
                               {node_id : self.feature_vector_to_gexf_viz_color_RGB(coord)
                                for node_id, coord in node_colors_dict.items()})

    def set_node_gexf_colors_rgba(self, node_colors_dict):

        # convert the node colors values to a valid dictionary and
        # set the colors viz for all the nodes
        self.set_node_gexf_viz(self.GEXF_VIZ_COLOR,
                               {node_id : self.feature_vector_to_gexf_viz_color_RGBA(coord)
                                for node_id, coord in node_colors_dict.items()})

    def set_node_gexf_alphas(self, node_alphas_dict):

        # convert the node colors values to a valid dictionary and
        # set the colors viz for all the nodes
        self.set_node_gexf_viz(self.GEXF_VIZ_COLOR,
                               {node_id : {self.GEXF_VIZ_COLOR_ALPHA : value}
                                for node_id, value in node_alphas_dict.items()})

    def set_node_gexf_sizes(self, node_sizes_dict):

        self.set_node_gexf_viz(self.GEXF_VIZ_SIZE, node_sizes_dict)

    def set_node_gexf_shape(self, node_shape_dict):

        self.set_node_gexf_viz(self.GEXF_VIZ_SHAPE, node_shape_dict)
