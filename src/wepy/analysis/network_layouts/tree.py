"""Class for generating layouts for resampling trees.

Routines
--------

ResamplingTreeLayout.layout

"""

from warnings import warn
from collections import defaultdict
from copy import copy
import itertools as it

import numpy as np
import networkx as nx

from wepy.analysis.network_layouts.layout import LayoutError

class ResamplingTreeLayout():
    """Class that wraps the parameters for generating resampling tree layouts.

    Use the 'layout' method to generate inputs to a LayoutGraph for rendering.

    Attributes
    ----------

    node_radius : float
        Default node radius to use.
    row_spacing : float
        Spacing between nodes in a single row in layout.
    step_spacing : float
        Spacing between the rows of nodes in each step of the layout.
    central_axis : float
        X coordinate value to center each row around in the tree layout.
    """

    def __init__(self,
                 node_radius=1.0,
                 row_spacing=5.0,
                 step_spacing=20.0,
                 central_axis=0.0):
        """Constructing the object is just a setting of the parameters and
        collection of methods for generating layout positions.

        Arguments
        ---------
        node_radius : float, optional
            Default node radius to use.
             (Default value = 1.0)

        row_spacing : float
            Spacing between nodes in a single row in layout.
             (Default value = 5.0)

        step_spacing : float
            Spacing between the rows of nodes in each step of the layout.
             (Default value = 20.0)

        central_axis : float
            X coordinate value to center each row around in the tree layout.
             (Default value = 0.0)
        """


        self.node_radius = node_radius
        self.row_spacing = row_spacing
        self.step_spacing = step_spacing
        self.central_axis = central_axis

    def _overlaps(self, positions, node_radii, node_idx):
        """

        Parameters
        ----------
        positions :

        node_radii :

        node_idx :


        Returns
        -------

        """

        # get the nodes that this one overlaps with
        overlaps = []
        for other_node_idx, other_node_position in enumerate(positions):

            # you can't overlap yourself
            if node_idx == other_node_idx:
                overlaps.append(False)
            else:
                # check if there is an overlap between nodes
                diff = np.abs(positions[node_idx] - other_node_position)
                if diff < node_radii[node_idx] + node_radii[other_node_idx] + self.row_spacing:
                    overlaps.append(True)
                else:
                    overlaps.append(False)

        return overlaps

    def _node_row_length(self, node_positions, node_radii):
        """Get the edge to edge length of a row of nodes.

        Parameters
        ----------
        node_positions :

        node_radii :

        Returns
        -------

        row_length : float

        """

        max_child_idx = np.argmax(node_positions)
        max_edge = node_positions[max_child_idx] + node_radii[max_child_idx]
        min_child_idx = np.argmin(node_positions)
        min_edge = node_positions[min_child_idx] - node_radii[min_child_idx]

        return abs(max_edge - min_edge)

    def _simple_gen_distribution(self, nodes_x, node_radii):
        """

        Parameters
        ----------
        nodes_x :

        node_radii :


        Returns
        -------

        new_nodes_positions

        """

        # we want to update the positions given so we copy that array
        node_positions = np.array(copy(nodes_x))

        # first we check to see if there are any groups of nodes that
        # are in identical positions. If there are we can lump them
        # into a single node with a bigger radii and move things
        # around it

        groups = []
        group_positions = []
        for node_idx, node_position in enumerate(node_positions):

            # first get if this nodes position is already in the list
            # of known positions
            if node_position in group_positions:
                # if it is then get which group it is and add it to the list for that
                idx = group_positions.index(node_position)
                groups[idx].append(node_idx)

            else:
                # start a new group for it
                group_positions.append(node_position)
                groups.append([node_idx])

        # now we filter out the groups that only have one node
        chosen_idxs = [idx for idx, group in enumerate(groups)
                       if len(group) > 1]

        group_positions = [group_positions[idx] for idx in chosen_idxs]

        groups = [groups[idx] for idx in chosen_idxs]
        n_groups = len(groups)

        # we make a collection of the nodes that are in groups
        grouped_node_idxs = set(it.chain(*groups))

        # then we remove these nodes from the positions and radii
        # arrays and replace them with single nodes of updated radii
        # and positions.
        group_radii = []
        for group_idx, group in enumerate(groups):

            # the new radii will be the sum of the diameters plus the
            # spacing between the nodes

            # the sum of the diamters
            sum_diameters = sum([2 * node_radii[node_idx]
                                 for node_idx in group])

            # the spacing between is for 1 less than the total number
            # of the nodes in the group
            sum_spacing = self.row_spacing * (len(group) - 1)

            # the whole groups radius
            group_radius = (sum_diameters + sum_spacing) / 2

            group_radii.append(group_radius)

        # then we rip out the nodes that are part of groups and are
        # left with the singletons
        singleton_idxs = [node_idx for node_idx, _ in enumerate(node_positions)
                          if node_idx not in grouped_node_idxs]
        singleton_positions = [node_positions[node_idx] for node_idx in singleton_idxs]
        singleton_radii = [node_radii[node_idx] for node_idx in singleton_idxs]

        # then we make the effective nodes to position later, we put
        # them at the beginning of the new positions list so we can
        # easily decompose it later and put them back out to how they
        # should be
        eff_positions = np.array(group_positions + singleton_positions)
        eff_radii = np.array(group_radii + singleton_radii)

        # now that we have made the effective nodes we go ahead and
        # place them

        # then for each node we place it sequentially
        for node_idx, node_radius in enumerate(eff_radii):

            # get the overlaps for this node
            overlaps = self._overlaps(eff_positions, eff_radii, node_idx)

            # get how many nodes overlap it to the left and to the
            # right of it's starting position
            left_overlaps = [i for i, _ in enumerate(overlaps)
                             if eff_positions[i] < eff_positions[node_idx]]
            right_overlaps = [i for i, _ in enumerate(overlaps)
                              if eff_positions[i] > eff_positions[node_idx]]

            left_n_overlaps = len(left_overlaps)
            right_n_overlaps = len(right_overlaps)

            # we just move all the other nodes to allow this node to
            # be placed where we proposed it

            # left overlaps
            if left_n_overlaps > 0:

                # repel the other nodes around it

                # get the node idxs to the left of this one
                detangle_left_node_idxs = [i for i, position in enumerate(eff_positions)
                                            if position < eff_positions[node_idx]]

                # get the (center to center) distances from this node to
                # all it's overlaps
                cc_left_overlap_dists = [abs(eff_positions[node_idx] - eff_positions[i])
                                         for i in left_overlaps]

                # then get the edge to edge distances (which can be
                # negative where they cross over) this is the sum of the
                # radii subtracted from the center to center distance (ee = d-(r_0 + r_1))
                ee_left_overlap_dists = [cc_left_overlap_dists[i] - (node_radius + eff_radii[node_i])
                                         for i, node_i in enumerate(left_overlaps)]

                # we want the minimum one of the edge to edges (even if
                # its negative)
                left_min_ee_overlap_dist = min(ee_left_overlap_dists)

                # this overlap should always be negative for the
                # overlapping nodes so we minimally need to move one
                # node this far away to have no overlap, on top of
                # that we add the spacing for the move distance to make
                left_move = -1 * (self.row_spacing + (-left_min_ee_overlap_dist))

                # then apply to those nodes
                eff_positions[detangle_left_node_idxs] += left_move


            # right overlaps
            if right_n_overlaps > 0:

                # see above for comments
                detangle_right_node_idxs = [i for i, position in enumerate(eff_positions)
                                            if position > eff_positions[node_idx]]

                cc_right_overlap_dists = [abs(eff_positions[node_idx] - eff_positions[i])
                                          for i in right_overlaps]

                ee_right_overlap_dists = [cc_right_overlap_dists[i] - (node_radius + eff_radii[node_i])
                                         for i, node_i in enumerate(right_overlaps)]

                right_min_ee_overlap_dist = min(ee_right_overlap_dists)

                right_move = +1 * (self.row_spacing + (-right_min_ee_overlap_dist))

                eff_positions[detangle_right_node_idxs] += right_move


        # now that we have placed the effective nodes we want to
        # reconstruct the original nodes from the grouped ones

        # make a new positions array that we can repopulate
        new_node_positions = [None for _ in node_positions]

        # first for each group we need to place the nodes that had
        # identical positions within the space created by their larger
        # effective nodes
        for group_idx, group in enumerate(groups):

            # the position of the group overall, since the grouped
            # nodes were at the front of the list of the effective
            # nodes with preserverd order we can use the group_index
            # to get them
            group_position = eff_positions[group_idx]
            group_radii = eff_radii[group_idx]

            member_radii = [node_radii[node_idx] for node_idx in group]

            # just place them like you would the initial distribution,
            # which is centered at 0
            member_positions = np.array(self._initial_parent_distribution(member_radii))

            # translate all nodes left (negative by the midpoint/radius)
            member_positions -= group_radii

            # then we translate all of them to the overall group position
            member_positions += group_position

            # now that we have good positions we can place them back
            # into the main level positions list
            for i, node_idx in enumerate(group):
                new_node_positions[node_idx] = member_positions[i]

        # The singletons were at the end of the effective positions so
        # we have N_group_nodes + i where i is the index over the singletons
        for i, node_idx in enumerate(singleton_idxs):

            new_node_positions[node_idx] = eff_positions[n_groups + i]

        # sanity check that we covered them all
        assert all([True if pos is not None else False
                    for pos in new_node_positions]),\
                        "not all positions recovered from the effective nodes"


        return new_node_positions

    def _simple_next_gen(self, parents_x, children_parent_idxs, node_radii):
        """

        Parameters
        ----------
        parents_x :

        children_parent_idxs :

        node_radii :


        Returns
        -------

        children_x

        """

        children_x = []
        for parent_idx in children_parent_idxs:
            children_x.append(parents_x[parent_idx])

        children_x = self._simple_gen_distribution(children_x, node_radii)

        # check to make sure there are no overlaps
        for node_idx in range(len(children_x)):
            overlaps = self._overlaps(children_x, node_radii, node_idx)
            if any(overlaps):
                pass
                #raise LayoutError("node {} has an overlap".format(node_idx))

        return children_x


    def _initial_parent_distribution(self, node_radii):
        """

        Parameters
        ----------
        node_radii :

        Returns
        -------

        positions

        """

        # start at the origin and add nodes at positions that
        # accomodate their diameter and that are spaced by spacing_factor
        origin = 0.0

        positions = []

        # place the first node at the origin, offset by its radius so
        # that the leading edge is at 0.0, then iterate for the rest
        # of them
        positions.append(origin + node_radii[0])

        # compute the edge so we know how much to space the next one
        # by
        last_edge = origin + 2 * node_radii[0]

        # then do the rest
        for node_idx, node_radius in enumerate(node_radii[1:]):

            node_idx += 1

            # set the position from the last edge plus the space and
            # the new nodes radius
            position = last_edge + self.row_spacing + node_radius

            positions.append(position)

            # then calculate that ones edge
            last_edge = position + node_radius

        return positions

    def _center_row(self, positions, radii, center):
        """Centers the midpoint of a row of nodes around a number.

        Parameters
        ----------
        positions :

        radii :

        center : float


        Returns
        -------

        centered_positions

        """

        # calculate the length of the row
        row_length = self._node_row_length(positions, radii)

        min_child_idx = np.argmin(positions)
        min_edge = positions[min_child_idx] - radii[min_child_idx]

        # row midpoint is the minimum edge plus the radius of the row
        row_midpoint = min_edge + (row_length / 2)

        # translate all the points so that the midpoint is at 0
        centered_positions = positions - row_midpoint

        return centered_positions

    def _layout_array(self, parent_table, radii_array):
        """Generates the structured array of positions for nodes in the parent
        forest tree layout given the corresponding radii.

        Parameters
        ----------
        parent_table :

        radii_array :

        Returns
        -------

        node_positions

        """

        n_timesteps = len(parent_table)
        n_walkers = len(parent_table[0])

        # initialize the positions to zeros, we add one to the
        # timesteps for the roots of the resampling trees
        node_positions = np.zeros((n_timesteps+1, n_walkers, 3))

        # initialize the first generation (cycle) node positions, this
        # is the root positions, we give it the radii of the first row in the array
        first_gen_positions = np.array(self._initial_parent_distribution(radii_array[0]))

        # then center them around the central axis
        first_gen_positions = self._center_row(first_gen_positions, radii_array[0],
                                               self.central_axis)

        # save them as full coordinates for visualization
        node_positions[0] = np.array([[x, 0.0, 0.0] for x in first_gen_positions])

        # we use the last gen positions to make the next gen
        last_gen_positions = first_gen_positions

        # layout all the rest of the steps
        for step_idx in range(n_timesteps):

            # the generation index is the one used for the actual tree
            # and it is always one more than the step index, since the
            # first generation is not counted as a step and was
            # already made outside the loop
            generation_idx = step_idx + 1

            # generate the starting positions for the next generation nodes
            curr_gen_positions = self._simple_next_gen(last_gen_positions,
                                                       parent_table[step_idx],
                                                       radii_array[generation_idx])

            # center them around the central axis
            curr_gen_positions = self._center_row(curr_gen_positions,
                                                  radii_array[generation_idx],
                                                   self.central_axis)


            # figure out how big the increase in the Y direction
            # should be. This is based on the largest radii of the
            # last step and the largets radii of this step with the
            # mandatory spacing in between

            # get the y dimension of the last step
            last_y = node_positions[generation_idx-1, 0, 1]

            # get the largest radii of the last step
            last_max_radius = max(radii_array[generation_idx-1])

            # get the largest radii of this step
            this_max_radius = max(radii_array[generation_idx])

            # compute the step y dimension, which is from the last y
            # value plus the max radii plus the spacing
            step_y = last_y + last_max_radius + self.step_spacing + this_max_radius

            # then generate the coordinates
            node_positions[generation_idx] = np.array(
                [np.array([x, step_y, 0.0])
                 for x in curr_gen_positions])

            # set the last gen positions
            last_gen_positions = curr_gen_positions

        return node_positions


    def layout(self, parent_forest, node_radii=None):
        """Given a parent forest object, returns a dictionary
        mapping nodes to their xyz layout coordinates.

        If the node radii are given (as a dictionary mapping node ID
        to the desired radius) these are used as the the node radii
        and the default node radius parameter of the layout object is
        ignored.

        Radii are needed to calculate the proper positions so there is
        no overlaps between nodes.

        Parameters
        ----------
        parent_forest : ParentForest object

        node_radii : dict of node_id: float
            A dictionary mapping the nodes to node radii.
             (Default value = None)


        Returns
        -------
        node_coords : dict of node_id: array_like of float of dim (3,)
            x, y, z coordinates for all nodes. Z will be 0 for all.

        Raises
        ------
        ValueError
            If an invalid node_id is given.

        Warns
        -----
        If some but not all nodes were given assigned radii. Uses
        default value for unspecified nodes.

        """

        # get the parent table from the parent forest
        parent_table = parent_forest.parent_table

        # default your node radii to the layout parameter, and fill in
        # the rest with the node_radii given to this method

        # we need to generate an array of node radii, the size of
        # which is the parent table plus one extra row for the
        # root parents
        radii_array = []

        # count the number of nodes to check later whether or not the
        # user has given complete coverage with their radii. We can
        # issue a warning so that the user is not confused.
        n_nodes = 0

        # the "root" row, it is the same length as the first in the
        # parent table
        radii_array.append([self.node_radius for col in parent_table[0]])
        n_nodes += len(parent_table[0])

        # the rest of them
        for row in parent_table:
            radii_array.append([self.node_radius for col in row])
            n_nodes += len(row)

        # if any radii are given overwrite the default size, we count
        # how many new radii we give so that we can compare to the
        # actual number of nodes
        n_new_radii = 0
        if node_radii is not None:
            for node_id, radius in node_radii.items():

                # we know that the node_id is the (cycle_idx,
                # walker_idx), so we use these as indices on the array
                cycle_idx, walker_idx = node_id

                # the indices in the node_array start at the -1 root
                # nodes so we increase cycle index by one to match,
                # this is called the generation index
                generation_idx = cycle_idx + 1

                try:
                    radii_array[generation_idx][walker_idx] = radius
                except IndexError:
                    raise ValueError("invalid node_id was given")

                n_new_radii += 1

        # check if we have fewer assignments than there were nodes
        if n_new_radii < n_nodes:
            warn(
                "not all nodes were assigned custom radii,"
                " default value {} was used instead".format(self.node_radius))

        layout_array = self._layout_array(parent_table, radii_array)

        # make a dictionary mapping node ids to layout values
        node_coords = {}
        for cycle_idx, layout_row in enumerate(layout_array):

            for walker_idx, coord in enumerate(layout_row):

                # since the layout array starts at the root nodes (cycle
                # -1) we just reduce the cycle_idx by one when we set the
                # node attribute
                node_coords[(cycle_idx-1, walker_idx)] = coord

        return node_coords
