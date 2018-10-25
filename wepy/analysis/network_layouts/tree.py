from collections import defaultdict
from copy import copy

import numpy as np


class ResamplingTreeLayout():

    def __init__(self,
                 node_radius=1.0,
                 spacing_factor=0.1,
                 step_size=20.0,
                 fanning_factor=1.5):

        self.step_size = step_size
        self.node_radius = node_radius
        self.spacing_factor = spacing_factor
        self.fanning_factor = fanning_factor

    @classmethod
    def _overlaps(cls, positions, node_idx, node_radius):

            # get the nodes that this one overlaps with
            overlaps = []
            for other_node_idx, other_node_position in enumerate(positions):

                # you can't overlap yourself
                if node_idx == other_node_idx:
                    overlaps.append(False)
                else:
                    # check if there is an overlap between nodes
                    if np.abs(positions[node_idx] - other_node_position) < node_radius*2:
                        overlaps.append(True)
                    else:
                        overlaps.append(False)

            return overlaps

    @classmethod
    def _simple_gen_distribution(cls, nodes_x, node_radii, fanning_factor):

        # we want to update the positions given so we copy that array
        nodes_x = copy(nodes_x)

        # then for each node we place it sequentially
        for node_idx, node_radius in enumerate(node_radii):

            # counters are used to keep track of which way to seed the
            # movement of the nodes from the baseline, that is right or
            # left
            counter_1 = 0
            counter_2 = 0
            while any(cls._overlaps(nodes_x, node_idx, node_radius)):
                counter_2 += 1
                counter_1 += 1
                if counter_1 % 2 == 0:
                    nodes_x[node_idx] = nodes_x[node_idx] + counter_2 * fanning_factor
                else:
                    nodes_x[node_idx] = nodes_x[node_idx] - counter_2 * fanning_factor

        return nodes_x


    @classmethod
    def _simple_next_gen(cls, parents_x, children_parent_idxs,
                         fanning_factor, node_radius):

        # to place the children such that the edges never overlap we first
        # layout the nodes within their parent group, then we treat the
        # parent groups as single larger nodes and lay them out together.

        # so make a dictionary of the "families" (i.e. parent : [children]
        # mappings)
        families = defaultdict(list)
        for child_idx, parent_idx in enumerate(children_parent_idxs):
            families[parent_idx].append(child_idx)

        # for each family perform a layout algorithm that gets rid of
        # overlaps for the children nodes
        family_distributions = {}
        # also figure out the overall family node radius
        family_node_radii = {}
        # and map a 'family_idx' (over the existing families in the next
        # generation) to the parent_idx that identifies a family, the
        # index of the parent_idxs are the family idxs
        parent_family_idxs = []
        for parent_idx, children_idxs in families.items():

            # if there are no children skip this
            if len(children_idxs) == 0:
                pass

            # otherwise layout the children
            else:

                # add this as a surviving family
                parent_family_idxs.append(parent_idx)

                # generate the starting x positions for the children,
                # which is 0.0 because it will get re-embedded based on
                # the families ultimate position
                children_init_positions = [0.0 for i in range(len(children_idxs))]

                # also the node radii of the children
                children_node_radii = [node_radius for i in range(len(children_idxs))]

                # then update the positions by removing the overlaps
                children_x = cls._simple_gen_distribution(children_init_positions,
                                                      children_node_radii,
                                                      fanning_factor)

                family_distributions[parent_idx] = children_x

                # get the radii of this family node, which is the end to end
                # distance of the nodes
                family_node_radius = (max(children_x) + node_radius) - (min(children_x) - node_radius)

                family_node_radii[parent_idx] = family_node_radius


        n_families = len(parent_family_idxs)

        # generate an array of node positions and radii in order for the
        # family nodes
        family_node_positions = [0.0 for i in range(n_families)]
        family_node_radii_arr = [0.0 for i in range(n_families)]
        for family_idx, parent_idx in enumerate(parent_family_idxs):

            # center them around the parents position
            family_node_positions[family_idx] = parents_x[parent_idx]

            # just copy over the family node radius
            node_radius = family_node_radii[parent_idx]
            family_node_radii_arr[family_idx] = node_radius

        # now we treat each family as a "node" with a new radius which is
        # the end to end of all the child nodes
        family_positions = cls._simple_gen_distribution(family_node_positions,
                                                    family_node_radii_arr,
                                                    fanning_factor)

        # with the family node positions we can then embed the individual
        # children nodes with absolute positions of the family nodes, and
        # assign them to a valid child of that parent

        # so we make a new array for the children positions
        children_x = [None for i in range(len(children_parent_idxs))]
        for family_idx, parent_idx in enumerate(parent_family_idxs):

            # get the family position
            family_position = family_positions[family_idx]

            # get the starting children positions
            children_positions = [family_distribution
                                  for family_distribution in family_distributions[parent_idx]]

            # then add the overall family node position to them all
            children_positions = [child_pos + family_position
                                  for child_pos in children_positions]

            # now we need to get the correct order of the family groups by
            # assigning the positions to the correct child idxs. To get an
            # available child idx we look them up in the families
            # dictionary double checking we still have parity

            # so make a temporary list we can pop things out of
            family_children_idxs = copy(families[parent_idx])

            # go through all the positions we have to allocate
            for child_position in children_positions:
                # pop a child idx to use for it, if this fails there is an
                # error somewhere else
                child_idx = family_children_idxs.pop()

                # then set the new child position with this
                children_x[child_idx] = child_position

        return children_x

    @classmethod
    def _initial_parent_distribution(cls, n_nodes,
                                     spacing_factor=None, node_radius=None):

        assert spacing_factor is not None, "must provide spacing factor"
        assert node_radius is not None, "must provide node radius"

        return np.linspace(spacing_factor * ((-n_nodes) + 1),
                           spacing_factor * (n_nodes - 1),
                           n_nodes)


    def _layout_array(self, parent_table):

        n_timesteps = len(parent_table)
        n_walkers = len(parent_table[0])

        # initialize the positions to zeros, we add one to the
        # timesteps for the roots of the resampling trees
        node_positions = np.zeros((n_timesteps+1, n_walkers, 3))

        # initialize the first generation (cycle) node positions, this
        # is the root positions
        first_gen_positions = self._initial_parent_distribution(n_walkers,
                                                                spacing_factor=self.spacing_factor,
                                                                node_radius=self.node_radius)

        # save them as full coordinates for visualization
        node_positions[0] = np.array([[x, 0.0, 0.0] for x in first_gen_positions])

        # we use the last gen positions to make the next gen
        last_gen_positions = first_gen_positions

        # DEBUG
        # import ipdb; ipdb.set_trace()

        # propagate and minimize the rest of the step nodes
        for step_idx in range(n_timesteps):

            # the generation index is the one used for the actual tree
            # and it is always one more than the step index, since the
            # first generation is not counted as a step and was
            # already made outside the loop
            generation_idx = step_idx + 1

            # generate the starting positions for the next generation nodes
            curr_gen_positions = self._simple_next_gen(last_gen_positions,
                                                       parent_table[step_idx],
                                                       self.fanning_factor,
                                                       self.node_radius)



            node_positions[generation_idx] = np.array(
                [np.array([x, float(generation_idx * self.step_size), 0.0])
                 for x in curr_gen_positions])


            # set the last gen positions
            last_gen_positions = curr_gen_positions


        return node_positions


    def layout(self, parent_forest):
        """Given the input of a parent table, returns a dictionary mapping
        nodes to their xyz layout coordinates."""

        # get the parent table from the parent forest
        parent_table = parent_forest.parent_table

        layout_array = self._layout_array(parent_table)

        # make a dictionary mapping node ids to layout values
        node_coords = {}
        for cycle_idx, layout_row in enumerate(layout_array):
            for walker_idx, coord in enumerate(layout_row):
                node_coords[(cycle_idx, walker_idx)] = coord

        return node_coords
