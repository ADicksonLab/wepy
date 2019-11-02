import os.path as osp
from collections import defaultdict
import itertools as it
import logging
from warnings import warn

from wepy.reporter.dashboard import ResamplerDashboardSection

import numpy as np
import pandas as pd
from tabulate import tabulate

class WExploreDashboardSection(ResamplerDashboardSection):
    RESAMPLER_SECTION_TEMPLATE = \
"""

Resampling Algorithm: {{ name }}

Parameters:
- Max Number of Regions: {{ max_n_regions }}
- Max Region Sizes: {{ max_region_sizes }}

Number of Regions per level:

{{ regions_per_level }}

** Walker Assignments

{{ walker_table }}

** Region Hierarchy

Defined Regions with the number of child regions per parent region:

{{ region_hierarchy }}

** Leaf Region Table

{{ leaf_region_table }}

** WExplore Log

{{ wexplore_log }}

"""

    def __init__(self, resampler=None,
                 max_n_regions=None,
                 max_region_sizes=None,
                 **kwargs
    ):

        if 'name' not in kwargs:
            kwargs['name'] = 'WExploreResampler'

        super().__init__(resampler=resampler,
                         max_n_regions=max_n_regions,
                         max_region_sizes=max_region_sizes,
                         **kwargs
        )

        if resampler is not None:
            self.max_n_regions = resampler.max_n_regions
            self.max_region_sizes = resampler.max_region_sizes
        else:
            assert max_n_regions is not None, \
                "If a resampler is not given must give parameters: max_n_regions"
            assert max_region_sizes is not None, \
                "If a resampler is not given must give parameters: max_n_regions"

            self.max_n_regions = max_n_regions
            self.max_region_sizes = max_region_sizes


        self.n_levels = len(self.max_n_regions)

        # updatables
        self.root_region = ()
        init_leaf_region = tuple([0 for i in range(self.n_levels)])
        self.region_ids = [init_leaf_region]
        self.regions_per_level = []
        self.children_per_region = {}

        # resampling
        self.walker_assignments = []
        self.walker_image_distances = []
        self.curr_region_probabilities = defaultdict(int)
        self.curr_region_counts = defaultdict(int)

        #wexplore
        self.branch_records = []


    def _leaf_regions_to_all_regions(self, region_ids):

        # make a set of all the regions starting with the root region
        regions = set([self.root_region])
        for region_id in region_ids:
            for i in range(len(region_id)):
                regions.add(region_id[0:i+1])

        regions = list(regions)
        regions.sort()

        return regions


    def update_values(self, **kwargs):

        # the region assignments for walkers
        assignments = []
        walker_weights = [walker.weight for walker in kwargs['new_walkers']]
        # re-initialize the current weights dictionary
        self.curr_region_probabilities = defaultdict(int)
        self.curr_region_counts = defaultdict(int)
        for walker_record in kwargs['resampling_data']:

            assignment = tuple(walker_record['region_assignment'])
            walker_idx = walker_record['walker_idx'][0]
            assignments.append((walker_idx, assignment))

            # calculate the probabilities and counts of the regions
            # given the current distribution of walkers
            self.curr_region_probabilities[assignment] += walker_weights[walker_idx]
            self.curr_region_counts[assignment] += 1

        # sort them to get the walker indices in the right order
        assignments.sort()
        # then just get the assignment since it is sorted
        self.walker_assignments = [assignment for walker, assignment in assignments]

        # add to the records for region creation in WExplore
        for resampler_record in kwargs['resampler_data']:

            # get the values
            new_leaf_id = tuple(resampler_record['new_leaf_id'])
            branching_level = resampler_record['branching_level'][0]
            walker_image_distance = resampler_record['distance'][0]

            # add the new leaf id to the list of regions in the order they were created
            self.region_ids.append(new_leaf_id)

            # make a new record for a branching event which is:
            # (region_id, level branching occurred, distance of walker that triggered the branching)
            branch_record = (new_leaf_id,
                             branching_level,
                             walker_image_distance)

            # save it in the records
            self.branch_records.append(branch_record)

        # count the number of child regions each region has
        self.children_per_region = {}
        all_regions = self._leaf_regions_to_all_regions(self.region_ids)
        for region_id in all_regions:
            # if its a leaf region it has no children
            if len(region_id) == self.n_levels:
                self.children_per_region[region_id] = 0

            # all others we cound how many children it has
            else:
                # get all regions that have this one as a root
                children_idxs = set()
                for poss_child_id in all_regions:

                    # get the root at the level of this region for the child
                    poss_child_root = poss_child_id[0:len(region_id)]
                    # if the root is the same we keep it without
                    # counting children below the next level, but we skip the same region
                    if (poss_child_root == region_id) and (poss_child_id != region_id):

                        child_idx = poss_child_id[len(region_id)]

                        children_idxs.add(child_idx)

                # count the children of this region
                self.children_per_region[region_id] = len(children_idxs)

        # count the number of regions at each level
        self.regions_per_level = [0 for i in range(self.n_levels)]
        for region_id, n_children in self.children_per_region.items():
            level = len(region_id)

            # skip the leaves
            if level == self.n_levels:
                continue

            self.regions_per_level[level] += n_children


    def gen_fields(self, **kwargs):

        fields = super().gen_fields(**kwargs)

        regions = self._leaf_regions_to_all_regions(self.region_ids)
        region_children = [self.children_per_region[region] for region in regions]
        region_children_pairs = it.chain(*zip(regions, region_children))
        region_hierarchy = '\n'.join(
            ['{}     {}' for i in range(len(regions))]
        ).format(*region_children_pairs)

        # make a table for the regions
        region_table_colnames = ('region', 'n_walkers', 'curr_weight')
        region_table_d = {}
        region_table_d['region'] = self.region_ids
        region_table_d['n_walkers'] = [self.curr_region_counts[region]
                                       for region in self.region_ids]
        region_table_d['curr_weight'] = [self.curr_region_probabilities[region]
                                         for region in self.region_ids]
        leaf_region_table_df = pd.DataFrame(region_table_d,
                                            columns=region_table_colnames)
        leaf_region_table_df.set_index('region', drop=True)

        leaf_region_table_str = tabulate(leaf_region_table_df,
                                         headers=leaf_region_table_df.columns,
                                         tablefmt='orgtbl')

        # log of branching events
        branching_table_colnames = ('new_leaf_id', 'branching_level', 'trigger_distance')
        branching_table_df = pd.DataFrame(self.branch_records, columns=branching_table_colnames)
        branching_table_str = tabulate(branching_table_df,
                                       headers=branching_table_df.columns,
                                       tablefmt='orgtbl')

        ## walker weights
        walker_weights = [walker.weight for walker in kwargs['new_walkers']]
        # make the table of walkers using pandas, using the order here
        walker_table_colnames = ('weight', 'assignment')
        walker_table_d = {}
        walker_table_d['weight'] = walker_weights
        walker_table_d['assignment'] = self.walker_assignments

        walker_table_df = pd.DataFrame(walker_table_d, columns=walker_table_colnames)
        walker_table_str = tabulate(walker_table_df,
                                    headers=walker_table_df,
                                    tablefmt='orgtbl')


        new_fields = {
            'max_n_regions' : self.max_n_regions,
            'max_region_sizes' : self.max_region_sizes,
            'regions_per_level' : self.regions_per_level,
            'region_hierarchy' : region_hierarchy,
            'leaf_region_table' : leaf_region_table_str,
            'wexplore_log' : branching_table_str,
            'walker_table' : walker_table_str
        }

        fields.update(new_fields)

        return fields

