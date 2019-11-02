import os.path as osp
from collections import defaultdict
import itertools as it
import logging

from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.reporter.dashboard import ResamplerDashboardSection

import numpy as np
import pandas as pd

class REVODashboardSection(ResamplerDashboardSection):

    RESAMPLER_TEMPLATE = \
"""

Resampling Algorithm: {{ name }}

Distance Exponent: {{ dist_exponent }}
Characteristic Distance: {{ char_dist }}
Merge Distance: {{ merge_dist }}

** Resamplig
Cycle index: {{ cycle_idx }}
The percentage of cloned walkers: {{ percentage_cloned_walkers }} %
The percentage of merged walkers: {{ percentage_merged_walkers }} %

** Statistics
Average All to All Distance: {{ avg_distance }}
Minimum All to All Distance: {{ min_distance }}
Maximum All to All Distance: {{ max_distance }}
Variation value = {{ variation }}
"""

    def __init__(self, resampler=None,
                 dist_exponent=None,
                 merge_dist=None,
                 lpmin=None,
                 char_dist=None,
                 seed=None,
                 decision=None,
                 **kwargs):

        if 'name' not in kwargs:
            kwargs['name'] = 'REVOResampler'

        super().__init__(resampler=resampler,
                 dist_exponent=dist_exponent,
                 merge_dist=merge_dist,
                 lpmin=lpmin,
                 char_dist=char_dist,
                 seed=seed,
                 decision=decision,
        )

        if resampler is not None:
            self.dist_exponent = resampler.dist_exponent
            self.merge_dist = resampler.merge_dist
            self.lpmin = resampler.lpmin
            self.char_dist = resampler.char_dist
            self.seed = resampler.seed
            self.decision = resampler.DECISION

        else:
            assert dist_exponent is not None, \
                "if no resampler given must give parameters: dist_exponent"
            assert merge_dist is not None, \
                "if no resampler given must give parameters: merge_dist"
            assert lpmin is not None, \
                "if no resampler given must give parameters: lpmin"
            assert char_dist is not None, \
                "if no resampler given must give parameters: char_dist"
            assert seed is not None, \
                "if no resampler given must give parameters: seed"
            assert decision is not None, \
                "if no resampler given must give parameters: decision"

            self.dist_exponent = dist_exponent
            self.merge_dist = merge_dist
            self.lpmin = lpmin
            self.char_dist = char_dist
            self.seed = seed
            self.decision = decision

        # updatables
        self.percentage_cloned_walkers = 0
        self.percentage_merged_walkers = 0
        #REVO
        self.avg_distance = None
        self.min_distance = None
        self.max_distance = None
        self.variation_values = None

    def update_values(self, **kwargs):

        num_clones = 0
        num_merges = 0
        num_walkers = len(kwargs['resampling_data'])
        for walker_record in kwargs['resampling_data']:
            if walker_record['decision_id'][0]==self.decision.ENUM.CLONE.value:
                num_clones += 1
            elif walker_record['decision_id'][0]==self.decision.ENUM.KEEP_MERGE.value:
                num_merges += 1

        self.percentage_cloned_walkers = (num_clones/num_walkers) * 100
        self.percentage_merged_walkers = (num_merges/num_walkers) * 100

        #Get the statistics
        for resampler_record in kwargs['resampler_data']:
            self.variation_value = resampler_record['variation'][0]
            distance_matrix = resampler_record['distance_matrix']


        #get the upper triangle values of the distance_matrix
        distance_matrix = np.triu(distance_matrix)
        distance_values= distance_matrix[np.where(distance_matrix>0)]
        self.avg_distance = np.average(distance_values)
        self.min_distance = np.min(distance_values)
        self.max_distance  = np.max(distance_values)

        self.cycle_idx = kwargs['cycle_idx']

    def gen_fields(self, **kwargs):

        fields = super().gen_fields(**kwargs)

        new_fields = {
            'dist_exponent' : self.dist_exponent,
            'char_dist' : self.char_dist,
            'merge_dist' : self.merge_dist,
            'cycle_idx' : self.cycle_idx,
            'percentage_cloned_walkers' : self.percentage_cloned_walkers,
            'percentage_merged_walkers' : self.percentage_merged_walkers,
            'avg_distance' : self.avg_distance,
            'min_distance' : self.min_distance,
            'max_distance' : self.max_distance,
            'variation' : self.variation_value
        }

        fields.update(new_fields)

        return fields
