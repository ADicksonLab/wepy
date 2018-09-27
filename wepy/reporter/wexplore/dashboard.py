import os.path as osp
from collections import defaultdict
import itertools as it
import logging

from wepy.reporter.dashboard import DashboardReporter

import numpy as np
import pandas as pd

class WExploreDashboardReporter(DashboardReporter):

    SUGGESTED_EXTENSION = "wexplore.dash.org"

    DASHBOARD_TEMPLATE = \
"""* Weighted Ensemble Simulation

    Integration Step Size: {step_time} seconds
                           {step_time_femtoseconds} femtoseconds
    Last Cycle Index: {last_cycle_idx}
    Number of Cycles: {n_cycles}
    Single Walker Sampling Time: {walker_total_sampling_time} seconds
                                 {walker_total_sampling_time_microseconds} microseconds
    Total Sampling Time: {total_sampling_time} seconds
                         {total_sampling_time_microseconds} microseconds

* WExplore

    Max Number of Regions: {max_n_regions}
    Max Region Sizes: {max_region_sizes}
    Number of Regions per level:

        {regions_per_level}

** Region Hierarchy
Defined Regions with the number of child regions per parent region:
{region_hierarchy}

** WExplore Log

{wexplore_log}


* Walker Table
{walker_table}

* Leaf Region Table
{leaf_region_table}

* Warping through boundary conditions
    Cutoff Distance: {cutoff_distance}
    Number of Exit Points this Cycle: {cycle_n_exit_points}
    Total Number of Exit Points: {n_exit_points}
    Cumulative Unbound Weight {total_unbound_weight}
    Expected Reactive Traj. Time: {expected_unbinding_time} seconds
    Expected Reactive Traj. Rate: {reactive_traj_rate} 1/seconds
    Rate: {exit_rate} 1/seconds

** Warping Log

{warping_log}


* Performance
    Average Runner Time: {avg_runner_time}
    Average Boundary Conditions Time: {avg_bc_time}
    Average Resampling Time: {avg_resampling_time}
    Average Cycle Time: {avg_cycle_time}
    Worker Avg. Segment Times:

{worker_avg_segment_time}



** Cycle Performance Log

{cycle_log}

** Worker Performance Log

{performance_log}

"""

    def __init__(self,
                 step_time=None, # seconds
                 max_n_regions=None,
                 max_region_sizes=None,
                 bc_cutoff_distance=None,
                 **kwargs
                ):

        super().__init__(**kwargs)

        assert step_time is not None, "length of integration time step must be given"
        self.step_time = step_time

        assert max_n_regions is not None, "number of regions per level for WExplore must be given"
        self.max_n_regions = max_n_regions

        assert max_region_sizes is not None, "region sizes for WExplore must be given"
        self.max_region_sizes = max_region_sizes

        self.n_levels = len(self.max_n_regions)

        assert bc_cutoff_distance is not None, "cutoff distance for the boundary conditions must be given"
        self.bc_cutoff_distance = bc_cutoff_distance

        ## recalculated values

        # weighted ensemble
        self.walker_weights = []
        self.last_cycle_idx = 0
        self.n_cycles = 0
        self.walker_total_sampling_time = 0.0 # seconds
        self.total_sampling_time = 0.0 # seconds

        # warps
        self.n_exit_points = 0
        self.cycle_n_exit_points = 0
        self.total_unbound_weight = 0.0
        self.exit_rate = np.inf # 1 / seconds
        self.expected_unbinding_time = np.inf # seconds
        self.reactive_traj_rate = 0.0 # 1 / seconds

        # progress
        self.walker_distance_to_prot = [] # nanometers

        # WExplore

        # resampler
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

        # performance
        self.avg_cycle_time = np.nan
        self.avg_runner_time = np.nan
        self.avg_bc_time = np.nan
        self.avg_resampling_time = np.nan
        self.worker_agg_table = None


        ## Log of events variables

        # boundary conditions
        self.exit_point_weights = []
        self.exit_point_times = []
        self.warp_records = []

        # wexplore
        self.branch_records = []

        # performance
        self.cycle_compute_times = []
        self.cycle_runner_times = []
        self.cycle_bc_times = []
        self.cycle_resampling_times = []
        self.worker_records = []

    def report(self, cycle_idx, walkers,
               warp_data, bc_data, progress_data,
               resampling_data, resampler_data,
               n_steps=None,
               worker_segment_times=None,
               cycle_runner_time=None,
               cycle_bc_time=None,
               cycle_resampling_time=None,
               *args, **kwargs):

        # first recalculate the total sampling time, update the
        # number of cycles, and set the walker probabilities
        self.update_weighted_ensemble_values(cycle_idx, n_steps, walkers)

        # if there were any warps we need to set new values for the
        # warp variables and add records
        self.update_warp_values(cycle_idx, warp_data)

        # update progress towards the boundary conditions
        self.update_progress_values(cycle_idx, progress_data)

        # now we update the WExplore values
        self.update_wexplore_values(cycle_idx, resampling_data, resampler_data)

        # update the performance of the workers for our simulation
        self.update_performance_values(cycle_idx, n_steps, worker_segment_times,
                                       cycle_runner_time, cycle_bc_time, cycle_resampling_time)

        # write the dashboard
        self.write_dashboard()

    def update_weighted_ensemble_values(self, cycle_idx, n_steps, walkers):

        # the number of cycles
        self.last_cycle_idx = cycle_idx
        self.n_cycles += 1

        # amount of new sampling time for each walker
        new_walker_sampling_time = self.step_time * n_steps

        # accumulated sampling time for a single walker
        self.walker_total_sampling_time += new_walker_sampling_time

        # amount of sampling time for all walkers
        new_sampling_time = new_walker_sampling_time * len(walkers)

        # accumulated sampling time for the ensemble
        self.total_sampling_time += new_sampling_time

        # the weights of the walkers
        self.walker_weights = [walker.weight for walker in walkers]


    def update_warp_values(self, cycle_idx, warp_data):

        self.cycle_n_exit_points = 0
        for warp_record in warp_data:

            weight = warp_record['weight'][0]
            walker_idx = warp_record['walker_idx'][0]

            record = (walker_idx, weight, cycle_idx, self.walker_total_sampling_time)
            self.warp_records.append(record)

            # also add them to the individual records
            self.exit_point_weights.append(weight)
            self.exit_point_times.append(self.walker_total_sampling_time)

            # increase the number of exit points by 1
            self.n_exit_points += 1
            self.cycle_n_exit_points += 1

            # total accumulated unbound probability
            self.total_unbound_weight += weight

        # calculate the new rate using the Hill relation after taking
        # into account all of these warps
        self.exit_rate = self.total_unbound_weight / self.total_sampling_time

        # calculate the expected value of unbinding times
        self.expected_unbinding_time = np.sum([self.exit_point_weights[i] * self.exit_point_times[i]
                                               for i in range(self.n_exit_points)])

        # expected rate of reactive trajectories
        self.reactive_traj_rate = 1 / self.expected_unbinding_time


    def update_progress_values(self, cycle_idx, progress_data):

        self.walker_distance_to_prot = tuple(progress_data['min_distances'])

    def update_wexplore_values(self, cycle_idx, resampling_data, resampler_data):

        # the region assignments for walkers
        assignments = []
        # re-initialize the current weights dictionary
        self.curr_region_probabilities = defaultdict(int)
        self.curr_region_counts = defaultdict(int)
        for walker_record in resampling_data:

            assignment = tuple(walker_record['region_assignment'])
            walker_idx = walker_record['walker_idx'][0]
            assignments.append((walker_idx, assignment))

            # calculate the probabilities and counts of the regions
            # given the current distribution of walkers
            self.curr_region_probabilities[assignment] += self.walker_weights[walker_idx]
            self.curr_region_counts[assignment] += 1

        # sort them to get the walker indices in the right order
        assignments.sort()
        # then just get the assignment since it is sorted
        self.walker_assignments = [assignment for walker, assignment in assignments]

        # add to the records for region creation in WExplore
        for resampler_record in resampler_data:

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
        all_regions = self.leaf_regions_to_all_regions(self.region_ids)
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

                        try:
                            child_idx = poss_child_id[len(region_id)]
                        except IndexError:
                            import ipdb; ipdb.set_trace()

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


    def update_performance_values(self, cycle_idx, n_steps, worker_segment_times,
                                  cycle_runner_time, cycle_bc_time, cycle_resampling_time):


        ## worker specific performance

        # only do this part if there were any workers
        if len(worker_segment_times) > 0:

            # log of segment times for workers
            for worker_idx, segment_times in worker_segment_times.items():
                for segment_time in segment_times:
                    record = (cycle_idx, n_steps, worker_idx, segment_time)
                    self.worker_records.append(record)

            # make a table out of these and compute the averages for each
            # worker
            worker_df = pd.DataFrame(self.worker_records, columns=('cycle_idx', 'n_steps',
                                                                   'worker_idx', 'segment_time'))
            # the aggregated table for the workers
            self.worker_agg_table = worker_df.groupby('worker_idx')[['segment_time']].aggregate(np.mean)
            self.worker_agg_table.rename(columns={'segment_time' : 'avg_segment_time (s)'},
                                         inplace=True)

        else:
            self.worker_records = []
            self.worker_agg_table = pd.DataFrame({'avg_segment_time (s)' : []})


        ## cycle times

        # log of the components times
        self.cycle_runner_times.append(cycle_runner_time)
        self.cycle_bc_times.append(cycle_bc_time)
        self.cycle_resampling_times.append(cycle_resampling_time)

        # add up the three components to get the overall cycle time
        cycle_time = cycle_runner_time + cycle_bc_time + cycle_resampling_time

        # log of cycle times
        self.cycle_compute_times.append(cycle_time)

        # average of cycle components times
        self.avg_runner_time = np.mean(self.cycle_runner_times)
        self.avg_bc_time = np.mean(self.cycle_bc_times)
        self.avg_resampling_time = np.mean(self.cycle_resampling_times)

        # average cycle time
        self.avg_cycle_time = np.mean(self.cycle_compute_times)


    def leaf_regions_to_all_regions(self, region_ids):
        # make a set of all the regions starting with the root region
        regions = set([self.root_region])
        for region_id in region_ids:
            for i in range(len(region_id)):
                regions.add(region_id[0:i+1])

        regions = list(regions)
        regions.sort()

        return regions

    def dashboard_string(self):

        regions = self.leaf_regions_to_all_regions(self.region_ids)
        region_children = [self.children_per_region[region] for region in regions]
        region_children_pairs = it.chain(*zip(regions, region_children))
        region_hierarchy = '\n'.join(['{}     {}' for i in range(len(regions))]).format(*region_children_pairs)

        # make the table of walkers using pandas, using the order here
        # TODO add the image distances
        walker_table_colnames = ('weight', 'assignment', 'progress') #'image_distances'
        walker_table_d = {}
        walker_table_d['weight'] = self.walker_weights
        walker_table_d['assignment'] = self.walker_assignments
        walker_table_d['progress'] = self.walker_distance_to_prot
        walker_table_df = pd.DataFrame(walker_table_d, columns=walker_table_colnames)
        walker_table_str = walker_table_df.to_string()

        # make a table for the regions
        region_table_colnames = ('region', 'n_walkers', 'curr_weight')
        region_table_d = {}
        region_table_d['region'] = self.region_ids
        region_table_d['n_walkers'] = [self.curr_region_counts[region] for region in self.region_ids]
        region_table_d['curr_weight'] = [self.curr_region_probabilities[region] for region in self.region_ids]
        leaf_region_table_df = pd.DataFrame(region_table_d, columns=region_table_colnames)
        leaf_region_table_df.set_index('region', drop=True)
        leaf_region_table_str = leaf_region_table_df.to_string()

        # table for aggregeated worker stats
        worker_agg_table_str = self.worker_agg_table.to_string()

        # log of branching events
        branching_table_colnames = ('new_leaf_id', 'branching_level', 'trigger_distance')
        branching_table_df = pd.DataFrame(self.branch_records, columns=branching_table_colnames)
        branching_table_str = branching_table_df.to_string()

        # log of warp events
        warp_table_colnames = ('walker_idx', 'weight', 'cycle_idx', 'time (s)')
        warp_table_df = pd.DataFrame(self.warp_records, columns=warp_table_colnames)
        warp_table_str = warp_table_df.to_string()

        # log of cycle times
        cycle_table_colnames = ('cycle_time (s)', 'runner_time (s)', 'boundary_conditions_time (s)',
                                'resampling_time (s)')
        cycle_table_df = pd.DataFrame({'cycle_times' : self.cycle_compute_times,
                                       'runner_time' : self.cycle_runner_times,
                                       'boundary_conditions_time' : self.cycle_bc_times,
                                       'resampling_time' : self.cycle_resampling_times},
                                      columns=cycle_table_colnames)
        cycle_table_str = cycle_table_df.to_string()


        # log of workers performance
        worker_table_colnames = ('cycle_idx', 'n_steps', 'worker_idx', 'segment_time (s)',)
        worker_table_df = pd.DataFrame(self.worker_records, columns=worker_table_colnames)
        worker_table_str = worker_table_df.to_string()

        # format the dashboard string
        dashboard = self.DASHBOARD_TEMPLATE.format(
            step_time=self.step_time,
            step_time_femtoseconds=self.step_time * 10e15,
            last_cycle_idx=self.last_cycle_idx,
            n_cycles=self.n_cycles,
            walker_total_sampling_time=self.walker_total_sampling_time,
            walker_total_sampling_time_microseconds=self.walker_total_sampling_time * 10e6,
            total_sampling_time=self.total_sampling_time,
            total_sampling_time_microseconds=self.total_sampling_time * 10e6,
            cutoff_distance=self.bc_cutoff_distance,
            n_exit_points=self.n_exit_points,
            cycle_n_exit_points=self.cycle_n_exit_points,
            total_unbound_weight=self.total_unbound_weight,
            expected_unbinding_time=self.expected_unbinding_time,
            reactive_traj_rate=self.reactive_traj_rate,
            exit_rate=self.exit_rate,
            walker_distance_to_prot=self.walker_distance_to_prot,
            max_n_regions=self.max_n_regions,
            max_region_sizes=self.max_region_sizes,
            regions_per_level=self.regions_per_level,
            region_hierarchy=region_hierarchy,
            avg_runner_time=self.avg_runner_time,
            avg_bc_time=self.avg_bc_time,
            avg_resampling_time=self.avg_resampling_time,
            avg_cycle_time=self.avg_cycle_time,
            worker_avg_segment_time=worker_agg_table_str,
            walker_table=walker_table_str,
            leaf_region_table=leaf_region_table_str,
            warping_log=warp_table_str,
            wexplore_log=branching_table_str,
            cycle_log=cycle_table_str,
            performance_log=worker_table_str,
        )

        return dashboard

