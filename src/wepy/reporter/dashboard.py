"""WIP: Reporter that produces a text file that gives high level
information on the progress of a simulation.

"""
from collections import defaultdict
import itertools as it
import logging

import numpy as np
import pandas as pd

import logging

from wepy.reporter.reporter import ProgressiveFileReporter

class DashboardReporter(ProgressiveFileReporter):
    """A text based report of the status of a wepy simulation."""

    FILE_ORDER = ("dashboard_path",)
    SUGGESTED_EXTENSIONS = ("dash.org",)
    WEIGHTED_ENSEMBLE_TEMPLATE = \
"""
* Weighted Ensemble Simulation
Integration Step Size: {step_time} seconds
{step_time_femtoseconds} femtoseconds
Last Cycle Index: {last_cycle_idx}
Number of Cycles: {n_cycles}
Single Walker Sampling Time: {walker_total_sampling_time} seconds
{walker_total_sampling_time_microseconds} microseconds
Total Sampling Time: {total_sampling_time} seconds
{total_sampling_time_microseconds} microseconds
"""

    BOUNDARY_CONDITIONS_TEMPLATE = \
"""
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
"""
    PERFORMANCE_TEMPLATE = \
"""
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

    DASHBOARD_TEMPLATE = \
"""
{weighted_ensemble_string}
{resampler_string}
{boundary_conditions_string}
{performance_string}
"""
    def __init__(self,
                 step_time=None, # seconds
                 bc_cutoff_distance=None,
                 **kwargs
                ):
        """Constructor for the WExplore dashboard reporter.

        Parameters
        ----------

        step_time : float
            The length of the time in each dynamics step.


        bc_cutoff_distance : float
            The distance for which a walker will be warped.

        """

        super().__init__(**kwargs)

        assert step_time is not None, "length of integration time step must be given"
        self.step_time = step_time

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


        # resampler
        # should be implemented based on the resampler method

        # resampling
        # should be implemented based on the resampler method



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


        # performance
        self.cycle_compute_times = []
        self.cycle_runner_times = []
        self.cycle_bc_times = []
        self.cycle_resampling_times = []
        self.worker_records = []

    REPORT_ITEM_KEYS = ('cycle_idx', 'n_segment_steps',
                        'new_walkers',
                        'warp_data', 'bc_data', 'progress_dat',
                        'resampling_data', 'resampler_data',
                        'worker_segment_times', 'cycle_runner_time',
                        'cycle_bc_time', 'cycle_resampling_time',)
    def report(self, cycle_idx=None,
               n_segment_steps=None,
               new_walkers=None,
               warp_data=None,
               progress_data=None,
               resampling_data=None,
               resampler_data=None,
               worker_segment_times=None,
               cycle_runner_time=None,
               cycle_bc_time=None,
               cycle_resampling_time=None,
               **kwargs):
        """

        Parameters
        ----------
        cycle_idx :
             (Default value = None)
        n_segment_steps :
             (Default value = None)
        new_walkers :
             (Default value = None)
        warp_data :
             (Default value = None)
        progress_data :
             (Default value = None)
        resampling_data :
             (Default value = None)
        resampler_data :
             (Default value = None)
        worker_segment_times :
             (Default value = None)
        cycle_runner_time :
             (Default value = None)
        cycle_bc_time :
             (Default value = None)
        cycle_resampling_time :
             (Default value = None)
        **kwargs :


        Returns
        -------

        """

        # first recalculate the total sampling time, _update the
        # number of cycles, and set the walker probabilities
        self._update_weighted_ensemble_values(cycle_idx,
                                             n_segment_steps,
                                             new_walkers)

        # if there were any warps we need to set new values for the
        # warp variables and add records
        self._update_warp_values(cycle_idx, warp_data)

        # _update progress towards the boundary conditions
        self._update_progress_values(cycle_idx, progress_data)

        # now we _update the WExplore values
        self._update_resampler_values(cycle_idx,
                                    resampling_data,
                                    resampler_data)

        # _update the performance of the workers for our simulation
        self._update_performance_values(cycle_idx,
                                       n_segment_steps, worker_segment_times,
                                       cycle_runner_time, cycle_bc_time,
                                       cycle_resampling_time)

        # write the dashboard
        self.write_dashboard()

    def _update_weighted_ensemble_values(self, cycle_idx, n_steps, walkers):
        """Update the values held in this object related to general WE
        simulation details.

        Parameters
        ----------
        cycle_idx : int
            The last cycle that was completed.

        n_steps : int
            The number of dynamics steps that were completed in the last cycle

        walkers : list of Walker objects
            The walkers generated from the last runner segment.

        """

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


    def _update_warp_values(self, cycle_idx, warp_data):
        """Update values associated with the boundary conditions.

        Parameters
        ----------
        cycle_idx : int
            The index of the last completed cycle.

        warp_data : list of dict of str : value
            List of dict-like records for each warping event from the
            last cycle.

        """

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


    def _update_progress_values(self, cycle_idx, progress_data):
        """Update values associated with the boundary conditions.

        Parameters
        ----------
        cycle_idx : int
            The index of the last completed cycle.

        progress_data : dict str : list
            A record indicating the progress values for each walker in
            the last cycle.

        """

        self.walker_distance_to_prot = tuple(progress_data['min_distances'])


    def _update_performance_values(self, cycle_idx, n_steps, worker_segment_times,
                                  cycle_runner_time, cycle_bc_time, cycle_resampling_time):
        """Update the value associated with performance metrics of the
        simulation.

        Parameters
        ----------
        cycle_idx : int

        n_steps : int

        worker_segment_times : dict of int : list of float
            Mapping worker index to the times they took for each
            segment they processed.

        cycle_runner_time : float
            Total time runner took in last cycle.

        cycle_bc_time : float
            Total time boundary conditions took in last cycle.

        cycle_resampling_time : float
            Total time resampler took in last cycle.

        """

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



    def _update_resampler_values(cycle_idx,
                                    resampling_data,
                                    resampler_data):
        pass

    def _weighted_ensemble_string(self):
        weighted_ensemble_string = self.WEIGHTED_ENSEMBLE_TEMPLATE.format(
            step_time=self.step_time,
            step_time_femtoseconds=self.step_time * 10e14,
            last_cycle_idx=self.last_cycle_idx,
            n_cycles=self.n_cycles,
            walker_total_sampling_time=self.walker_total_sampling_time,
            walker_total_sampling_time_microseconds=self.walker_total_sampling_time * 10e6,
            total_sampling_time=self.total_sampling_time,
            total_sampling_time_microseconds=self.total_sampling_time * 10e6,
            )

        return weighted_ensemble_string

    def _resampler_string(self):
        pass


    def _boundary_conditions_string(self):
        # log of warp events
        warp_table_colnames = ('walker_idx', 'weight', 'cycle_idx', 'time (s)')
        warp_table_df = pd.DataFrame(self.warp_records, columns=warp_table_colnames)
        warp_table_str = warp_table_df.to_string()
        warping_log = self.BOUNDARY_CONDITIONS_TEMPLATE.format(
            cutoff_distance=self.bc_cutoff_distance,
            n_exit_points=self.n_exit_points,
            cycle_n_exit_points=self.cycle_n_exit_points,
            total_unbound_weight=self.total_unbound_weight,
            expected_unbinding_time=self.expected_unbinding_time,
            reactive_traj_rate=self.reactive_traj_rate,
            exit_rate=self.exit_rate,
            warping_log=warp_table_str
        )

    def _performance_string(self):
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


        # table for aggregeated worker stats
        worker_agg_table_str = self.worker_agg_table.to_string()

        performance_string = self.PERFORMANCE_TEMPLATE.format(
            avg_runner_time=self.avg_runner_time,
            avg_bc_time=self.avg_bc_time,
            avg_resampling_time=self.avg_resampling_time,
            avg_cycle_time=self.avg_cycle_time,
            worker_avg_segment_time=worker_agg_table_str,
            cycle_log=cycle_table_str,
            performance_log=worker_table_str
            )

        return performance_string

    def dashboard_string(self):
        """Generate the dashboard string for the currrent state."""

        weighted_ensemble_log = self._weighted_ensemble_string()
        resampler_log = self._resampler_string()
        boundary_conditions_log = self._boundary_conditions_string()
        performance_log = self._performance_string()

        dashboard = self.DASHBOARD_TEMPLATE.format(
            weighted_ensemble_string=weighted_ensemble_log,
            resampler_string=resampler_log,
            boundary_conditions_string=boundary_conditions_log,
            performance_string=performance_log
        )

        return dashboard


    def write_dashboard(self):
        """Write the dashboard to the file."""

        with open(self.file_path, mode=self.mode) as dashboard_file:
            dashboard_file.write(self.dashboard_string())
