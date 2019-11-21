"""WIP: Reporter that produces a text file that gives high level
information on the progress of a simulation.

"""
from collections import defaultdict
import itertools as it
import logging
from datetime import datetime
import time
from copy import copy

import numpy as np
import pandas as pd
from tabulate import tabulate
from jinja2 import Template

import logging

from wepy.reporter.reporter import ProgressiveFileReporter

class DashboardReporter(ProgressiveFileReporter):
    """A text based report of the status of a wepy simulation.

    This serves as a container for different dashboard components to
    go inside.

    """

    FILE_ORDER = ("dashboard_path",)
    SUGGESTED_EXTENSIONS = ("dash.org",)

    # TODO: add in a section for showing the number of walkers in each
    # cycle. This isn't relevant for our constant walker number
    # simulations though so I have elided it following YAGNI

    SIMULATION_SECTION_TEMPLATE = \
"""
Init Datetime: {{ init_date_time }}
Last write Datetime: {{ curr_date_time }}

Total Run time: {{ total_run_time }} s

Last Cycle Index: {{ last_cycle_idx }}
Number of Cycles: {{ n_cycles }}

** Walkers Summary

{{ walker_cycle_summary_table }}

"""

    PERFORMANCE_SECTION_TEMPLATE =\
"""
Average Cycle Time: {{ avg_cycle_time }}

{% if avg_runner_time %}Average Runner Time: {{ avg_runner_time }}{% else %}{% endif %}
{% if avg_bc_time %}Average Boundary Conditions Time: {{ avg_bc_time }}{% else %}{% endif %}
{% if avg_resampling_time %}Average Resampling Time: {{ avg_resampling_time }}{% else %}{% endif %}


Worker Avg. Segment Times:

{{ worker_avg_segment_time }}

** Cycle Performance Log

{{ cycle_log }}

** Worker Performance Log

{{ performance_log }}


"""

    DASHBOARD_TEMPLATE = \
"""* Simulation

{{ simulation }}

{% if resampler %}* Resampler{% else %}{% endif %}
{% if resampler %}{{ resampler }}{% else %}{% endif %}

{% if boundary_condition %}* Boundary Condition{% else %}{% endif %}
{% if boundary_condition %}{{ boundary_condition }}{% else %}{% endif %}

{% if runner %}* Runner{% else %}{% endif %}
{% if runner %}{{ runner }}{% else %}{% endif %}

* Performance

{{ performance }}

"""


    def __init__(self,
                 resampler_dash=None,
                 runner_dash=None,
                 bc_dash=None,
                 **kwargs):
        """

        Parameters
        ----------

        resampler_dash

        runner_dash

        bc_dash
        """

        super().__init__(**kwargs)

        self.resampler_dash = resampler_dash
        self.runner_dash = runner_dash
        self.bc_dash = bc_dash

        # recalculated values

        # general simulation

        # total number of cycles run
        self.n_cycles = 0

        # start date and time
        self.init_date_time = None
        self.init_sys_time = None

        # the total run time
        self.total_run_time = None

        # walker probabilities statistics
        self.walker_prob_summaries = []


        # performance
        self.cycle_compute_times = []
        self.cycle_runner_times = []
        self.cycle_bc_times = []
        self.cycle_resampling_times = []
        self.worker_records = []


    def init(self, **kwargs):

        super().init(**kwargs)

        self.init_date_time = datetime.today()
        self.total_run_time = self.init_date_time

        self.init_sys_time = time.time()

    def calc_walker_summary(self, **kwargs):

        walker_weights = [walker.weight for walker in kwargs['new_walkers']]

        summary = {
            'total' : np.sum(walker_weights),
            'min' : np.min(walker_weights),
            'max' : np.max(walker_weights),
        }

        return summary

    def update_values(self, **kwargs):

        ### simulation

        self.n_cycles += 1
        self.walker_prob_summaries.append(self.calc_walker_summary(**kwargs))

        self.update_performance_values(**kwargs)

        # update all the sections values
        if self.resampler_dash is not None:
            self.resampler_dash.update_values(**kwargs)
        if self.runner_dash is not None:
            self.runner_dash.update_values(**kwargs)
        if self.bc_dash is not None:
            self.bc_dash.update_values(**kwargs)

    def update_performance_values(self, **kwargs):

        ## worker specific performance

        # only do this part if there were any workers
        if len(kwargs['worker_segment_times']) > 0:

            # log of segment times for workers
            for worker_idx, segment_times in kwargs['worker_segment_times'].items():
                for segment_time in segment_times:
                    record = (
                        kwargs['cycle_idx'],
                        kwargs['n_segment_steps'],
                        worker_idx,
                        segment_time,
                    )
                    self.worker_records.append(record)

            # make a table out of these and compute the averages for each
            # worker
            worker_df = pd.DataFrame(self.worker_records, columns=('cycle_idx', 'n_steps',
                                                                   'worker_idx', 'segment_time'))
            # the aggregated table for the workers
            self.worker_agg_table = worker_df.groupby('worker_idx')\
                                    [['segment_time']].aggregate(np.mean)
            self.worker_agg_table.rename(columns={'segment_time' : 'avg_segment_time (s)'},
                                         inplace=True)

        else:
            self.worker_records = []
            self.worker_agg_table = pd.DataFrame({'avg_segment_time (s)' : []})


        ## cycle times

        # log of the components times
        self.cycle_runner_times.append(kwargs['cycle_runner_time'])
        self.cycle_bc_times.append(kwargs['cycle_bc_time'])
        self.cycle_resampling_times.append(kwargs['cycle_resampling_time'])

        # TODO: produces nan if one of them is not given
        # add up the three components to get the overall cycle time
        cycle_time = (kwargs['cycle_runner_time'] +
                      kwargs['cycle_bc_time'] +
                      kwargs['cycle_resampling_time'])

        # log of cycle times
        self.cycle_compute_times.append(cycle_time)

        # average of cycle components times
        self.avg_runner_time = np.mean(self.cycle_runner_times)
        self.avg_bc_time = np.mean(self.cycle_bc_times)
        self.avg_resampling_time = np.mean(self.cycle_resampling_times)

        # average cycle time
        self.avg_cycle_time = np.mean(self.cycle_compute_times)



    def write_dashboard(self, report_str):
        """Write the dashboard to the file."""

        with open(self.file_path, mode=self.mode) as dashboard_file:
            dashboard_file.write(report_str)

    def gen_sim_section(self, **kwargs):
        """"""

        walker_df = pd.DataFrame(self.walker_prob_summaries)
        walker_summary_tbl_str = tabulate(walker_df,
                                          headers=walker_df.columns,
                                          tablefmt='orgtbl')

        # render the simulation section
        sim_section_d = {
            'init_date_time' : self.init_date_time,
            'curr_date_time' : datetime.today().isoformat(),
            'total_run_time' : time.time() - self.init_sys_time,
            'last_cycle_idx' : kwargs['cycle_idx'],
            'n_cycles' : self.n_cycles,
            'walker_cycle_summary_table' : walker_summary_tbl_str,
        }

        sim_section_str = Template(self.SIMULATION_SECTION_TEMPLATE).render(
            **sim_section_d
        )

        return sim_section_str

    def gen_performance_section(self, **kwargs):

        # log of cycle times
        cycle_table_colnames = ('cycle_time (s)',
                                'runner_time (s)',
                                'boundary_conditions_time (s)',
                                'resampling_time (s)')

        cycle_table_df = pd.DataFrame({'cycle_times (s)' : self.cycle_compute_times,
                                       'runner_time (s)' : self.cycle_runner_times,
                                       'boundary_conditions_time (s)' : self.cycle_bc_times,
                                       'resampling_time (s)' : self.cycle_resampling_times},
                                      columns=cycle_table_colnames)

        cycle_table_str = tabulate(cycle_table_df,
                                   headers=cycle_table_df.columns,
                                   tablefmt='orgtbl')

        # log of workers performance
        worker_table_colnames = ('cycle_idx', 'n_steps', 'worker_idx', 'segment_time (s)',)
        worker_table_df = pd.DataFrame(self.worker_records, columns=worker_table_colnames)
        worker_table_str = tabulate(worker_table_df,
                                    headers=worker_table_df.columns,
                                    tablefmt='orgtbl',
                                    showindex=False)


        # table for aggregeated worker stats
        worker_agg_table_str = tabulate(self.worker_agg_table,
                                        headers=self.worker_agg_table.columns,
                                        tablefmt='orgtbl')


        performance_section_d = {
            'avg_cycle_time' : self.avg_cycle_time,
            'worker_avg_segment_time' : worker_agg_table_str,
            'cycle_log' : cycle_table_str,
            'performance_log' : worker_table_str,

            # optionals
            'avg_runner_time' : self.avg_runner_time,
            'avg_bc_time' : self.avg_bc_time,
            'avg_resampling_time' : self.avg_resampling_time,
        }

        performance_section_str = Template(self.PERFORMANCE_SECTION_TEMPLATE).render(
            **performance_section_d
        )

        return performance_section_str

    def report(self, **kwargs):

        # update the values that update each call to report
        self.update_values(**kwargs)

        # the two sections that are always there
        sim_section_str = self.gen_sim_section(**kwargs)
        performance_section_str = self.gen_performance_section(**kwargs)

        # the other optional sections

        # resampler
        if self.resampler_dash is not None:
            resampler_section_str = self.resampler_dash.gen_resampler_section(**kwargs)
        else:
            resampler_section_str = None

        # runner
        if self.runner_dash is not None:
            runner_section_str = self.runner_dash.gen_runner_section(**kwargs)
        else:
            runner_section_str = None

        # boundary conditions
        if self.bc_dash is not None:
            bc_section_str = self.bc_dash.gen_bc_section(**kwargs)
        else:
            bc_section_str = None

        # render the whole template
        report_str = Template(self.DASHBOARD_TEMPLATE).render(
            simulation=sim_section_str,
            resampler=resampler_section_str,
            boundary_condition=bc_section_str,
            runner=runner_section_str,
            performance=performance_section_str
        )

        # write the thing
        self.write_dashboard(report_str)



class ResamplerDashboardSection():

    RESAMPLER_SECTION_TEMPLATE = \
"""
Resampler: {{ name }}
"""

    def __init__(self, resampler=None,
                 name=None,
                 **kwargs):

        if resampler is not None:
            self.resampler_name = type(resampler).__name__

        elif name is not None:
            self.resampler_name = name

        else:
            self.resampler_name = "Unknown"

    def update_values(self, **kwargs):
        pass

    def gen_fields(self, **kwargs):

        fields = {
            'name' : self.resampler_name
        }

        return fields


    def gen_resampler_section(self, **kwargs):

        section_kwargs = self.gen_fields(**kwargs)

        section_str = Template(self.RESAMPLER_SECTION_TEMPLATE).render(
            **section_kwargs
        )

        return section_str

class RunnerDashboardSection():

    RUNNER_SECTION_TEMPLATE = \
"""
Runner: {{ name }}
"""

    def __init__(self, runner=None,
                 name=None,
                 **kwargs):

        if runner is not None:
            self.runner_name = type(runner).__name__

        elif name is not None:
            self.runner_name = name

        else:
            self.runner_name = "Unknown"

    def update_values(self, **kwargs):
        pass

    def gen_fields(self, **kwargs):

        fields = {
            'name' : self.runner_name
        }

        return fields


    def gen_runner_section(self, **kwargs):

        section_kwargs = self.gen_fields(**kwargs)


        section_str = Template(self.RUNNER_SECTION_TEMPLATE).render(
            **section_kwargs
        )

        return section_str


class BCDashboardSection():

    BC_SECTION_TEMPLATE = \
"""

Boundary Condition: {{ name }}

Total Number of Dynamics segments: {{ total_n_walker_segments }}

Total Number of Crossings: {{ total_crossings }}

Cumulative Boundary Crossed Weight: {{ total_crossed_weight }}

** Warping Log

{{ warping_log }}

"""


    WARP_RECORD_COLNAMES = (
        'cycle_idx',
        'walker_idx',
        'weight',
        'target_idx',
        'discontinuous',
    )


    def __init__(self, bc=None,
                 discontinuities=None,
                 name=None,
                 **kwargs):

        if bc is not None:
            self.bc_name = type(bc).__name__

        elif name is not None:
            self.bc_name = name

        else:
            self.bc_name = "Unknown"


        if bc is not None:
            self.bc_discontinuities = copy(bc.DISCONTINUITY_TARGET_IDXS)

        else:
            assert discontinuities is not None, \
                "If the bc is not given must give parameter: discontinuities"
            self.bc_discontinuities = discontinuities

        self.warp_records = []
        self.total_n_walker_segments = 0
        self.total_crossings = 0
        self.total_crossed_weight = 0.

    def update_values(self, **kwargs):

        # keep track of exactly how many walker segments are run, this
        # is useful for rate calculations via Hill's relation.
        self.total_n_walker_segments += len(kwargs['new_walkers'])

        # just create the bare warp records, since we know no more
        # domain knowledge, feel free to override and add more data to
        # this table
        for warp_record in kwargs['warp_data']:

            # the cycle
            cycle_idx = kwargs['cycle_idx']

            # the individual values from the warp record
            weight = warp_record['weight'][0]
            walker_idx = warp_record['walker_idx'][0]
            target_idx = warp_record['target_idx'][0]

            # determine if it was discontinuous

            # all targets are discontinuous
            if self.bc_discontinuities is Ellipsis:
                discont = True
            # none of them are discontinuous
            elif self.bc_discontinuities is None:
                discont = False
            # then it is a list of the discontinuous targets
            else:
                discont = True if target_idx in self.bc_discontinuities else False

            record = (cycle_idx, walker_idx, weight, target_idx, discont)
            self.warp_records.append(record)


    def gen_fields(self, **kwargs):

        # make the table for the collected warping records
        warp_table_df = pd.DataFrame(self.warp_records, columns=self.WARP_RECORD_COLNAMES)
        warp_table_str = tabulate(warp_table_df,
                                  headers=warp_table_df.columns,
                                  tablefmt='orgtbl')

        fields = {
            'name' : self.bc_name,
            'total_n_walker_segments' : self.total_n_walker_segments,
            'total_crossings' : self.total_crossings,
            'total_crossed_weight' : self.total_crossed_weight,
            'warping_log' : warp_table_str,
        }

        return fields

    def gen_bc_section(self, **kwargs):

        section_kwargs = self.gen_fields(**kwargs)

        section_str = Template(self.BC_SECTION_TEMPLATE).render(
            **section_kwargs
        )

        return section_str


