"""WIP: Reporter that produces a text file that gives high level
information on the progress of a simulation.

"""
from collections import defaultdict
import itertools as it
import logging
from datetime import datetime
import time

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

    PERFORMANCE_SECTION =\
"""

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

        self.n_cycles += 1

        self.walker_prob_summaries.append(self.calc_walker_summary(**kwargs))

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

        return ""

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


class BaseDashboardSection():
    pass

class ResamplerDashboardSection(BaseDashboardSection):
    pass

class RunnerDashboardSection(BaseDashboardSection):
    pass

class BCDashboardSection(BaseDashboardSection):
    pass

