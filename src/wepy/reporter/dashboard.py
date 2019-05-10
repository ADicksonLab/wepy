"""WIP: Reporter that produces a text file that gives high level
information on the progress of a simulation.

"""
import logging

from wepy.reporter.reporter import ProgressiveFileReporter

class DashboardReporter(ProgressiveFileReporter):
    """ """

    FILE_ORDER = ("dashboard_path",)
    SUGGESTED_EXTENSIONS = ("dash.org",)

    def dashboard_string(self):
        """ """
        raise NotImplementedError

    def write_dashboard(self):
        """ """

        with open(self.file_path, mode=self.mode) as dashboard_file:
            dashboard_file.write(self.dashboard_string())
