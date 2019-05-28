"""WIP: Reporter that produces a text file that gives high level
information on the progress of a simulation.

"""
import logging

from wepy.reporter.reporter import ProgressiveFileReporter

class DashboardReporter(ProgressiveFileReporter):
    """A text based report of the status of a wepy simulation."""

    FILE_ORDER = ("dashboard_path",)
    SUGGESTED_EXTENSIONS = ("dash.org",)

    def dashboard_string(self):
        """Generate the dashboard string for the currrent state."""
        raise NotImplementedError

    def write_dashboard(self):
        """Write the dashboard to the file."""

        with open(self.file_path, mode=self.mode) as dashboard_file:
            dashboard_file.write(self.dashboard_string())
