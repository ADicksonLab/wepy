import logging

from wepy.reporter.reporter import ProgressiveFileReporter

class DashboardReporter(ProgressiveFileReporter):

    SUGGESTED_EXTENSION = "dash.org"

    def dashboard_string(self):
        raise NotImplementedError

    def write_dashboard(self):

        with open(self.file_path, mode=self.mode) as dashboard_file:
            dashboard_file.write(self.dashboard_string())
