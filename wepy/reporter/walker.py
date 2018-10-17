import numpy as np

from wepy.reporter.reporter import ProgressiveFileReporter

from wepy.util import json_to_mdtraj_topology

import mdtraj as mdj

class WalkerReporter(ProgressiveFileReporter):

    def __init__(self, json_topology=None,
                 **kwargs):

        super().__init__(**kwargs)

        assert json_topology is not None, "must give a JSON format topology"

        self.json_top = json_topology

        # make an mdtraj top so we can make trajs easily
        self.mdtraj_top = json_to_mdtraj_topology(self.json_top)

    def report(self, new_walkers=None,
               **kwargs):

        kwargs = self._select_report_kwargs(**kwargs)

        # make a trajectory from these walkers
        traj = mdj.Trajectory(np.array([state['positions'] for state in new_walkers]),
                              topology=self.mdtraj_top)

        # write to the file for this trajectory
        traj.save_dcd(self.file_path)
