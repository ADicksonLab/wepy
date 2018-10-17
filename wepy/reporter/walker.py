import numpy as np

from wepy.reporter.reporter import ProgressiveFileReporter
from wepy.util import json_to_mdtraj_topology, traj_box_vectors_to_lengths_angles

import mdtraj as mdj

class WalkerReporter(ProgressiveFileReporter):

    def __init__(self, json_topology=None,
                 main_rep_idxs=None,
                 init_state=None, init_state_path=None,
                 **kwargs):

        super().__init__(**kwargs)

        assert json_topology is not None, "must give a JSON format topology"
        assert main_rep_idxs is not None, "must give the indices of the atoms the topology represents"
        assert init_state is not None, "must give an init state for the topology PDB"
        assert init_state_path is not None, "must give a path to save an initial state PDB"

        self.json_top = json_topology
        self.main_rep_idxs = main_rep_idxs
        self.init_state = init_state
        self.init_state_path = init_state_path

        # make an mdtraj top so we can make trajs easily
        self.mdtraj_top = json_to_mdtraj_topology(self.json_top)

        # get the main rep idxs only
        main_rep_positions = self.init_state['positions'][self.main_rep_idxs]

        # convert the box vectors
        unitcell_lengths, unitcell_angles = traj_box_vectors_to_lengths_angles(
                                                   self.init_state['box_vectors'])

        # make a traj for the initial state to use as a topology for
        # visualizing the walkers
        traj = mdj.Trajectory(main_rep_positions,
                              unitcell_lengths=unitcell_lengths,
                              unitcell_angles=unitcell_angles,
                              topology=self.mdtraj_top)

        # save it
        traj.save_pdb(self.init_state_path)

    def report(self, new_walkers=None,
               **kwargs):

        # slice off the main_rep indices because that is all we want
        # to write for these
        main_rep_positions = np.array([walker.state['positions'][self.main_rep_idxs]
                                       for walker in new_walkers])

        # convert the box vectors
        unitcell_lengths, unitcell_angles = traj_box_vectors_to_lengths_angles(self.box_vectors)

        # make a trajectory from these walkers
        traj = mdj.Trajectory(main_rep_positions,
                              unitcell_lengths=unitcell_lengths,
                              unitcell_angles=unitcell_angles,
                              topology=self.mdtraj_top)


        # write to the file for this trajectory
        traj.save_dcd(self.file_path)
