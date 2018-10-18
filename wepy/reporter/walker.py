import numpy as np

from wepy.reporter.reporter import ProgressiveFileReporter
from wepy.util import json_to_mdtraj_topology, traj_box_vectors_to_lengths_angles

import mdtraj as mdj

class WalkerReporter(ProgressiveFileReporter):


    SUGGESTED_EXTENSIONS = ('init.pdb', 'walkers.dcd')

    def __init__(self, *, walker_output_path=None,
                 init_state=None, init_state_path=None,
                 json_topology=None,
                 main_rep_idxs=None,
                 **kwargs):

        assert json_topology is not None, "must give a JSON format topology"
        assert main_rep_idxs is not None, "must give the indices of the atoms the topology represents"
        assert init_state is not None, "must give an init state for the topology PDB"
        assert init_state_path is not None, "must give a path to save an initial state PDB"
        assert walker_output_path is not None, "must give a path to save the walkers"

        self.json_top = json_topology
        self.main_rep_idxs = main_rep_idxs
        self.init_state = init_state

        # the path for the init_state
        self.init_state_path = init_state_path

        # the path for the actual walker trajectory
        self.walker_path = walker_output_path

        # then we call the superclass method to validate etc the paths
        super().__init__(file_paths=[self.walker_path, self.init_state_path], **kwargs)

        # make an mdtraj top so we can make trajs easily
        self.mdtraj_top = json_to_mdtraj_topology(self.json_top)

        # get the main rep idxs only
        main_rep_positions = self.init_state['positions'][self.main_rep_idxs]

        # convert the box vectors
        unitcell_lengths, unitcell_angles = traj_box_vectors_to_lengths_angles(
                                                   self.init_state['box_vectors'])

        # make a traj for the initial state to use as a topology for
        # visualizing the walkers
        self.init_traj = mdj.Trajectory(main_rep_positions,
                              unitcell_lengths=unitcell_lengths,
                              unitcell_angles=unitcell_angles,
                              topology=self.mdtraj_top)

        self.init_traj.save_pdb(self.init_state_path)

    def report(self, cycle_idx=None, new_walkers=None,
               **kwargs):

        # if this is the first cycle we need to save the initial state
        if cycle_idx == 0:
            self.init_traj

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
