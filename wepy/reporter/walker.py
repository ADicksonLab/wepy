import numpy as np

from wepy.reporter.reporter import ProgressiveFileReporter
from wepy.util.util import box_vectors_to_lengths_angles
from wepy.util.mdtraj import json_to_mdtraj_topology

import mdtraj as mdj

class WalkerReporter(ProgressiveFileReporter):


    # the order the files are in
    FILE_ORDER = ('init_state_path', 'walker_path')

    # the extnesions that will be used in orchestration
    SUGGESTED_EXTENSIONS = ('init.pdb', 'walkers.dcd')

    def __init__(self, *,
                 init_state=None,
                 json_topology=None,
                 main_rep_idxs=None,
                 **kwargs):

        super().__init__(**kwargs)

        assert json_topology is not None, "must give a JSON format topology"
        assert main_rep_idxs is not None, "must give the indices of the atoms the topology represents"
        assert init_state is not None, "must give an init state for the topology PDB"

        self.json_top = json_topology
        self.main_rep_idxs = main_rep_idxs
        self.init_state = init_state

        # make an mdtraj top so we can make trajs easily
        all_atoms_mdtraj_top = json_to_mdtraj_topology(self.json_top)

        # get a subset of this to use for the walker top
        self.mdtraj_top = all_atoms_mdtraj_top.subset(self.main_rep_idxs)

        # get the main rep idxs only
        main_rep_positions = self.init_state['positions'][self.main_rep_idxs]

        # convert the box vectors
        unitcell_lengths, unitcell_angles = box_vectors_to_lengths_angles(
                                                   self.init_state['box_vectors'])

        # make a traj for the initial state to use as a topology for
        # visualizing the walkers
        self.init_traj = mdj.Trajectory([main_rep_positions],
                                        unitcell_lengths=[unitcell_lengths],
                                        unitcell_angles=[unitcell_angles],
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
        traj.save_dcd(self.walker_path)
