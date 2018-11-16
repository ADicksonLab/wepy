import numpy as np

from wepy.reporter.reporter import ProgressiveFileReporter
from wepy.util.util import box_vectors_to_lengths_angles, json_top_subset
from wepy.util.mdtraj import json_to_mdtraj_topology, mdtraj_to_json_topology

import mdtraj as mdj

class WalkerReporter(ProgressiveFileReporter):


    # the order the files are in
    FILE_ORDER = ('init_state_path', 'walker_path')

    # the extnesions that will be used in orchestration
    SUGGESTED_EXTENSIONS = ('init_top.pdb', 'walkers.dcd')

    def __init__(self, *,
                 init_state=None,
                 json_topology=None,
                 main_rep_idxs=None,
                 **kwargs):

        super().__init__(**kwargs)

        assert json_topology is not None, "must give a JSON format topology"
        assert main_rep_idxs is not None, "must give the indices of the atoms the topology represents"
        assert init_state is not None, "must give an init state for the topology PDB"

        self.main_rep_idxs = main_rep_idxs

        # take a subset of the topology using the main rep atom idxs
        self.json_main_rep_top = json_top_subset(json_topology, self.main_rep_idxs)

        # get the main rep idxs only
        self.init_main_rep_positions = init_state['positions'][self.main_rep_idxs]

        # convert the box vectors
        self.init_unitcell_lengths, self.init_unitcell_angles = box_vectors_to_lengths_angles(
                                                                       init_state['box_vectors'])


    def init(self, **kwargs):

        super().init(**kwargs)

        # load the json topology as an mdtraj one
        mdtraj_top = json_to_mdtraj_topology(self.json_main_rep_top)

        # make a traj for the initial state to use as a topology for
        # visualizing the walkers
        init_traj = mdj.Trajectory([self.init_main_rep_positions],
                                   unitcell_lengths=[self.init_unitcell_lengths],
                                   unitcell_angles=[self.init_unitcell_angles],
                                   topology=mdtraj_top)

        # write out the init traj as a pdb
        init_traj.save_pdb(self.init_state_path)

    def report(self, cycle_idx=None, new_walkers=None,
               **kwargs):

        # load the json topology as an mdtraj one
        mdtraj_top = json_to_mdtraj_topology(self.json_main_rep_top)

        # slice off the main_rep indices because that is all we want
        # to write for these
        main_rep_positions = np.array([walker.state['positions'][self.main_rep_idxs]
                                       for walker in new_walkers])

        # convert the box vectors
        unitcell_lengths, unitcell_angles = traj_box_vectors_to_lengths_angles(
            np.array([walker.state['box_vectors'] for walker in new_walkers]))

        # make a trajectory from these walkers
        traj = mdj.Trajectory(main_rep_positions,
                              unitcell_lengths=unitcell_lengths,
                              unitcell_angles=unitcell_angles,
                              topology=mdtraj_top)


        # write to the file for this trajectory
        traj.save_dcd(self.walker_path)
