"""Provides a reporter for generating 3D molecular structure files for
walkers.

This is useful for getting a snapshot of what is happening in the
simulation as it progresses.

"""

import logging

import numpy as np

from wepy.reporter.reporter import ProgressiveFileReporter
from wepy.util.util import box_vectors_to_lengths_angles, traj_box_vectors_to_lengths_angles
from wepy.util.json_top import json_top_subset
from wepy.util.mdtraj import json_to_mdtraj_topology, mdtraj_to_json_topology

import mdtraj as mdj

class WalkerReporter(ProgressiveFileReporter):
    """Reporter for generating 3D molecular structure files of the walkers
    produced by a cycle.

    It generates two different files using the mdtraj library: a PDB
    format and a DCD file.

    The PDB is used as a "topology" that is only generated once at the
    beginning of a simulation (in the call to the 'init' method). This
    defines the atom types and bonds for non-protein molecules. Most
    molecular viewers have special algorithms that automatically
    determine bond topologies from amino acid residue designations.

    The DCD is a binary format that carries only information about the
    positions of atoms (essentially a (N_frames, N_atoms, 3) array)
    that is recognized by many different molecular viewer
    software. This is generated every cycle and the number of frames
    is the number of walkers.

    Because this representation is meant to be amenable to visual
    inspection, there is functionality for subsetting atoms from the
    full simulation system (i.e. to remove solvent).

    """


    # the order the files are in
    FILE_ORDER = ('init_state_path', 'walker_path')

    # the extnesions that will be used in orchestration
    SUGGESTED_EXTENSIONS = ('init_top.pdb', 'walkers.dcd')

    def __init__(self, *,
                 init_state=None,
                 json_topology=None,
                 main_rep_idxs=None,
                 **kwargs):
        """Constructor for the WalkerReporter.

        Parameters
        ----------

        init_state : object implementing WalkerState
            An initial state, only used for writing the PDB topology.

        json_topology : str
            A molecular topology in the common JSON format, that
            matches the main_rep_idxs.

        main_rep_idxs : listlike of int
            The indices of the atoms to select from the full representation.

        """

        super().__init__(**kwargs)

        assert json_topology is not None, "must give a JSON format topology"
        assert init_state is not None, "must give an init state for the topology PDB"

        # if the main rep indices were not given infer them as all of the atoms
        if main_rep_idxs is None:
            self.main_rep_idxs = list(range(init_state['positions'].shape[0]))
        else:
            self.main_rep_idxs = main_rep_idxs

        # take a subset of the topology using the main rep atom idxs
        self.json_main_rep_top = json_top_subset(json_topology, self.main_rep_idxs)

        # get the main rep idxs only
        self.init_main_rep_positions = init_state['positions'][self.main_rep_idxs]

        # convert the box vectors
        self.init_unitcell_lengths, self.init_unitcell_angles = box_vectors_to_lengths_angles(
                                                                       init_state['box_vectors'])


    def init(self, **kwargs):
        """Initialize the reporter at simulation time.

        This will generate the initial state PDB file.

        """

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
        logging.info("Writing initial state to {}".format(self.init_state_path))
        init_traj.save_pdb(self.init_state_path)

    def report(self, cycle_idx=None, new_walkers=None,
               **kwargs):
        """Report the current cycle's walker states as 3D molecular
        structures.

        Parameters
        ----------
        cycle_idx : int

        new_walkers : list of Walker objects

        """

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
