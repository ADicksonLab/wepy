import logging

import numpy as np

import mdtraj as mdj

from wepy.reporter.reporter import ProgressiveFileReporter
from wepy.util.mdtraj import json_to_mdtraj_topology, mdtraj_to_json_topology
from wepy.util.json_top import json_top_subset

class WExploreAtomImageReporter(ProgressiveFileReporter):
    """Reporter for generating 3D molecular structures from WExplore
    region images.

    This will only be meaningful for WExplore simulations where the
    region images are actually 3D coordinates.

    """

    FILE_ORDER = ("init_state_path", "image_path")
    SUGGESTED_EXTENSIONS = ("image_top.pdb", "wexplore_images.dcd")


    def __init__(self,
                 init_image=None,
                 image_atom_idxs=None,
                 json_topology=None,
                 **kwargs):
        """Constructor for the WExploreAtomImageReporter.

        Parameters
        ----------

        init_image : numpy.array, optional
            The initial region image. Used for generating the topology
            as well. If not given will be eventually generated.
            (Default = None)

        image_atom_idxs : list of int
            The indices of the atoms that are part of the topology
            subset that comprises the image.

        json_topology : str
            JSON format topology for the whole system. A subset of the
            atoms will be taken using the image_atom_idxs.

        """

        super().__init__(**kwargs)

        assert json_topology is not None, "must give a JSON format topology"
        assert image_atom_idxs is not None, \
            "must give the indices of the atoms for the subset of the topology that is the image"

        self.image_atom_idxs = image_atom_idxs

        self.json_main_rep_top = json_top_subset(json_topology, self.image_atom_idxs)

        self.init_image = None
        self._top_pdb_written = False
        self.image_traj_positions = []

        # if an initial image was given use it, otherwise just don't
        # worry about it, the reason for this is that there is no
        # interface for getting image indices from distance metrics as
        # of now.
        if init_image is not None:
            self.init_image = init_image
            self.image_traj_positions.append(self.init_image)

        # and times
        self.times = [0]


    def init(self, **kwargs):

        super().init(**kwargs)

        if self.init_image is not None:

            image_mdj_topology = json_to_mdtraj_topology(self.json_main_rep_top)

            # initialize the initial image into the image traj
            init_image_traj = mdj.Trajectory([self.init_image],
                                             time=self.times,
                                             topology=image_mdj_topology)



            # save this as a PDB for a topology to view in VMD etc. to go
            # along with the trajectory we will make
            logging.info("Writing initial image to {}".format(self.init_state_path))
            init_image_traj.save_pdb(self.init_state_path)

            self._top_pdb_written = True

    def report(self, cycle_idx=None, resampler_data=None,
               **kwargs):

        # load the json topology as an mdtraj one
        image_mdj_topology = json_to_mdtraj_topology(self.json_main_rep_top)

        # collect the new images defined
        new_images = []
        for resampler_rec in resampler_data:
            image = resampler_rec['image']
            new_images.append(image)

        times = np.array([cycle_idx + 1 for _ in range(len(new_images))])


        # combine the new image positions and times with the old
        self.image_traj_positions.extend(new_images)
        self.times.extend(times)

        # only save if we have an image yet
        if len(self.image_traj_positions) > 0:

            # make a trajectory of the new images, using the cycle_idx as the time
            new_image_traj = mdj.Trajectory(self.image_traj_positions,
                                            time=self.times,
                                            topology=image_mdj_topology)

            # if we haven't already written a topology PDB write it now
            if not self._top_pdb_written:
                new_image_traj[0].save_pdb(self.init_state_path)
                self._top_pdb_written = True

            # then the images to the trajectory file
            new_image_traj.save_dcd(self.image_path)
