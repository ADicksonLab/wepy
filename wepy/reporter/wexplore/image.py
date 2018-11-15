import numpy as np

import mdtraj as mdj

from wepy.reporter.reporter import ProgressiveFileReporter
from wepy.util.mdtraj import json_to_mdtraj_topology, mdtraj_to_json_topology
from wepy.util.util import json_to_mdtraj_topology, mdtraj_to_json_topology, json_top_subset

class WExploreAtomImageReporter(ProgressiveFileReporter):

    FILE_ORDER = ("init_state_path", "image_path")
    SUGGESTED_EXTENSIONS = ("image_top.pdb", "wexplore_images.dcd")


    def __init__(self,
                 init_state=None,
                 image_atom_idxs=None,
                 json_topology=None,
                 **kwargs):

        super().__init__(**kwargs)

        assert init_state is not None, "must give the initial state to generate the initial image"
        assert json_topology is not None, "must give a JSON format topology"
        assert image_atom_idxs is not None, \
            "must give the indices of the atoms for the subset of the topology that is the image"

        self.image_atom_idxs = image_atom_idxs

        # get the image using the image atom indices from the
        # init_state positions
        self.init_image_positions = init_state['positions'][self.image_atom_idxs]

        self.json_main_rep_top = json_top_subset(json_topology, self.image_atom_idxs)

        # the array for the positions of the trajectory images
        self.image_traj_positions = [self.init_image_positions]

        # and times
        self.times = [0]


    def init(self, **kwargs):

        super().init(**kwargs)

        image_mdj_topology = json_to_mdtraj_topology(self.json_main_rep_top)

        # initialize the initial image into the image traj
        init_image_traj = mdj.Trajectory([self.init_image_positions],
                                         time=self.times,
                                         topology=image_mdj_topology)



        # save this as a PDB for a topology to view in VMD etc. to go
        # along with the trajectory we will make
        init_image_traj.save_pdb(self.init_state_path)

    def report(self, cycle_idx=None, resampler_data=None,
               **kwargs):

        # load the json topology as an mdtraj one
        image_mdj_topology = json_to_mdtraj_topology(self.json_main_rep_top)

        # collect the new images defined
        new_images = []
        for resampler_rec in resampler_data:
            image = resampler_rec['image']
            new_images.append(image)

        new_images = np.array(new_images)
        times = np.array([cycle_idx + 1 for _ in range(len(new_images))])


        # combine the new image positions and times with the old
        self.image_traj_positions.extend(new_images)
        self.times.extend(times)

        # make a trajectory of the new images, using the cycle_idx as the time
        new_image_traj = mdj.Trajectory(self.image_traj_positions,
                                        time=self.times,
                                        topology=image_mdj_topology)

        # then write it to the file
        new_image_traj.save_dcd(self.image_path)
