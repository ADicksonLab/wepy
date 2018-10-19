class WExploreAtomImageReporter(ProgressiveFileReporter):

    FILE_ORDER = ("init_state_path", "image_path")
    SUGGESTED_EXTENSIONS = ("top.pdb", "wexplore_images.dcd")


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
        # take a subset of that topology for the image
        self.image_mdj_topology = json_to_mdtraj_topology(json_topology).subset(self.image_atom_idxs)

        # make an image from the initial state
        self.init_image = init_state['positions'][self.image_atom_idxs]

        # initialize the initial image into the image traj
        self.image_traj = mdj.Trajectory([self.init_image],
                                         time=[0],
                                         topology=self.image_mdj_topology)

        # save this as a PDB for a topology to view in VMD etc. to go
        # along with the trajectory we will make
        self.image_traj.save_pdb(self.init_state_path)

    def report(self, cycle_idx=None resampler_data=None):

        # collect the new images defined
        new_images = []
        for resampler_rec in resampler_data:
            image = resampler_rec['image']
            new_images.append(image)

        new_images = np.array(new_images)
        times = np.array([cycle_idx for i in range(len(new_images))])

        # make a trajectory of the new images, using the cycle_idx as the time
        new_image_traj = mdj.Trajectory(new_images,
                                        time=times,
                                        self.image_mdj_topology)

        # add the new images to the image traj
        self.image_traj = mdj.join(self.image_traj, new_image_traj)

        # then write it to the file
        self.image_traj.save_dcd(self.image_path)
