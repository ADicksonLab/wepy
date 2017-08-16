from wepy.reporter import FileReporter
from wepy.hdf5 import WepyHDF5

class WepyHDF5Reporter(FileReporter):

    def __init__(self, file_path, topology, mode='a'):

        super().__init__(file_path, mode=mode)
        self.wepy_run_idx = None
        self._tmp_topology = topology

    def init(self):

        # open and initialize the HDF5 file
        self.wepy_h5 = WepyHDF5(self.file_path, self._tmp_topology, mode=self.mode)
        # save space and delete the temp topology from the attributes
        del self._tmp_topology

        # initialize a new run
        run_grp = self.wepy_h5.new_run()
        self.wepy_run_idx = run_grp.attrs['run_idx']

    def report(self, cycle_idx, walkers, resampling_records):

        n_walkers = len(walkers)

        # add trajectory data for the walkers
        for walker_idx, walker in enumerate(walkers):
            # check to see if the walker has a trajectory in the run
            if walker_idx in self.wepy_h5.run_traj_idxs(self.wepy_run_idx):
                # if it does then append to the trajectory
                self.wepy_h5.append_traj(self.wepy_run_idx, walker_idx,
                                         weights=np.array([walker.weight]),
                                         **walker)
            else:
                # start a new trajectory
                traj_grp = self.wepy_h5.add_traj(self.wepy_run_idx, weights=np.array([walker.weight]),
                                                 **walker)
                # add as metadata the cycle idx where this walker started
                traj_grp.attrs['starting_cycle_idx'] = cycle_idx

