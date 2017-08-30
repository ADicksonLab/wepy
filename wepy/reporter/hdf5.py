import numpy as np

from wepy.reporter.reporter import FileReporter
from wepy.hdf5 import WepyHDF5

class WepyHDF5Reporter(FileReporter):

    def __init__(self, file_path, decisions, instruction_dtypes, topology, mode='a'):

        super().__init__(file_path, mode=mode)
        self.wepy_run_idx = None
        self._tmp_topology = topology
        self.decisions = decisions
        self.instruction_dtypes = instruction_dtypes

    def init(self):

        # open and initialize the HDF5 file
        self.wepy_h5 = WepyHDF5(self.file_path, topology=self._tmp_topology, mode=self.mode)
        # save space and delete the temp topology from the attributes
        del self._tmp_topology

        # initialize a new run
        run_grp = self.wepy_h5.new_run()
        self.run_grp = run_grp
        self.wepy_run_idx = run_grp.attrs['run_idx']

        # initialize the resampling group within this run
        self.wepy_h5.init_run_resampling(self.wepy_run_idx, self.decisions, self.instruction_dtypes)

    def report(self, cycle_idx, walkers, resampling_records, resampling_data):

        n_walkers = len(walkers)

        # add trajectory data for the walkers
        for walker_idx, walker in enumerate(walkers):

            # collect data from walker
            walker_data = {}
            for key, value in walker.dict().items():
                # if the result is None exclude it from the data
                if value is not None:
                    walker_data[key] = np.array([value])

            # check to see if the walker has a trajectory in the run
            if walker_idx in self.wepy_h5.run_traj_idxs(self.wepy_run_idx):
                # if it does then append to the trajectory
                self.wepy_h5.append_traj(self.wepy_run_idx, walker_idx,
                                         weights=np.array([walker.weight]),
                                         **walker_data)
            # start a new trajectory
            else:
                # add the traj for the walker with the data
                traj_grp = self.wepy_h5.add_traj(self.wepy_run_idx, weights=np.array([walker.weight]),
                                                 **walker_data)
                # add as metadata the cycle idx where this walker started
                traj_grp.attrs['starting_cycle_idx'] = cycle_idx

        # add resampling records
        self.wepy_h5.add_cycle_resampling_records(self.wepy_run_idx, resampling_records)

        # TODO
        # add the resampling data
        # self.add_resampling_data(self.wepy_run_idx, resampling_data)


    def cleanup(self):
        self.wepy_h5.close()
