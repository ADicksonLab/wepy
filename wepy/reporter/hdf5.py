import numpy as np

from wepy.reporter.reporter import FileReporter
from wepy.hdf5 import WepyHDF5

class WepyHDF5Reporter(FileReporter):

    def __init__(self, file_path, mode='a',
                 decisions=None, instruction_dtypes=None,
                 bc_dtype=None,
                 topology=None):

        super().__init__(file_path, mode=mode)
        self.wepy_run_idx = None
        self._tmp_topology = topology
        self.decisions = decisions
        self.instruction_dtypes = instruction_dtypes
        self.bc_dtype = bc_dtype

    def init(self):

        # open and initialize the HDF5 file
        self.wepy_h5 = WepyHDF5(self.file_path, topology=self._tmp_topology, mode=self.mode)
        # save space and delete the temp topology from the attributes
        del self._tmp_topology

        # initialize a new run in a context
        with self.wepy_h5 as wepy_h5:
            run_grp = wepy_h5.new_run()
            self.wepy_run_idx = run_grp.attrs['run_idx']

            # initialize the resampling group within this run
            wepy_h5.init_run_resampling(self.wepy_run_idx,
                                   self.decisions,
                                   self.instruction_dtypes)

            # initialize the boundary condition group within this run
            wepy_h5.init_run_bc(self.wepy_run_idx, self.bc_dtype)


    def report(self, cycle_idx, walkers,
               warp_records, warp_data,
               resampling_records, resampling_data,
               debug_prints=False):

        n_walkers = len(walkers)

        with self.wepy_h5 as wepy_h5:

            # add trajectory data for the walkers
            for walker_idx, walker in enumerate(walkers):

                # collect data from walker
                walker_data = {}
                for key, value in walker.dict().items():
                    # if the result is None exclude it from the data
                    if value is not None:
                        walker_data[key] = np.array([value])

                # check to see if the walker has a trajectory in the run
                if walker_idx in wepy_h5.run_traj_idxs(self.wepy_run_idx):
                    # if it does then append to the trajectory
                    wepy_h5.append_traj(self.wepy_run_idx, walker_idx,
                                             weights=np.array([walker.weight]),
                                             **walker_data)
                # start a new trajectory
                else:
                    # add the traj for the walker with the data
                    traj_grp = wepy_h5.add_traj(self.wepy_run_idx, weights=np.array([walker.weight]),
                                                     **walker_data)
                    # add as metadata the cycle idx where this walker started
                    traj_grp.attrs['starting_cycle_idx'] = cycle_idx


            # add warp records from boundary conditions
            wepy_h5.add_cycle_warp_records(self.wepy_run_idx, warp_records)

            # add resampling records
            wepy_h5.add_cycle_resampling_records(self.wepy_run_idx, resampling_records)

        # TODO
        # add the warp data
        # self.add_cycle_warp_data(self.wepy_run_idx, warp_data)
        # add the resampling data
        # self.add_resampling_data(self.wepy_run_idx, resampling_data)


    def cleanup(self):

        # it should be already closed at this point but just in case
        if not self.wepy_h5.closed:
            self.wepy_h5.close()
