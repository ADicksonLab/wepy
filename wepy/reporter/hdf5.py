import numpy as np

from wepy.reporter.reporter import FileReporter
from wepy.hdf5 import WepyHDF5

class WepyHDF5Reporter(FileReporter):

    def __init__(self, file_path, mode='a',
                 decisions=None, instruction_dtypes=None,
                 resampling_aux_dtypes=None, resampling_aux_shapes=None,
                 warp_dtype=None,
                 warp_aux_dtypes=None, warp_aux_shapes=None,
                 bc_dtype=None,
                 bc_aux_dtypes=None, bc_aux_shapes=None,
                 topology=None,
                 units=None,
                 **kwargs):

        super().__init__(file_path, mode=mode)
        self.wepy_run_idx = None
        self._tmp_topology = topology
        self.decisions = decisions
        self.instruction_dtypes = instruction_dtypes
        self.warp_dtype = warp_dtype
        self.bc_dtype = bc_dtype
        self.resampling_aux_dtypes = resampling_aux_dtypes
        self.resampling_aux_shapes = resampling_aux_shapes
        self.warp_aux_dtypes = warp_aux_dtypes
        self.warp_aux_shapes = warp_aux_shapes
        self.bc_aux_dtypes = bc_aux_dtypes
        self.bc_aux_shapes = bc_aux_shapes
        self.kwargs = kwargs


        # if units were given add them otherwise set as an empty dictionary
        if units is None:
            self.units = {}
        else:
            self.units = units

    def init(self):

        # open and initialize the HDF5 file

        self.wepy_h5 = WepyHDF5(self.file_path, mode=self.mode,
                                topology=self._tmp_topology,  **self.units, **self.kwargs)

        # save space and delete the temp topology from the attributes
        # del self._tmp_topology

        # initialize a new run in a context
        with self.wepy_h5 as wepy_h5:
            run_grp = wepy_h5.new_run()
            self.wepy_run_idx = run_grp.attrs['run_idx']

            # initialize the resampling group within this run
            wepy_h5.init_run_resampling(self.wepy_run_idx,
                                        self.decisions,
                                        self.instruction_dtypes,
                                        resampling_aux_dtypes=self.resampling_aux_dtypes,
                                        resampling_aux_shapes=self.resampling_aux_shapes)

            # initialize the boundary condition group within this run
            wepy_h5.init_run_warp(self.wepy_run_idx, self.warp_dtype,
                                  warp_aux_dtypes=self.warp_aux_dtypes,
                                  warp_aux_shapes=self.warp_aux_shapes)

            # initialize the boundary condition group within this run
            wepy_h5.init_run_bc(self.wepy_run_idx,
                                  bc_aux_dtypes=self.bc_aux_dtypes,
                                  bc_aux_shapes=self.bc_aux_shapes)

        # if this was opened in a truncation mode, we don't want to
        # overwrite old runs with future calls to init(). so we
        # change the mode to read/write 'r+'
        if self.mode == 'w':
            self.mode = 'r+'



    def report(self, cycle_idx, walkers,
               warp_records, warp_aux_data,
               bc_records, bc_aux_data,
               resampling_records, resampling_aux_data,
               debug_prints=False):

        n_walkers = len(walkers)

        with self.wepy_h5 as wepy_h5:

            # add trajectory data for the walkers
            for walker_idx, walker in enumerate(walkers):

                # collect data from walker
                walker_data = {}
                # iterate through the feature vectors
                for key, vector in walker.dict().items():
                    # if the result is None exclude it from the data
                    if vector is not None:
                        walker_data[key] = np.array([vector])

                # check to see if the walker has a trajectory in the run
                if walker_idx in wepy_h5.run_traj_idxs(self.wepy_run_idx):
                    # if it does then append to the trajectory
                    wepy_h5.extend_traj(self.wepy_run_idx, walker_idx,
                                             weights=np.array([[walker.weight]]),
                                             **walker_data)
                # start a new trajectory
                else:
                    # add the traj for the walker with the data
                    traj_grp = wepy_h5.add_traj(self.wepy_run_idx, weights=np.array([walker.weight]),
                                                     **walker_data)
                    # add as metadata the cycle idx where this walker started
                    traj_grp.attrs['starting_cycle_idx'] = cycle_idx



            # if there was warping done by the boundary conditions save those records
            if len(warp_records) > 0:
                # add warp records
                wepy_h5.add_cycle_warp_records(self.wepy_run_idx, warp_records)

                # add warp data
                wepy_h5.add_cycle_warp_aux_data(self.wepy_run_idx, warp_aux_data)

            # TODO add boundary condition records
            # wepy_h5.add_bc_records(self.wepy_run_idx, bc_records)
            # if there is any boundary conditions function
            if len(bc_aux_data) > 0:
                # add the auxiliary data from checking boundary conditions
                wepy_h5.add_cycle_bc_aux_data(self.wepy_run_idx, bc_aux_data)

            # add resampling records
            wepy_h5.add_cycle_resampling_records(self.wepy_run_idx, resampling_records)

            # add resampling data
            wepy_h5.add_cycle_resampling_aux_data(self.wepy_run_idx, resampling_aux_data)


    def cleanup(self):

        # it should be already closed at this point but just in case
        if not self.wepy_h5.closed:
            self.wepy_h5.close()
