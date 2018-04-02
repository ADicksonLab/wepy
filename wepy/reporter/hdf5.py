from copy import deepcopy

import numpy as np

from wepy.reporter.reporter import FileReporter
from wepy.hdf5 import WepyHDF5
from wepy.util.util import json_top_atom_count

class WepyHDF5Reporter(FileReporter):

    ALL_ATOMS_REP_KEY = 'all_atoms'

    def __init__(self, file_path, mode='a',
                 save_fields=None,
                 topology=None,
                 units=None,
                 sparse_fields=None,
                 feature_shapes=None, feature_dtypes=None,
                 n_dims=None,
                 main_rep_idxs=None,
                 all_atoms_rep_freq=None,
                 # dictionary of alt_rep keys and a tuple of (idxs, freq)
                 alt_reps=None,

                 # pass in the resampler and boundary
                 # conditions classes to automatically extract the
                 # needed data, the objects themselves are not saves
                 resampler=None,
                 boundary_conditions=None,

                 # or pass the things we need from them in manually
                 resampling_fields=None,
                 resampler_fields=None,
                 warping_fields=None,
                 progress_fields=None,
                 bc_fields=None,
                 ):

        super().__init__(file_path, mode=mode)
        self.wepy_run_idx = None
        self._tmp_topology = topology
        # which fields from the walker to save, if None then save all of them
        self.save_fields = save_fields
        # dictionary of sparse_field_name -> int : frequency of cycles
        # to save the field
        self.sparse_fields = sparse_fields
        self.feature_shapes = feature_shapes
        self.feature_dtypes = feature_dtypes
        self.n_dims = n_dims

        # get and set the record fields (naems, shapes, dtypes) for
        # the resampler and the boundary conditions
        if resampling_fields is not None:
            self.resampling_fields = resampling_fields
        elif resampler is not None:
            self.resampling_fields = resampler.resampling_fields()
        else:
            self.resampling_fields = None

        if resampler_fields is not None:
            self.resampler_fields = resampler_fields()
        elif resampler is not None:
            self.resampler_fields = resampler.resampler_fields()
        else:
            self.resampler_fields = None

        if warping_fields is not None:
            self.warping_fields = warping_fields()
        elif boundary_conditions is not None:
            self.warping_fields = boundary_conditions.warping_fields()
        else:
            self.warping_fields = None

        if progress_fields is not None:
            self.progress_fields = progress_fields()
        elif boundary_conditions is not None:
            self.progress_fields = boundary_conditions.progress_fields()
        else:
            self.progress_fields = None

        if bc_fields is not None:
            self.bc_fields = bc_fields()
        elif boundary_conditions is not None:
            self.bc_fields = boundary_conditions.bc_fields()
        else:
            self.bc_fields = None

        # the atom indices of the whole system that will be saved as
        # the main positions representation
        self.main_rep_idxs = main_rep_idxs

        # the idxs for alternate representations of the system
        # positions
        if alt_reps is not None:
            self.alt_reps_idxs = {key: list(tup[0]) for key, tup in alt_reps.items()}

            # add the frequencies for these alt_reps to the
            # sparse_fields frequency dictionary
            self.sparse_fields.update({"alt_reps/{}".format(key): tup[1] for key, tup
                                       in alt_reps.items()})
        else:
            self.alt_reps_idxs = {}

        # check for alt_reps of this name because this is reserved for
        # the all_atoms flag.
        if self.ALL_ATOMS_REP_KEY in self.alt_reps_idxs:
            raise ValueError("Cannot name an alt_rep 'all_atoms'")

        # if there is a frequency for all atoms rep then we make an
        # alt_rep for the all_atoms system with the specified
        # frequency
        if all_atoms_rep_freq is not None:
            # count the number of atoms in the topology and set the
            # alt_reps to have the full slice for all atoms
            n_atoms = json_top_atom_count(self._tmp_topology)
            self.alt_reps_idxs[self.ALL_ATOMS_REP_KEY] = np.arange(n_atoms)
            # add the frequency for this sparse fields to the
            # sparse fields dictionary
            self.sparse_fields["alt_reps/{}".format(self.ALL_ATOMS_REP_KEY)] = all_atoms_rep_freq

        # if there are no sparse fields set it as an empty dictionary
        if self.sparse_fields is None:
            self.sparse_fields = {}

        # if units were given add them otherwise set as an empty dictionary
        if units is None:
            self.units = {}
        else:
            self.units = units


    def init(self):

        # open and initialize the HDF5 file

        self.wepy_h5 = WepyHDF5(self.file_path, mode=self.mode,
                                topology=self._tmp_topology,
                                units=self.units,
                                sparse_fields=list(self.sparse_fields.keys()),
                                feature_shapes=self.feature_shapes,
                                feature_dtypes=self.feature_dtypes,
                                n_dims=self.n_dims,
                                main_rep_idxs=self.main_rep_idxs,
                                alt_reps=self.alt_reps_idxs)

        with self.wepy_h5:
            # initialize a new run
            run_grp = self.wepy_h5.new_run()
            self.wepy_run_idx = run_grp.attrs['run_idx']

            # initialize the run record groups using their fields
            self.wepy_h5.init_run_fields_resampling(self.wepy_run_idx, self.resampling_fields)
            self.wepy_h5.init_run_fields_resampler(self.wepy_run_idx, self.resampler_fields)
            self.wepy_h5.init_run_fields_warping(self.wepy_run_idx, self.warping_fields)
            self.wepy_h5.init_run_fields_progress(self.wepy_run_idx, self.progress_fields)
            self.wepy_h5.init_run_fields_bc(self.wepy_run_idx, self.bc_fields)

        # if this was opened in a truncation mode, we don't want to
        # overwrite old runs with future calls to init(). so we
        # change the mode to read/write 'r+'
        if self.mode == 'w':
            self.mode = 'r+'

    def cleanup(self, *args):

        # it should be already closed at this point but just in case
        if not self.wepy_h5.closed:
            self.wepy_h5.close()


    def report(self, cycle_idx, walkers,
               warp_data, bc_data, progress_data,
               resampling_data, resampler_data,
               debug_prints=False):

        n_walkers = len(walkers)

        # determine which fields to save. If there were none specified
        # save all of them
        if self.save_fields is None:
            save_fields = list(walkers[0].dict().keys())
        else:
            save_fields = self.save_fields

        with self.wepy_h5:

            # add trajectory data for the walkers
            for walker_idx, walker in enumerate(walkers):

                walker = deepcopy(walker)
                walker_data = walker.state.dict()

                # iterate through the feature vectors of the walker
                # (fields), and the keys for the alt_reps
                for field_path in list(walker_data.keys()):

                    # save the field if it is in the list of save_fields
                    if field_path not in save_fields:
                        walker_data.pop(field_path)
                        continue

                    # if the result is None don't save anything
                    if walker_data[field_path] is None:
                        walker_data.pop(field_path)
                        continue

                    # if this is a sparse field we decide
                    # whether it is a valid cycle to save on
                    if field_path in self.sparse_fields:
                        if cycle_idx % self.sparse_fields[field_path] != 0:
                            # this is not a valid cycle so we
                            # remove from the walker_data
                            walker_data.pop(field_path)
                            continue


                # Add the alt_reps fields by slicing the positions
                for alt_rep_key, alt_rep_idxs in self.alt_reps_idxs.items():
                    alt_rep_path = "alt_reps/{}".format(alt_rep_key)
                    # check to make sure this is a cycle this is to be
                    # saved to, if it is add it to the walker_data
                    if cycle_idx % self.sparse_fields[alt_rep_path] == 0:
                        # if the idxs are None we want all of the atoms
                        if alt_rep_idxs is None:
                            alt_rep_data = walker_data['positions'][:]
                        # otherwise get only th atoms we want
                        else:
                            alt_rep_data = walker_data['positions'][alt_rep_idxs]
                        walker_data[alt_rep_path] = alt_rep_data


                # lastly reduce the atoms for the main representation
                # if this option was given
                if self.main_rep_idxs is not None:
                    walker_data['positions'] = walker_data['positions'][self.main_rep_idxs]


                # for all of these fields we wrap them in another
                # dimension to make them feature vectors
                for field_path in list(walker_data.keys()):
                    walker_data[field_path] = np.array([walker_data[field_path]])

                # save the data to the HDF5 file for this walker

                # check to see if the walker has a trajectory in the run
                if walker_idx in self.wepy_h5.run_traj_idxs(self.wepy_run_idx):

                    # if it does then append to the trajectory
                    self.wepy_h5.extend_traj(self.wepy_run_idx, walker_idx,
                                             weights=np.array([[walker.weight]]),
                                             data=walker_data)
                # start a new trajectory
                else:
                    # add the traj for the walker with the data

                    traj_grp = self.wepy_h5.add_traj(self.wepy_run_idx,
                                                     weights=np.array([[walker.weight]]),
                                                     data=walker_data)

                    # add as metadata the cycle idx where this walker started
                    traj_grp.attrs['starting_cycle_idx'] = cycle_idx


            # report the boundary conditions records data
            self.report_warping(cycle_idx, warp_data)
            self.report_bc(cycle_idx, bc_data)
            self.report_progress(cycle_idx, progress_data)

            # report the resampling records data
            self.report_resampling(cycle_idx, resampling_data)
            self.report_resampler(cycle_idx, resampler_data)


    # sporadic
    def report_warping(self, cycle_idx, warping_data):

        if len(warping_data) > 0:
            self.wepy_h5.append_warping_records(self.wepy_run_idx, cycle_idx, warping_data)

    def report_bc(self, cycle_idx, bc_data):

        if len(bc_data) > 0:
            self.wepy_h5.append_bc_records(self.wepy_run_idx, cycle_idx, bc_data)

    def report_resampler(self, cycle_idx, resampler_data):

        if len(resampler_data) > 0:
            self.wepy_h5.append_resampler_records(self.wepy_run_idx, cycle_idx, resampler_data)

    # the resampling records are provided every cycle but they need to
    # be saved as sporadic because of the variable number of walkers
    def report_resampling(self, cycle_idx, resampling_data):

        self.wepy_h5.append_resampling_records(self.wepy_run_idx, cycle_idx, resampling_data)

    # continual
    def report_progress(self, cycle_idx, progress_data):

        self.wepy_h5.append_progress_records(self.wepy_run_idx, cycle_idx, [progress_data])



