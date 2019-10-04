from copy import deepcopy
import logging

import numpy as np

from wepy.reporter.reporter import FileReporter
from wepy.hdf5 import WepyHDF5
from wepy.walker import Walker, WalkerState
from wepy.util.json_top import json_top_atom_count

class WepyHDF5Reporter(FileReporter):
    """Reporter for generating an HDF5 format (WepyHDF5) data file from
    simulations.

    This is the most important reporter as it is the principle output
    format for storing weighted ensemble simulation data.

    Files generated with this reporter can be opened using the
    wepy.hdf5.WepyHDF5 class.

    See Also
    --------

    wepy.hdf5.WepyHDF5


    """

    # this is the name of the dataset that the all atoms will be saved
    # under in the HDF5 alt_reps group
    ALL_ATOMS_REP_KEY = 'all_atoms'

    FILE_ORDER = ('wepy_hdf5_path',)

    # this is the suggested extension for naming WepyHDF5 files used
    # by this reporter, e.g. results.wepy.h5
    SUGGESTED_EXTENSIONS = ('wepy.h5',)

    def __init__(self,
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
                 decision_enum_dict=None,
                 resampler_fields=None,
                 warping_fields=None,
                 progress_fields=None,
                 bc_fields=None,

                 resampling_records=None,
                 resampler_records=None,
                 warping_records=None,
                 bc_records=None,
                 progress_records=None,

                 # other settings
                 swmr_mode=False,

                 **kwargs
                 ):
        """Constructor for the WepyHDF5Reporter.

        Parameters
        ----------

        save_fields : tuple of str, default: None
           A selection of fields from the walker states to be
           stored. Allows for the ignoring of some states. If None all
           fields from states will attempted to be saved.

        topology : str
            JSON string representing topology of system being simulated.

        units : dict of str: str, optional
            Mapping of trajectory field names to string specs
            for units.

        sparse_fields : list of str, optional
            List of trajectory fields that should be initialized as sparse.

        feature_shapes : dict of str: shape_spec, optional
            Mapping of trajectory fields to their shape spec for initialization.

        feature_dtypes : dict of str: dtype_spec, optional
            Mapping of trajectory fields to their shape spec for initialization.

        n_dims : int, default: 3
            Set the number of spatial dimensions for the default
            positions trajectory field.

        alt_reps : dict of str: tuple of (list of int, int), optional
            Specifies that there will be 'alt_reps' of positions each
            named by the keys of this mapping and containing the
            indices in each value list as the first value of the tuple
            and the second value being the frequency at which this
            field gets saved. Setting `all_atoms_rep_freq` is the
            equivalent of setting an entry {'all_atoms' : ([...],
            `all_atoms_rep_freq`)}.

        main_rep_idxs : list of int, optional
            The indices of atom positions to save as the main 'positions'
            trajectory field. Defaults to all atoms.

        all_atoms_rep_freq : int, optional
            The frequency at which to set an 'alt_rep' for all of the
            atoms in a simulation. Will be set as the field
            'alt_rep/all_atoms'.

        resampler : Resampler object, optional but recommended
            The resampler being used for the simulation. Is used as a
            convenient container for a variety of constants needed for
            specifying data for the resampling records. If this is not
            given then these of the Other Parameters below must be
            specified manually: resampling_fields, decision_enum_dict,
            resampler_fields, resampling_records, resampler_records.

        boundary_conditions : BoundaryConditions object, optional but recommended
            The boundary conditions being used for the simulation. Is
            used as a convenient container for a variety of constants
            needed for specifying data for the warping and progress
            records. If this is not given then these of the Other
            Parameters below must be specified manually:
            warping_fields, progress_fields, bc_fields,
            warping_records, bc_records, progress_records

        swmr_mode : bool
           Whether to write to open the HDF5 in single-writer
           multi-reader (SWMR) mode.


        Other Parameters
        ----------------

        resampling_fields : list of str
            The names of the fields for resampling records

        decision_enum_dict : dict of str : int
            Mapping of the names of resampling decision enum to their
            integer values.

        resampler_fields : list of str
            The names of the fields for the resampler records

        warping_fields : list of str
            The names of the fields for the warping records

        progress_fields : list of str
            The names of the fields for the progress records

        bc_fields : list of str
            The names of the fields for the bounadry condition records.

        resampling_records : list of str, optional
            Names of the resampling_fields that will be used in
            table-like views.

        resampler_records : list of str, optional
            Names of the resampler_fields that will be used in
            table-like views.

        warping_records : list of str, optional
            Names of the warping_fields that will be used in
            table-like views.

        bc_records : list of str, optional
            Names of the bc_fields that will be used in table-like
            views.

        progress_records : list of str, optional
            Names of the progress_fields that will be used in
            table-like views.

        """

        # initialize inherited attributes
        super().__init__(**kwargs)

        # set the preference for swmr mode, True or False, if this is
        # True then SWMR mode will be turned on when the file is
        # written to during reporting
        self.swmr_mode = swmr_mode

        # do all the WepyHDF5 specific stuff

        self.wepy_run_idx = None
        self._tmp_topology = topology
        # which fields from the walker to save, if None then save all of them
        self.save_fields = save_fields
        # dictionary of sparse_field_name -> int : frequency of cycles
        # to save the field
        self._sparse_fields = sparse_fields
        self._feature_shapes = feature_shapes
        self._feature_dtypes = feature_dtypes
        self._n_dims = n_dims

        # get and set the record fields (naems, shapes, dtypes) for
        # the resampler and the boundary conditions
        if (resampling_fields is not None) and (decision_enum_dict is not None):
            self.resampling_fields = resampling_fields
            self.decision_enum = decision_enum_dict
        elif resampler is not None:
            self.resampling_fields = resampler.resampling_fields()
            self.decision_enum = resampler.DECISION.enum_dict_by_name()
        else:
            self.resampling_fields = None
            self.decision_enum = None

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


        # the fields which are records for table like reports
        if resampling_records is not None:
            self.resampling_records = resampling_records
        elif resampler is not None:
            self.resampling_records = resampler.resampling_record_field_names()
        else:
            self.resampling_records = None

        if resampler_records is not None:
            self.resampler_records = resampler_records
        elif resampler is not None:
            self.resampler_records = resampler.resampler_record_field_names()
        else:
            self.resampler_records = None

        if bc_records is not None:
            self.bc_records = bc_records
        elif boundary_conditions is not None:
            self.bc_records = boundary_conditions.bc_record_field_names()
        else:
            self.bc_records = None

        if warping_records is not None:
            self.warping_records = warping_records
        elif boundary_conditions is not None:
            self.warping_records = boundary_conditions.warping_record_field_names()
        else:
            self.warping_records = None

        if progress_records is not None:
            self.progress_records = progress_records
        elif boundary_conditions is not None:
            self.progress_records = boundary_conditions.progress_record_field_names()
        else:
            self.progress_records = None


        # the atom indices of the whole system that will be saved as
        # the main positions representation
        self.main_rep_idxs = main_rep_idxs

        # the idxs for alternate representations of the system
        # positions
        if alt_reps is not None:

            self.alt_reps_idxs = {key: list(tup[0]) for key, tup in alt_reps.items()}

            # add the frequencies for these alt_reps to the
            # sparse_fields frequency dictionary
            for key, (idxs, freq) in alt_reps.items():
                alt_rep_key = "alt_reps/{}".format(key)

                # if the frequency is Ellipsis or 1 then we save it
                # every frame and don't make it sparse because that is
                # very innefficient in comparison
                if freq is Ellipsis or freq == 1 or freq == 0:
                    pass
                else:
                    self._sparse_fields[alt_rep_key] = freq

                self.alt_reps_idxs[key] = list(idxs)

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
            self._sparse_fields["alt_reps/{}".format(self.ALL_ATOMS_REP_KEY)] = all_atoms_rep_freq

        # if there are no sparse fields set it as an empty dictionary
        if self._sparse_fields is None:
            self._sparse_fields = {}

        # if units were given add them otherwise set as an empty dictionary
        if units is None:
            self.units = {}
        else:
            self.units = units


    def init(self, continue_run=None,
             init_walkers=None,
             **kwargs):

        # do the inherited stuff
        super().init(**kwargs)

        # open and initialize the HDF5 file
        logging.info("Initializing HDF5 file at {}".format(self.file_path))

        self.wepy_h5 = WepyHDF5(self.file_path, mode=self.mode,
                                topology=self._tmp_topology,
                                units=self.units,
                                sparse_fields=list(self._sparse_fields.keys()),
                                feature_shapes=self._feature_shapes,
                                feature_dtypes=self._feature_dtypes,
                                n_dims=self._n_dims,
                                main_rep_idxs=self.main_rep_idxs,
                                alt_reps=self.alt_reps_idxs)

        # if we specify save fields only save these for the initial walkers
        if self.save_fields is not None:

            state_fields = list(init_walkers[0].state.dict().keys())

            # make sure all the save_fields are present in the state
            assert all([True if save_field in state_fields else False
                        for save_field in self.save_fields]), \
                            "Not all specified save_fields present in walker states"

            filtered_init_walkers = []
            for walker in init_walkers:
                # make a new state by filtering the attributes of the old ones
                state_d = {k : v for k, v in walker.state.dict().items()
                           if k in self.save_fields}

                # and saving alternate representations as we would
                # expect them

                # if there are any alternate representations set them
                for alt_rep_name, alt_rep_idxs in self.alt_reps_idxs.items():

                    alt_rep_path = 'alt_reps/{}'.format(alt_rep_name)

                    # if the idxs are None we want all of the atoms
                    if alt_rep_idxs is None:
                        state_d[alt_rep_path] = state_d['positions'][:]
                    # otherwise get only the atoms we want
                    else:
                        state_d[alt_rep_path] = state_d['positions'][alt_rep_idxs]

                # if the main rep is different then the full state
                # positions set that
                if self.main_rep_idxs is not None:
                    state_d['positions'] = state_d['positions'][self.main_rep_idxs]

                # then making the new state
                new_state = WalkerState(**state_d)

                filtered_init_walkers.append(Walker(new_state, walker.weight))
        # otherwise save the full state
        else:
            filtered_init_walkers = init_walkers

        self.wepy_h5.set_mode(mode='r+')
        with self.wepy_h5:

            # if this is a continuation run of another run we want to
            # initialize it as such

            # initialize a new run
            run_grp = self.wepy_h5.new_run(filtered_init_walkers, continue_run=continue_run)
            self.wepy_run_idx = run_grp.attrs['run_idx']

            # initialize the run record groups using their fields
            self.wepy_h5.init_run_fields_resampling(self.wepy_run_idx, self.resampling_fields)
            # the enumeration for the values of resampling
            self.wepy_h5.init_run_fields_resampling_decision(self.wepy_run_idx, self.decision_enum)
            self.wepy_h5.init_run_fields_resampler(self.wepy_run_idx, self.resampler_fields)
            # set the fields that are records for tables etc. unless
            # they are already set
            if 'resampling' not in self.wepy_h5.record_fields:
                self.wepy_h5.init_record_fields('resampling', self.resampling_records)
            if 'resampler' not in self.wepy_h5.record_fields:
                self.wepy_h5.init_record_fields('resampler', self.resampler_records)

            # if there were no warping fields set there is no boundary
            # conditions and we don't initialize them
            if self.warping_fields is not None:
                self.wepy_h5.init_run_fields_warping(self.wepy_run_idx, self.warping_fields)
                self.wepy_h5.init_run_fields_progress(self.wepy_run_idx, self.progress_fields)
                self.wepy_h5.init_run_fields_bc(self.wepy_run_idx, self.bc_fields)
                # table records
                if 'warping' not in self.wepy_h5.record_fields:
                    self.wepy_h5.init_record_fields('warping', self.warping_records)
                if 'boundary_conditions' not in self.wepy_h5.record_fields:
                    self.wepy_h5.init_record_fields('boundary_conditions', self.bc_records)
                if 'progress' not in self.wepy_h5.record_fields:
                    self.wepy_h5.init_record_fields('progress', self.progress_records)

        # if this was opened in a truncation mode, we don't want to
        # overwrite old runs with future calls to init(). so we
        # change the mode to read/write 'r+'
        if self.mode == 'w':
            self.set_mode(0, 'r+')

    def cleanup(self, **kwargs):

        # it should be already closed at this point but just in case
        if not self.wepy_h5.closed:
            self.wepy_h5.close()

        # remove reference to the WepyHDF5 file so we can serialize this object
        del self.wepy_h5

        super().cleanup(**kwargs)


    def report(self, new_walkers=None,
               cycle_idx=None,
               warp_data=None,
               bc_data=None,
               progress_data=None,
               resampling_data=None,
               resampler_data=None,
               **kwargs):

        n_walkers = len(new_walkers)

        # determine which fields to save. If there were none specified
        # save all of them
        if self.save_fields is None:
            save_fields = list(new_walkers[0].state.dict().keys())
        else:
            save_fields = self.save_fields

        with self.wepy_h5:

            # turn on SWMR mode if requested for reporting
            if self.swmr_mode:
                self.wepy_h5.swmr_mode = True

            # add trajectory data for the walkers
            for walker_idx, walker in enumerate(new_walkers):

                walker_weight = walker.weight
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
                    if field_path in self._sparse_fields:
                        if cycle_idx % self._sparse_fields[field_path] != 0:
                            # this is not a valid cycle so we
                            # remove from the walker_data
                            walker_data.pop(field_path)
                            continue


                # Add the alt_reps fields by slicing the positions
                for alt_rep_key, alt_rep_idxs in self.alt_reps_idxs.items():
                    alt_rep_path = "alt_reps/{}".format(alt_rep_key)

                    # if the alt rep is also a sparse field check this
                    if alt_rep_path in self._sparse_fields:

                        # check to make sure this is a cycle this is
                        # to be saved to, if it is not continue on to
                        # the next field without saving this one
                        if cycle_idx % self._sparse_fields[alt_rep_path] != 0:

                            continue

                    # slice them and save them

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
                                             weights=np.array([[walker_weight]]),
                                             data=walker_data)
                # start a new trajectory
                else:
                    # add the traj for the walker with the data

                    traj_grp = self.wepy_h5.add_traj(self.wepy_run_idx,
                                                     weights=np.array([[walker_weight]]),
                                                     data=walker_data)

                    # add as metadata the cycle idx where this walker started
                    traj_grp.attrs['cycle_idx'] = cycle_idx


            # report the boundary conditions records data, if boundary
            # conditions were initialized
            if self.warping_fields is not None:
                self._report_warping(cycle_idx, warp_data)
                self._report_bc(cycle_idx, bc_data)
                self._report_progress(cycle_idx, progress_data)

            # report the resampling records data
            self._report_resampling(cycle_idx, resampling_data)

            self._report_resampler(cycle_idx, resampler_data)

        super().report(**kwargs)


    # sporadic
    def _report_warping(self, cycle_idx, warping_data):
        """Method to write warping specific information.

        Parameters
        ----------
        cycle_idx : int

        warp_data : list of dict of str : value
            List of dict-like records for each warping event from the
            last cycle.

        """

        if len(warping_data) > 0:
            self.wepy_h5.extend_cycle_warping_records(self.wepy_run_idx, cycle_idx, warping_data)

    def _report_bc(self, cycle_idx, bc_data):
        """Method to write boundary condition update specific information.

        Parameters
        ----------
        cycle_idx : int

        bc_data : list of dict of str : value
           List of dict-like records specifying the changes to the
           state of the boundary conditions in the last cycle.

        """

        if len(bc_data) > 0:
            self.wepy_h5.extend_cycle_bc_records(self.wepy_run_idx, cycle_idx, bc_data)

    def _report_resampler(self, cycle_idx, resampler_data):
        """Method to write resampler update specific information.

        Parameters
        ----------
        cycle_idx : int

        resampler_data : list of dict of str : value
            List of records specifying the changes to the state of the
            resampler in the last cycle.

        """

        if len(resampler_data) > 0:
            self.wepy_h5.extend_cycle_resampler_records(self.wepy_run_idx, cycle_idx, resampler_data)

    # the resampling records are provided every cycle but they need to
    # be saved as sporadic because of the variable number of walkers
    def _report_resampling(self, cycle_idx, resampling_data):
        """Method to write resampling specific information.

        Parameters
        ----------
        cycle_idx : int

        resampling_data : list of dict of str : value
            List of records specifying the resampling to occur at this
            cycle.

        """

        self.wepy_h5.extend_cycle_resampling_records(self.wepy_run_idx, cycle_idx, resampling_data)

    # continual
    def _report_progress(self, cycle_idx, progress_data):
        """Method to write progress specific information.

        Parameters
        ----------
        cycle_idx : int

        progress_data : dict str : list
            A record indicating the progress values for each walker in
            the last cycle.

        """

        self.wepy_h5.extend_cycle_progress_records(self.wepy_run_idx, cycle_idx, [progress_data])



