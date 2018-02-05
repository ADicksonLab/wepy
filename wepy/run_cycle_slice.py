import numpy as np

import mdtraj as mdj
import numpy as npxx
from wepy.hdf5 import TrajHDF5
from wepy.hdf5 import TRAJ_DATA_FIELDS
import h5py

class RunCycleSlice(object):

    def __init__(self, run_idx, cycles, wepy_hdf5_file):

        self._h5 = wepy_hdf5_file._h5
        self.run_idx = run_idx
        self.cycles = cycles
        self.mode = wepy_hdf5_file.mode

    def traj(self, run_idx, traj_idx):
                 return self._h5['runs/{}/trajectories/{}'.format(run_idx, traj_idx)]
    def run_trajs(self, run_idx):
        return self._h5['runs/{}/trajectories'.format(run_idx)]

    def n_run_trajs(self, run_idx):
        return len(self._h5['runs/{}/trajectories'.format(run_idx)])

    def run_traj_idxs(self, run_idx):
        return range(len(self._h5['runs/{}/trajectories'.format(run_idx)]))

    def run_traj_idx_tuples(self):
        tups = []
        for traj_idx in self.run_traj_idxs(self.run_idx):
            tups.append((self.run_idx, traj_idx))

        return tups
    def run_cycle_idx_tuples(self):
        tups = []
        for cycle_idx in self.cycles:
             tups.append((self.run_idx, cycle_idx))
        return tups


    def iter_trajs(self, idxs=False, traj_sel=None):
        """Generator for all of the trajectories in the dataset across all
        runs. If idxs=True will return a tuple of (run_idx, traj_idx).

        run_sel : if True will iterate over a subset of
        trajectories. Possible values are an iterable of `(run_idx,
        traj_idx)` tuples.

        """


        # set the selection of trajectories to iterate over
        if traj_sel is None:
            idx_tups = self.run_traj_idx_tuples()
        else:
            idx_tups = traj_sel

        # get each traj for each idx_tup and yield them for the generator
        for run_idx, traj_idx in idx_tups:
            traj = self.traj(run_idx, traj_idx)
            if idxs:
                yield (run_idx, traj_idx), traj
            else:
                yield traj

    def iter_cycle_fields(self, fields, cycle_idx, idxs=False, traj_sel=None):
        """Generator for all of the specified non-compound fields
        h5py.Datasets for all trajectories in the dataset across all
        runs. Fields is a list of valid relative paths to datasets in
        the trajectory groups.

        """

        for field in fields:
            dsets = {}
            fields_data = ()
            for idx_tup, traj in self.iter_trajs(idxs=True, traj_sel=traj_sel):
                run_idx, traj_idx = idx_tup
                try:
                    dset = traj[field][cycle_idx]
                    if not isinstance(dset, np.ndarray):
                        dset = np.array([dset])
                    if len(dset.shape)==1:
                        fields_data += (dset,)
                    else:
                        fields_data += ([dset],)
                except KeyError:
                    warn("field \"{}\" not found in \"{}\"".format(field, traj.name), RuntimeWarning)
                    dset = None

            dsets =  np.concatenate(fields_data, axis=0)

            if idxs:
                yield (run_idx, traj_idx, field), dsets
            else:
                yield field, dsets


    def iter_cycles_fields(self, fields, idxs=False, traj_sel=None, debug_prints=False):

        for cycle_idx in self.cycles:
            dsets = {}
            for field, dset in self.iter_cycle_fields(fields, cycle_idx, traj_sel=traj_sel):
                dsets[field] = dset
            if idxs:
                yield (cycle_idx,  dsets)
            else:
                yield dsets


    def traj_cycles_map(self, func, fields, *args, map_func=map, idxs=False, traj_sel=None,
                        debug_prints=False):
        """Function for mapping work onto field of trajectories in the
        WepyHDF5 file object. Similar to traj_map, except `h5py.Group`
        objects cannot be pickled for message passing. So we select
        the fields to serialize instead and pass the `numpy.ndarray`s
        to have the work mapped to them.

        func : the function that will be mapped to trajectory groups

        fields : list of fields that will be serialized into a dictionary
                 and passed to the map function. These must be valid
                 `h5py` path strings relative to the trajectory
                 group. These include the standard fields like
                 'positions' and 'weights', as well as compound paths
                 e.g. 'observables/sasa'.

        map_func : the function that maps the function. This is where
                        parallelization occurs if desired.  Defaults to
                        the serial python map function.

        traj_sel : a trajectory selection. This is a valid `traj_sel`
        argument for the `iter_trajs` function.

        *args : additional arguments to the function. If this is an
                 iterable it will be assumed that it is the appropriate
                 length for the number of trajectories, WARNING: this will
                 not be checked and could result in a run time
                 error. Otherwise single values will be automatically
                 mapped to all trajectories.

        """

        # check the args and kwargs to see if they need expanded for
        # mapping inputs
        mapped_args = []
        for arg in args:
            # if it is a sequence or generator we keep just pass it to the mapper
            if isinstance(arg, list) and not isinstance(arg, str):
                assert len(arg) == len(self.cycles), "Sequence has fewer"
                mapped_args.append(arg)
            # if it is not a sequence or generator we make a generator out
            # of it to map as inputs
            else:
                mapped_arg = (arg for i in range(len(self.cycles)))
                mapped_args.append(mapped_arg)



        results = map_func(func, self.iter_cycles_fields(fields, traj_sel=traj_sel, idxs=False,
                                                        debug_prints=debug_prints),
                           *mapped_args)

        if idxs:
            if traj_sel is None:
                traj_sel = self.run_cycle_idx_tuples()
            return zip(traj_sel, results)
        else:
            return results


    def compute_observable(self, func, fields, *args,
                           map_func=map,
                           traj_sel=None,
                           save_to_hdf5=None, idxs=False, return_results=True,
                           debug_prints=False):
        """Compute an observable on the trajectory data according to a
        function. Optionally save that data in the observables data group for
        the trajectory.
        """

        if save_to_hdf5 is not None:
            assert self.mode in ['w', 'w-', 'x', 'r+', 'c', 'c-'],\
                "File must be in a write mode"
            assert isinstance(save_to_hdf5, str),\
                "`save_to_hdf5` should be the field name to save the data in the `observables` group in each trajectory"
            field_name=save_to_hdf5

            #DEBUG enforce this until sparse trajectories are implemented
            assert traj_sel is None, "no selections until sparse trajectory data is implemented"

        idx =0

        for result in self.traj_cycles_map(func, fields, *args,
                                       map_func=map_func, traj_sel=traj_sel, idxs=True,
                                       debug_prints=debug_prints):
            idx_tup, obs_value = result
            run_idx, traj_idx = idx_tup

            # if we are saving this to the trajectories observables add it as a dataset
            if save_to_hdf5:

                if debug_prints:
                    print("Saving run {} traj {} observables/{}".format(
                        run_idx, traj_idx, field_name))

                # try to get the observables group or make it if it doesn't exist
                try:
                    obs_grp = self.traj(run_idx, traj_idx)['observables']
                except KeyError:

                    if debug_prints:
                        print("Group uninitialized. Initializing.")

                    obs_grp = self.traj(run_idx, traj_idx).create_group('observables')

                # try to create the dataset
                try:
                    obs_grp.create_dataset(field_name, data=obs_value)
                # if it fails we either overwrite or raise an error
                except RuntimeError:
                    # if we are in a permissive write mode we delete the
                    # old dataset and add the new one, overwriting old data
                    if self.mode in ['w', 'w-', 'x', 'r+']:

                        if debug_prints:
                            print("Dataset already present. Overwriting.")

                        del obs_grp[field_name]
                        obs_grp.create_dataset(field_name, data=obs_value)
                    # this will happen in 'c' and 'c-' modes
                    else:
                        raise RuntimeError(
                            "Dataset already exists and file is in concatenate mode ('c' or 'c-')")

            # also return it if requested
            if return_results:
                if idxs:
                    yield idx_tup, obs_value
                else:
                    yield obs_value


    # def calc_angle(self, v1, v2):
    #    return np.degrees(np.arccos(np.dot(v1, v2)/(la.norm(v1) * la.norm(v2))))

    # def calc_length(self, v):
    #    return la.norm(v)

   # def make_cycle_tarj(self, run, cycle_idx, walker_idx, topology):
    #     times = []
    #     weights = [ ]

    #     unitcell_angles = []

    #     unitcell_lengths = []

    #     xyz = np.zeros((walker_num, topology.n_atoms,3))

    #     positions = list(self.hd['/runs/{}/trajectories/{}/positions'.
    #                       format(run, walker_idx, data)])[0:walker_num]

    #     times = list(self.hd['/runs/{}/trajectories/{}/time'.
    #                          format(run, walker_idx, data)])[0:walker_num]


    # def make_traj(self, run, cycle_idx, walkers_idx_list, topology):

    #     walker_num = len(walkers_idx_list)

    #     times = []
    #     weights = [ ]

    #     unitcell_angles = []
    #     unitcell_lengths = []

    #     xyz = np.zeros((walker_num, topology.n_atoms,3))
    #     for walker_idx in walkers_idx_list:
    #        xyz[walker_idx,:,:] = self.get_data(run, cycle_idx,walker_idx, 'positions')
    #        box_vectors = self.get_data(run, cycle_idx, walker_idx, 'box_vectors')
    #        weight = self.get_data(run, cycle_idx, walker_idx, 'weights')
    #        weights.append(weight)
    #        time = self.get_data(run, cycle_idx, walker_idx,'time')
    #        times.append(time)

    #        unitcell_lengths.append( [self.calc_length(v) for v in box_vectors])
    #        unitcell_angles.append(np.array([self.calc_angle(box_vectors[i], box_vectors[j])
    #                                         for i, j in [(0,1), (1,2), (2,0)]]))

    #     traj = mdj.Trajectory(xyz, topology, time=times,
    #                                 unitcell_lengths=unitcell_angles, unitcell_angles=unitcell_angles)
    #     return traj

    # def get_cycle_data(self, run, cycle_idx, walker_num, data_type):
    #      Data = []
    #      for walker_idx in range(walker_num):
    #          field_data = self.get_data(run, cycle_idx, walker_idx, data_type)
    #          Data.append(field_data)
    #      return Data
