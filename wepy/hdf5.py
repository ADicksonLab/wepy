import h5py


def load_dataset(path):
    return None

class TrajHDF5(object):

    def __init__(self, filename, mode='x', overwrite=True,
                 topology=None,
                 positions=None,
                 time=None,
                 box_vectors=None,
                 velocities=None,
                 positions_unit=None,
                 time_unit=None,
                 box_vectors_unit=None,
                 velocities_unit=None,
                 forces=None,
                 parameters=None,
                 observables=None
    ):
        """Initializes a TrajHDF5 object which is a format for storing
        trajectory data in and HDF5 format file which can be used on
        it's own or encapsulated in a WepyHDF5 object.

        mode:
        r        Readonly, file must exist
        w        Create file, truncate if exists
        x        Create file, fail if exists
        a        Append mode, file must exist

        If `overwrite` is True then the previous data will be
        re-initialized upon this constructor being called.

        """
        assert mode in ['r', 'w', 'x', 'a'], "mode must be either r, w, x or a"

        self._filename = filename

        # open the file
        h5 = h5py.File(filename, mode)
        self._h5 = h5
        self.closed = False

        if mode in ['w', 'x'] and overwrite:
            if topology is None:
                raise ValueError("topology must be given for new files")
            else:
                # set the units
                self.positions_units = positions_unit
                self.time_unit = time_unit
                self.box_vectors_unit = box_vectors_unit
                self.velocities_unit = velocities_unit
                # use the setters to add the data
                self.topology = topology
                self.positions = positions
                self.time = time
                self.box_vectors = box_vectors
                self.velocities = velocities
                self.forces = forces
                self.parameters = parameters
                self.observables = observables


    @property
    def filename(self):
        return self._filename

    def close(self):
        if not self.closed:
            self._h5.close()
            self.closed = True

    def __del__(self):
        self.close()

    @property
    def h5(self):
        return self._h5

    @property
    def topology(self):
        return self._h5['topology']

    @topology.setter
    def topology(self, topology):
        self._h5.create_dataset('topology', data=topology)

    @property
    def positions(self):
        return self._h5['positions']

    @positions.setter
    def positions(self, positions):
        self._h5.create_dataset('positions', data=positions)

    @property
    def time(self):
        return self._h5['time']

    @time.setter
    def time(self, time):
        self._h5.create_dataset('time', data=time)

    @property
    def box_vectors(self):
        return self._h5['box_vectors']

    @box_vectors.setter
    def box_vectors(self, box_vectors):
        self._h5.create_dataset('box_vectors', data=box_vectors)

    @property
    def velocities(self):
        return self._h5['velocities']

    @velocities.setter
    def velocities(self, velocities):
        self._h5.create_dataset('velocities', data=velocities)

    @property
    def forces(self):
        return self._h5['forces']

    @forces.setter
    def forces(self, forces):
        self._h5.create_dataset('forces', data=forces)

    @property
    def parameters(self):
        return self._h5['parameters']

    @parameters.setter
    def parameters(self, parameters):
        self._h5.create_dataset('parameters', data=parameters)

    @property
    def observables(self):
        return self._h5['observables']

    @observables.setter
    def observables(self, observables):
        self._h5.create_dataset('observables', data=observables)



class WepyHDF5(object):

    def __init__(self, filename, mode='x', topology=None, overwrite=True):
        """Initialize a new Wepy HDF5 file. This is a file that organizes
        wepy.TrajHDF5 dataset subsets by simulations by runs and
        includes resampling records for recovering walker histories.

        mode:
        r        Readonly, file must exist
        w        Create file, truncate if exists
        x        Create file, fail if exists
        a        Append mode, file must exist

        If `overwrite` is True then the previous data will be
        re-initialized upon this constructor being called.

        """
        assert mode in ['r', 'w', 'x', 'a'], "mode must be either r, w, x or a"

        self._filename = filename

        # open the file
        h5 = h5py.File(filename, mode)
        self._h5 = h5
        self.closed = False

        if mode in ['w', 'x'] and overwrite:
            self._runs = self._h5.create_group('runs')
            # this keeps track of the number of runs. The current
            # value will be the name of the next run that is added,
            # and this should be incremented when that happens
            self._run_idx_counter = 0
            if topology:
                self.topology = topology


    @property
    def filename(self):
        return self._filename

    def close(self):
        if not self.closed:
            self._h5.close()
            self.closed = True

    def __del__(self):
        self.close()

    @property
    def h5(self):
        return self._h5

    @property
    def runs(self):
        return self._h5['runs']

    @property
    def topology(self):
        return self._h5['topology']

    @topology.setter
    def topology(self, topology):
        self._h5.create_dataset('topology', data=topology)

    def new_run(self, **kwargs):
        # create a new group named the next integer in the counter
        run_grp = self._h5.create_group('runs/{}'.format(str(self._run_idx_counter)))
        # increment the counter
        self._run_idx_counter += 1

        # add metadata if given
        for key, val in kwargs.items():
            run_grp.attrs[key] = val

        return run_grp
