import h5py


def load_dataset(path):
    return None

class TrajHDF5(object):

    def __init__(self, filename, mode='x', overwrite=True,
                 topology=None,
                 positions=None,
                 time=None,
                 box_vectors=None
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
    def topology(self):
        return self._h5['topology']

    @topology.setter
    def topology(self, topology):
        self._h5.create_dataset('topology', data=topology)



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
