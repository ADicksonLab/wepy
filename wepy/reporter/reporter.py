import os
import os.path as osp
import pickle

class Reporter(object):

    def __init__(self):
        pass

    def init(self, *args, **kwargs):
        pass

    def report(self, *args, **kwargs):
        pass

    def cleanup(self, *args, **kwargs):
        pass

class FileReporter(Reporter):

    def __init__(self, file_path, mode='x'):
        self.file_path = file_path
        self.mode = mode

class ProgressiveFileReporter(FileReporter):
    """Super class for a reporter that will successively overwrite the
    same file over and over again. The base FileReporter really only
    supports creation of file one time.

    """

    def __init__(self, file_path, mode='x'):

        super().__init__(file_path, mode=mode)

    def init(self, *args, **kwargs):

        # because we want to overwrite the file at every cycle we
        # need to change the mode to write with truncate. This allows
        # the file to first be opened in 'x' or 'w-' and check whether
        # the file already exists (say from another run), and warn the
        # user. However, once the file has been created for this run
        # we need to overwrite it many times forcefully.

        # so if the mode is 'x' or 'w-' we check to make sure the file
        # doesn't exist
        if self.mode in ['x', 'w-']:
            if osp.exists(self.file_path):
                raise FileExistsError("File exists: '{}'".format(self.file_path))

        # now that we have checked if the file exists we set it into
        # overwrite mode
        self.mode = 'w'


class WalkersPickleReporter(Reporter):

    def __init__(self, save_dir='./', freq=100, num_backups=2):
        # the directory to save the pickles in
        self.save_dir = save_dir
        # the frequency of cycles to backup the walkers as a pickle
        self.backup_freq = freq
        # the number of sets of walker pickles to keep, this will keep
        # the last `num_backups`
        self.num_backups = num_backups

    def init(self, *args, **kwargs):
        # make sure the save_dir exists
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # delete backup pickles in the save_dir if they exist
        else:
            for pkl_fname in os.listdir(self.save_dir):
                os.remove(osp.join(self.save_dir, pkl_fname))

    def report(self, cycle_idx, walkers,
               *args, **kwargs):

        # ignore all args and kwargs

        # total number of cycles completed
        n_cycles = cycle_idx + 1
        # if the cycle is on the frequency backup walkers to a pickle
        if n_cycles % self.backup_freq == 0:

            pkl_name = "walkers_cycle_{}.pkl".format(cycle_idx)
            pkl_path = osp.join(self.save_dir, pkl_name)
            with open(pkl_path, 'wb') as wf:
                pickle.dump(walkers, wf)

            # remove old pickles if we have more than the `num_backups`
            if self.num_backups is not None:
                if (cycle_idx // self.backup_freq) >= self.num_backups:
                    old_idx = cycle_idx - self.num_backups * self.backup_freq
                    old_pkl_fname = "walkers_cycle_{}.pkl".format(old_idx)
                    os.remove(osp.join(self.save_dir, old_pkl_fname))
