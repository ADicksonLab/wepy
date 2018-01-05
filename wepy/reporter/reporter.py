import os
import os.path as osp
import pickle

class Reporter(object):

    def __init__(self):
        pass

    def init(self):
        pass

    def report(self, *args, **kwargs):
        pass

    def cleanup(self):
        pass

class FileReporter(Reporter):

    def __init__(self, file_path, mode='x'):
        self.file_path = file_path
        self.mode = mode


class WalkersPickleReporter(Reporter):

    def __init__(self, save_dir='./', freq=100, num_backups=2):
        # the directory to save the pickles in
        self.save_dir = save_dir
        # the frequency of cycles to backup the walkers as a pickle
        self.backup_freq = freq
        # the number of sets of walker pickles to keep, this will keep
        # the last `num_backups`
        self.num_backups = num_backups

    def init(self):
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
