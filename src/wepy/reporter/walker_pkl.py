# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import os
import os.path as osp
import pickle

# First Party Library
from wepy.reporter.reporter import Reporter


class WalkerPklReporter(Reporter):
    def __init__(self, save_dir="./", freq=100, num_backups=2):
        # the directory in which to save the pickles
        self.save_dir = save_dir
        # the frequency of cycles to backup the walkers as a pickle
        self.backup_freq = freq
        # the number of most recent walker pickles to keep, this will remove the rest
        self.num_backups = num_backups

    def init(self, *args, **kwargs):
        # make sure the save_dir exists
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def report(self, cycle_idx=None, new_walkers=None, **kwargs):
        # total number of cycles completed
        n_cycles = cycle_idx + 1
        # if the cycle is on the frequency backup walkers to a pickle
        if n_cycles % self.backup_freq == 0:
            pkl_name = "walkers_cycle_{}.pkl".format(cycle_idx)
            pkl_path = osp.join(self.save_dir, pkl_name)
            with open(pkl_path, "wb") as wf:
                pickle.dump(new_walkers, wf)
            # remove old pickles if we have more than the num_backups
            if (cycle_idx // self.backup_freq) >= self.num_backups:
                old_idx = cycle_idx - self.num_backups * self.backup_freq
                old_pkl_fname = "walkers_cycle_{}.pkl".format(old_idx)
                os.remove(osp.join(self.save_dir, old_pkl_fname))
