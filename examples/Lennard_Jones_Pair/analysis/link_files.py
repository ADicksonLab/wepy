"""Example to show how to link runs from multiple files into a single
WepyHDF5 using the external link feature of HDF5.

To use this you must first run `we.py` as well as a restart to another
file.

"""
from shutil import copy2
import os

import numpy as np

from wepy.hdf5 import WepyHDF5

file1 = '../outputs/results.wepy.h5'
file2 = '../outputs/results_copy.wepy.h5'

all_results_file = '../outputs/all_results.wepy.h5'

# make a copy of the result hdf5 file to use as a proxy for another
# run, first remove the copy so we can remake it
#os.remove(file2)
copy2(file1, file2)

# Load wepy hdf5 file into python script
wepy_1_h5 = WepyHDF5(file1, mode = 'r')
wepy_2_h5 = WepyHDF5(file2, mode = 'r')

# we make another WepyHDF5 that will contain both as external links,
# so we clone one of the ones we are linking from to get a WepyHDF5
# file with no runs in it, before it is opened
with wepy_1_h5:
    all_wepy_h5 = wepy_1_h5.clone(all_results_file, mode='w')

with all_wepy_h5:

    # link all the file1 runs together preserving continuations
    file_run_idxs = all_wepy_h5.link_file_runs(file1)

    # add the continuation run that is in another file
    run2_grp = all_wepy_h5.link_run(file2, 0, continues=)

