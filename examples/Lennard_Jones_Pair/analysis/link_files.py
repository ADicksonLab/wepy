"""Example to show how to link runs from multiple files into a single
WepyHDF5 using the external link feature of HDF5.

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

# we make another WepyHDF5 that will contain both as external links
all_wepy_h5 = WepyHDF5(all_results_file, mode='x')
with all_wepy_h5:

    # add the runs in the order you want the run idxs as links
    run1_grp = all_wepy_h5.link_run(file1, 0)
    run2_grp = all_wepy_h5.link_run(file2, 0)

