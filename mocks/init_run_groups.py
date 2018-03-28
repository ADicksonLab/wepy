import json

from wepy.hdf5 import WepyHDF5
from wepy.resampling.wexplore1 import WExplore1Resampler

with open('LJ_pair.top.json', 'r') as rf:
    top_json = rf.read()

hdf5_filename = 'tmp.wepy.h5'
wepy_h5 = WepyHDF5(hdf5_filename, mode='w', topology=top_json)
wepy_h5.open()

wepy_h5.init_run_resampling(0, WExplore1Resampler)

