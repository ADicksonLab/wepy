"""This is a exmaple of conventional random walke simultion. There is
no resampling after the random walk dynamics.

"""
import sys
import os
import os.path as osp
from pathlib import Path

import numpy as np

from wepy.resampling.resamplers.resampler import NoResampler
from wepy.resampling.distances.randomwalk import RandomWalkDistance
from wepy.runners.randomwalk import RandomWalkRunner, UNIT_NAMES
from wepy.walker import Walker, WalkerState

from wepy_tools.sim_makers.toys.randomwalk import RandomwalkProfiler



SAVE_FIELDS = ('positions')
# Name of field's unit in the HDF5
UNITS = UNIT_NAMES

outputs_dir = Path('_output')

if not osp.exists(outputs_dir):
    os.makedirs(outputs_dir)

# sets the input paths
hdf5_filename = 'rw_results.wepy.h5'
reporter_filename = 'randomwalk_conventional.org'

hdf5_path= outputs_dir / hdf5_filename
reporter_path = outputs_dir / reporter_filename



if __name__=="__main__":
    if sys.argv[1] == "--help" or sys.argv[1] == '-h':
        print("arguments: n_cycles, n_walkers, dimension")
    else:
        n_runs = int(sys.argv[1])
        n_cycles = int(sys.argv[2])
        n_walkers = int(sys.argv[3])
        dimension = int(sys.argv[4])

    # set up  the distance function
    distance = RandomWalkDistance();



    # set up the NOResampler
    resampler = NoResampler()

    # set up a RandomWalkProfilier
    rw_profiler = RandomwalkProfiler(resampler,
                                     dimension,
                                     hdf5_filename=str(hdf5_path),
                                     reporter_filename=str(reporter_path))

    # runs the simulations and gets the result
    rw_profiler.run(
        num_runs=n_runs,
        num_cycles=n_cycles,
        num_walkers=n_walkers,
    )
