"""This is a exmaple of random walke simultion using the WExplore
resampler.

"""
import sys
import os
import os.path as osp
from pathlib import Path

import numpy as np

from wepy.resampling.resamplers.wexplore import WExploreResampler
from wepy.resampling.distances.randomwalk import RandomWalkDistance
from wepy.runners.randomwalk import RandomWalkRunner, UNIT_NAMES
from wepy.walker import Walker, WalkerState

from wepy_tools.sim_makers.toys.randomwalk import RandomwalkProfiler


ON = True
OFF = False
# the maximum weight allowed for a walker
PMAX = 0.1
# the minimum weight allowed for a walker
PMIN = 1e-100

# set the value of distance exponent
DIST_EXPONENT = 4
# the merge distance value
MERGE_DIST = 2.5
# field in the HDF5
SAVE_FIELDS = ('positions')
# Name of field's unit in the HDF5
UNITS = UNIT_NAMES
PROBABILITY=0.25
WEIGHTS=ON

# the maximum number of regions allowed under each parent region
MAX_N_REGIONS = (10, 10, 10, 10)

# the maximum size of regions, new regions will be created if a walker
# is beyond this distance from each voronoi image unless there is an
# already maximal number of regions
MAX_REGION_SIZES = (16, 4, 1, .25)

outputs_dir = Path('_output')
if not osp.exists(outputs_dir):
    os.makedirs(outputs_dir)


# sets the input paths
hdf5_filename = 'rw_results.wepy.h5'
reporter_filename = 'randomwalk_wexplore.org'

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


    dimension = 5

    # set up initial state for walkers
    position_coords = np.zeros((1, dimension))

    init_state = WalkerState(positions=position_coords, time=0.0)

    # set up  the distance function
    distance = RandomWalkDistance();



    # set up the WExplore Resampler with the parameters
    resampler = WExploreResampler(distance=distance,
                                  init_state=init_state,
                                  max_n_regions=MAX_N_REGIONS,
                                  max_region_sizes=MAX_REGION_SIZES,
                                  pmin=PMIN, pmax=PMAX)


    # set up a RandomWalkProfilier
    rw_profiler = RandomwalkProfiler(resampler,
                                     dimension,
                                     hdf5_filename=str(hdf5_path),
                                     reporter_filename=str(reporter_path))

    # runs the simulations and gets the result
    rw_profiler.run(num_runs=n_runs, num_cycles=n_cycles,
                    num_walkers=n_walkers)

        #set up the Wexplore Resampler with the parameters
