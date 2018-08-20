import sys
import numpy  as np

from wepy.resampling.resamplers.wexplore import WExploreResampler

from wepy.resampling.distances.randomwalk import RandomWalkDistance
from wepy.walker import Walker, WalkerState
from randomwalk_profiler import RandomwalkProfiler


#Maximum weight allowed for a walker
PMAX = 0.1
# the minimum weight allowed for a walker
PMIN = 10e-100

# the maximum number of regions allowed under each parent region
MAX_N_REGIONS = (10, 10, 10, 10)

# the maximum size of regions, new regions will be created if a walker
# is beyond this distance from each voronoi image unless there is an
# already maximal number of regions
MAX_REGION_SIZES = (16, 4, 1, .25) # nanometers

if __name__=="__main__":
    if sys.argv[1] == "--help" or sys.argv[1] == '-h':
        print("arguments: n_cycles, n_walkers, dimension")
    else:

        num_cycles = int(sys.argv[1])
        num_walkers = int(sys.argv[2])
        dimension =  int(sys.argv[3])
        h5_path = str(sys.argv[4])
        # set up  the distance function
        rw_distance = RandomWalkDistance();

        # set up initial state for walkers
        positions = np.zeros((1, dimension))

        init_state = WalkerState(positions=positions, time=0.0)

        #set up the Wexplore Resampler with the parameters
        resampler = WExploreResampler(distance=rw_distance,
                                                 init_state=init_state,
                                                 max_n_regions=MAX_N_REGIONS,
                                                 max_region_sizes=MAX_REGION_SIZES,
                                                 pmin=PMIN, pmax=PMAX,
                                      debug_mode=True)

        # set up a RandomWalkProfilier
        rw_profiler = RandomwalkProfiler(resampler,
                                         hdf5_reporter_path=h5_path)

        rw_profiler.run_test(num_walkers=num_walkers,
                             num_cycles=num_cycles,
                             dimension=dimension)
