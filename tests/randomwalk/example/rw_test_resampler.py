from wepy.distance_functions.randomwalk_distances import RandomWalkDistance
from tests.randomwalk.random_walk_profiler import RandomwalkProfiler

from wepy.resampling.wexplore2 import WExplore2Resampler



if __name__=="__main__":

    # set up  the distance function
    distance_function = RandomWalkDistance();

    # set up the WExplore2 Resampler with the parameters
    resampler = WExplore2Resampler(pmax=0.1, pmin=10e-100,
                                   distance_function=distance_function)
    # set up a RandomWalkProfilier
    rw_profiler = RandomwalkProfiler(resampler)
    rw_profiler.run_test(num_walkers=200, num_cycles=100, dimension=5, debug_prints=False)

