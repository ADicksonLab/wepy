from wepy.resampling.wexplore1 import WExplore1Resampler

from wepy.resampling.distances.randomwalk import RandomWalkDistance
from wepy.resampling.scoring.scorer import AllToAllScorer

from randomwalk_profiler import RandomwalkProfiler


if __name__=="__main__":

    # set up  the distance function
    distance = RandomWalkDistance();


    #set up the WExplore1 Resampler with the parameters
    resampler = WExplore1Resampler(pmax=0.1, pmin=10e-100,
                                       max_region_sizes=[16, 4, 1, 0.25], distance=distance)

    # set up a RandomWalkProfilier
    rw_profiler = RandomwalkProfiler(resampler)
    rw_profiler.run_test(num_walkers=200, num_cycles=100, dimension=5, debug_prints=True)
