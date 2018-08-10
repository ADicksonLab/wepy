from wepy.resampling.resamplers.revo import REVOResampler

from wepy.resampling.distances.randomwalk import RandomWalkDistance
from wepy.resampling.scoring.scorer import AllToAllScorer

from randomwalk_profiler import RandomwalkProfiler


if __name__=="__main__":

    # set up  the distance function
    distance = RandomWalkDistance();
    scorer = AllToAllScorer(distance=distance)

    # set up the Revo Resampler with the parameters
    resampler = REVOResampler(pmax=0.1, pmin=10e-100,
                                        scorer=scorer)


    # set up a RandomWalkProfilier
    rw_profiler = RandomwalkProfiler(resampler)
    rw_profiler.run_test(num_walkers=200, num_cycles=100, dimension=5, debug_prints=True)
