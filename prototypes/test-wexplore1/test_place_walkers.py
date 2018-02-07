import numpy as np

from wepy.resampling.wexplore1 import WExplore1Resampler
from wepy.resampling.distances.distance import Distance
from wepy.runners.randomwalk import RandomWalker, RandomWalkState

class ScalarDistance(Distance):

    def __init__(self):
        pass

    def distance(self, a_state, b_state):
        return np.abs(b_state['positions'][0] - a_state['positions'][0])

seed = 3
regions = (2, 2, 2)
region_sizes = (1.0, 1.0, 1.0)

distance = ScalarDistance()
resampler = WExplore1Resampler(seed=seed, distance=distance,
                      max_n_regions=regions,
                      max_region_sizes=region_sizes)

init_states = [RandomWalkState(np.array([i]), 0.0) for i in range(5)]
init_walkers = [RandomWalker(state, 0.1) for state in init_states]

import ipdb; ipdb.set_trace()
assignments = resampler.place_walkers(init_walkers)
