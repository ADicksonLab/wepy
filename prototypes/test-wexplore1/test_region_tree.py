import numpy as np

from wepy.resampling.distances.distance import Distance


class ScalarDistance(Distance):

    def __init__(self):
        pass

    def preimage_distance(self, a_state, b_state):
        return np.abs(b_state['positions'][0] - a_state['positions'][0])


if __name__ == "__main__":
    from wepy.resampling.wexplore1 import RegionTree
    from wepy.runners.randomwalk import RandomWalkState, RandomWalker

    MAX_N_REGIONS = (4, 4, 4, 4)
    MAX_REGION_SIZE = (4.0, 3.0, 2.0, 1.0)

    distance = ScalarDistance()

    n_walkers = 10

    init_walkers = [RandomWalker(RandomWalkState(np.array([i]), 0.0), 1.0/n_walkers)
                    for i in range(n_walkers)]

    region_tree = RegionTree(init_walkers[0].state,
                             max_n_regions=MAX_N_REGIONS,
                             max_region_size=MAX_REGION_SIZE,
                             distance=distance)

    # place the states in the tree making new branches if necessary
    region_tree.place_walkers(init_walkers)

    # balance the tree
    merge_groups, walkers_num_clones = region_tree.balance_tree()
