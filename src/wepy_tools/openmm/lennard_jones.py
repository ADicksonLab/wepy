import numpy as np
from scipy.spatial.distance import euclidean

from wepy.resampling.distances.distance import Distance

from openmmtools.testsystems import LennardJonesPair

from wepy_examples.sim_maker import OpenMMSimMaker


# make the test system from openmmtools
test_sys = LennardJonesPair()

# get the initial state
init_state = OpenMMSimMaker.make_state(test_sys.system, test_sys.positions)

## Distance Metric
# we define a simple distance metric for this system, assuming the
# positions are in a 'positions' field
class PairDistance(Distance):

    def __init__(self, metric=euclidean):
        self.metric = metric

    def image(self, state):
        return state['positions']

    def image_distance(self, image_a, image_b):
        dist_a = self.metric(image_a[0], image_a[1])
        dist_b = self.metric(image_b[0], image_b[1])

        return np.abs(dist_a - dist_b)

# make a distance object which can be used to compute the distance
# between two walkers, for our scorer class
distance = PairDistance()


### parametrized sims

sim_maker = OpenMMSimMaker(
    distance=distance,
    init_state=init_state,
    system=test_sys.system,
    topology=test_sys.topology,
)


# make one with defaults

n_walkers = 10
apparatus = sim_maker.make_apparatus()
config = sim_maker.make_configuration(work_mapper='TaskMapper', platform='OpenCL')

sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)


# TODO do one with reporters (not needed for benchmarking the mappers
# etc.)
