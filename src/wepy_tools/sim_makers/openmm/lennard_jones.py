import numpy as np
from scipy.spatial.distance import euclidean

from wepy.resampling.distances.distance import Distance

from openmmtools.testsystems import LennardJonesPair

from wepy_tools.sim_makers.openmm import OpenMMToolsTestSysSimMaker


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

class LennardJonesPairOpenMMSimMaker(OpenMMToolsTestSysSimMaker):

    def __init__(self):

        test_sys = LennardJonesPair()

        init_state = self.make_state(test_sys.system, test_sys.positions)

        super().__init__(
            distance=PairDistance(),
            init_state=init_state,
            system=test_sys.system,
            topology=test_sys.topology,
        )

