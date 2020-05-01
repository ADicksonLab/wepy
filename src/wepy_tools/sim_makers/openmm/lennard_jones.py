import numpy as np
from scipy.spatial.distance import euclidean

from wepy.runners.openmm import GET_STATE_KWARG_DEFAULTS
from wepy.resampling.distances.distance import Distance
from wepy.boundary_conditions.receptor import UnbindingBC

from openmm_systems.test_systems import LennardJonesPair

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


# class PairUnbinding(BoundaryCondition):

#     pass

class LennardJonesPairOpenMMSimMaker(OpenMMToolsTestSysSimMaker):

    TEST_SYS = LennardJonesPair

    BCS = OpenMMToolsTestSysSimMaker.BCS + [UnbindingBC]

    LIGAND_IDXS = [0]
    RECEPTOR_IDXS = [1]

    UNBINDING_BC_DEFAULTS = {
        'cutoff_distance' : 1.0, # nm
        'periodic' : False,
    }

    DEFAULT_BC_PARAMS = OpenMMToolsTestSysSimMaker.DEFAULT_BC_PARAMS
    DEFAULT_BC_PARAMS.update(
        {
            'UnbindingBC' : UNBINDING_BC_DEFAULTS,
        }
    )

    def make_bc(self, bc_class, bc_params):

        if bc_class == UnbindingBC:
            bc_params.update(
                {
                    'distance' : self.distance,
                    'initial_state' : self.init_state,
                    'topology' : self.json_top(),
                    'ligand_idxs' : self.LIGAND_IDXS,
                    'receptor_idxs' : self.RECEPTOR_IDXS,
                }
            )

        bc = bc_class(**bc_params)

        return bc

    def __init__(self):

        # must set this here since we need it to generate the state,
        # will get called again in the superclass method
        self.getState_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
        if self.GET_STATE_KWARGS is not None:
            self.getState_kwargs.update(self.GET_STATE_KWARGS)

        test_sys = LennardJonesPair()

        init_state = self.make_state(test_sys.system, test_sys.positions)

        super().__init__(
            distance=PairDistance(),
            init_state=init_state,
            system=test_sys.system,
            topology=test_sys.topology,
        )

