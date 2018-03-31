import json
import pickle

import numpy as np

from openmmtools.testsystems import LennardJonesPair

from wepy.hdf5 import WepyHDF5
from wepy.resampling.wexplore1 import WExplore1Resampler
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.resampling.distances.distance import Distance
from scipy.spatial.distance import euclidean
from wepy.walker import Walker

import mdtraj as mdj

with open('LJ_pair.top.json', 'r') as rf:
    top_json = rf.read()

hdf5_filename = 'tmp.wepy.h5'
wepy_h5 = WepyHDF5(hdf5_filename, mode='w', topology=top_json)
with wepy_h5:

    wepy_h5.new_run()

    wepy_h5.init_run_resampling(0, WExplore1Resampler)
    wepy_h5.init_run_resampler(0, WExplore1Resampler)

    wepy_h5.init_run_bc(0, UnbindingBC)
    wepy_h5.init_run_warping(0, UnbindingBC)
    wepy_h5.init_run_progress(0, UnbindingBC)


    # make a resampler and boundary condition to generate mock records
    class PairDistance(Distance):

        def __init__(self, metric=euclidean):
            self.metric = metric

        def image(self, state):
            return state['positions']

        def image_distance(self, image_a, image_b):
            dist_a = self.metric(image_a[0], image_a[1])
            dist_b = self.metric(image_b[0], image_b[1])

            return np.abs(dist_a - dist_b)

    PMAX = 0.5
    PMIN = 1e-12
    MAX_N_REGIONS = (10, 10, 10, 10)
    MAX_REGION_SIZES = (1, 0.5, .35, .25) # nanometers
    resampler = WExplore1Resampler(distance=PairDistance(),
                               max_region_sizes=MAX_REGION_SIZES,
                               max_n_regions=MAX_N_REGIONS,
                               pmin=PMIN, pmax=PMAX)

    test_sys = LennardJonesPair()
    # the mdtraj here is needed for the distance function
    mdtraj_topology = mdj.Topology.from_openmm(test_sys.topology)
    with open("LJ_init_state.pkl", 'rb') as rf:
        init_state = pickle.load(rf)

    # initialize the unbinding boundary conditions
    ubc = UnbindingBC(cutoff_distance=1.0,
                      initial_state=init_state,
                      topology=mdtraj_topology,
                      ligand_idxs=np.array(test_sys.ligand_indices),
                      binding_site_idxs=np.array(test_sys.receptor_indices))


    # make some records for resampling that show resampler records as well.
    walkers = [Walker(init_state, 1.0) for i in range(10)]
    resampled_walkers, resampling_data, resampler_data = resampler.resample(walkers)

