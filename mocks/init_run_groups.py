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

import os
import os.path as osp
import pickle
from copy import copy

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from openmmtools.testsystems import LennardJonesPair

from wepy.runners.openmm import OpenMMRunner, OpenMMState
from wepy.runners.openmm import UNIT_NAMES, GET_STATE_KWARG_DEFAULTS



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
    MAX_REGION_SIZES = (0.6, 0.4, 0.2, 0.1) # nanometers
    resampler = WExplore1Resampler(distance=PairDistance(),
                               max_region_sizes=MAX_REGION_SIZES,
                               max_n_regions=MAX_N_REGIONS,
                               pmin=PMIN, pmax=PMAX)

    # make the test system from openmmtools
    test_sys = LennardJonesPair()

    # integrator
    TEMPERATURE= 300.0*unit.kelvin
    FRICTION_COEFFICIENT = 1/unit.picosecond
    STEP_SIZE = 0.002*unit.picoseconds
    integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

    # the mdtraj here is needed for unbininding BC
    mdtraj_topology = mdj.Topology.from_openmm(test_sys.topology)

    # the initial state for the BC
    context = omm.Context(test_sys.system, copy(integrator))
    context.setPositions(test_sys.positions)
    get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
    init_sim_state = context.getState(**get_state_kwargs)
    init_state = OpenMMState(init_sim_state)

    # initialize the unbinding boundary conditions
    ubc = UnbindingBC(cutoff_distance=1.0,
                      initial_state=init_state,
                      topology=mdtraj_topology,
                      ligand_idxs=np.array(test_sys.ligand_indices),
                      binding_site_idxs=np.array(test_sys.receptor_indices))

    # make some states so that they will make new branches when
    # assigned to the resampler's region tree
    init_walkers = []
    init_positions = []
    for i in range(4):

        zero_pos = [0., 0., 0.]
        positions = np.array([zero_pos, [2*i + 1.0, 0., 0.]]) * test_sys.positions.unit

        init_positions.append(positions)

        # make a context and set the positions
        context = omm.Context(test_sys.system, copy(integrator))
        context.setPositions(positions)

        # get the data from this context so we have a state to start the
        # simulation with
        get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
        sim_state = context.getState(**get_state_kwargs)
        state = OpenMMState(sim_state)
        walker = Walker(state, 1.0)

        init_walkers.append(walker)

    # make some records for resampling that show resampler records as well.
    resampled_walkers, resampling_data, resampler_data = resampler.resample(init_walkers)

    # make some unbinding BC records
    

    # report these
    
