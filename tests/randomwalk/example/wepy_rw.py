import os
import sys
import time

import numpy as np
import h5py
import scoop.futures
import pandas as pd
import mdtraj as mdj
import json

from wepy.resampling.wexplore2 import WExplore2Resampler
from wepy.distance_functions.randomwalk_distances import RandomWalkDistance
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.sim_manager import Manager
from wepy.openmm import UNITS
from tests.randomwalk.randomwalk import RandomWalker, RandomWalkRunner, State

if __name__ == "__main__":


    #initialize RandomWalkers with 200 replicas and dimension 5 and probability of 0.25
    num_walkers = 200
    probability = 0.25
    dimension = 5

    # set up initial state for walkers
    positions = np.zeros((1, dimension))

    init_state = State(positions, dimension)


    # create list of init_walkers
    initial_weight = 1/num_walkers
    init_walkers = [RandomWalker(init_state, initial_weight) for i in range(num_walkers)]

    # set up  the distance function
    distance_function = RandomWalkDistance();

    # set up the WExplore2 Resampler with the parameters
    resampler = WExplore2Resampler(pmax=0.1,
                                   distance_function=distance_function)

    # set up raunner for system
    runner = RandomWalkRunner(dimension=dimension, probability=probability)

    # instantiate a wexplore2 unbindingboudaryconditiobs
    units = {}
    for key, value in dict(UNITS).items():
        try:
            unit_name = value.get_name()
        except AttributeError:
            print("not a unit")
            unit_name = False

        if unit_name:
            units[key] = unit_name

    n_cycles = 1000
    segment_length = 10
    debug_prints = True

    walkers = init_walkers
    report_path = 'wepy_rw_results.h5'

    # load a json string of the topology
    with open("randomwalk_system.top.json", mode='r') as rf:
        randomwalk_system_top_json = rf.read()


    reporter = WepyHDF5Reporter(report_path, mode='w',
                                save_fields=['positions', 'weights'],
                                decisions=resampler.DECISION,
                                instruction_dtypes=resampler.INSTRUCTION_DTYPES,
                                resampling_aux_dtypes=None,
                                resampling_aux_shapes=None,
                                sparse_fields=None,
                                topology=randomwalk_system_top_json,
                                units=units,
                                n_dims=dimension)

    # running the simulation
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=map,
                          reporters=[reporter])


    # run a simulation with the manager for n_steps cycles of length 1000 each
    steps = [ segment_length for i in range(n_cycles)]

    print("Running simulation")
    # measuring the simulation time
    start_time = time.time()

    # Run the simulation
    sim_manager.run_simulation(n_cycles, steps, debug_prints=True)

    print("--- %s seconds ---" % (time.time() - start_time))
