import os
import sys

import numpy as np
import h5py
import scoop.futures

from wepy.resampling.wexplore2 import WExplore2Resampler
from wepy.distance_functions.openmm_distance import OpenMMDistance
from wepy.randomwalk import RandomWalker, RandomWalkRunner, State
from wepy.distance_functions.randomwalk_distance import RandomWalkDistance
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.sim_manager import Manager



# define a function for saving datat in hdf5 file
def save_data(file_handler, dimension, cycle_idx, resampled_walkers, resampling_records):
    
     for walker_idx, walker in enumerate(resampled_walkers):
         file_handler.create_dataset('d_{:0>5}/cycle_{:0>5}/walker_{:0>5}/positions'.
                                     format(dimension, cycle_idx, walker_idx),
                                     data = walker.positions)
         file_handler.create_dataset('d_{:0>5}/cycle_{:0>5}/walker_{:0>5}/weight'.
                                     format(dimension, cycle_idx, walker_idx), data=walker.weight)
                     

if __name__ == "__main__":

    # run this example with -m scoop , without scoop your program does not work    
    #initialize RandomWalkers
    num_walkers = 48
    dimension = 8

    # set up initial state for walkers
    positions = np.zeros((dimension))
    time = 0
    init_state = State(positions, dimension)
    

    # create list of init_walkers
    initial_weight = 1/num_walkers
    init_walkers = [RandomWalker(init_state, initial_weight) for i in range(num_walkers)]

    # set up distance function
    distance_function = RandomWalkDistance()

    # set up the WExplore2 Resampler with the parameters
    resampler = WExplore2Resampler(pmax=0.2,
                                   # algorithm parameters
                                   distance_function=distance_function)
    # set up raunner for system
    runner = RandomWalkRunner(dimension=dimension ,probability=0.25)
   
    
    # makes ref_traj and selects lingand_atom and protein atom  indices

    # instantiate a wexplore2 unbindingboudaryconditiobs

    n_cycles = 3
    segment_length = 10000
    debug_prints = True

    walkers = init_walkers
    report_path = 'wepy_results'
    reporter = WepyHDF5Reporter(report_path, mode='w', 
                                decisions=resampler.DECISION,
                                instruction_dtypes=resampler.INSTRUCTION_DTYPES,
                                resampling_aux_dtypes=None,
                                resampling_aux_shapes=None,
                                topology='{}',
                                n_dims=dimension)


    # running the simulation
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=scoop.futures.map,
                          reporter=reporter)
    n_steps = 10000
    n_cycles = 100

    # run a simulation with the manager for n_steps cycles of length 1000 each
    steps = [ n_steps for i in range(n_cycles)]
    print("Running simulation")
    sim_manager.run_simulation(n_cycles,
                               steps,
                               debug_prints=True)


    

             
    
    
    
    
 
