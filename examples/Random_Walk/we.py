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
def save_data(file_handler,cycle_idx, resampled_walkers, resampling_records):
    
     for walker_idx, walker in enumerate(resampled_walkers):
         file_handler.create_dataset('cycle_{:0>5}/walker_{:0>5}/positions'.format(cycle_idx,walker_idx),
                                     data = walker.positions)
         file_handler.create_dataset('cycle_{:0>5}/walker_{:0>5}/weight'.format(cycle_idx,walker_idx), data=walker.weight)
                     

if __name__ == "__main__":

    # run this example with -m scoop , without scoop your program does not work    
    #initialize RandomWalkers
    num_walkers = 8
    dimension = 5

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
    # the main cycle loop
    if debug_prints:
        result_template_str = "|".join(["{:^10}" for i in range(num_walkers + 1)])
        sys.stdout.write("Starting simulation\n")


    # create a hdf5 file to write
    h5file_handler = h5py.File(os.getcwd()+'/wepy_results.h5',mode='w')
    for cycle_idx in range(n_cycles):

        if debug_prints:
            sys.stdout.write("Begin cycle {}\n".format(cycle_idx))

        #run the segment
        
        new_walkers = list(map(runner.run_segment,
                                             walkers,
                                             (segment_length for i in range(num_walkers)
                                             )
                        ))

        if debug_prints:
            sys.stdout.write("End cycle {}\n".format(cycle_idx))

            # apply rules of boundary conditions and warp walkers through space
        # warped_walkers, warped_walker_records, warp_aux_data = \
        #                 self.boundary_conditions.warp_walkers(new_walkers,
        #                                                       debug_prints=debug_prints)

            # resample walkers
        resampled_walkers, resampling_records, resampling_aux_data = resampler.resample(new_walkers,
                                                                                       debug_prints=debug_prints)
        
        if debug_prints:
            # print results for this cycle
            print("Net state of walkers after resampling:")
            print("--------------------------------------")
            # slots
            slot_str = result_template_str.format("slot",
                                                  *[i for i in range(len(resampled_walkers))])
            print(slot_str)
            # states
            walker_state_str = result_template_str.format("state",
                *[str(walker.state) for walker in resampled_walkers])
            print(walker_state_str)
            # weights
            walker_weight_str = result_template_str.format("weight",
                *[str(walker.weight) for walker in resampled_walkers])
            print(walker_weight_str)
            #print warped_walker_records

        # report results to the reporter

        # prepare resampled walkers for running new state changes
        walkers = resampled_walkers




    




    
    # # instantiate a reporter for HDF5
    # report_path = 'wepy_results.h5'
    # reporter = WepyHDF5Reporter(report_path, mode='w',
    #                             decisions=resampler.DECISION,
    #                             instruction_dtypes=resampler.INSTRUCTION_DTYPES,
    #                             resampling_aux_dtypes=None,
    #                             resampling_aux_shapes=None,
    #                              )
    # # Instantiate a simulation manager
    # sim_manager = Manager(init_walkers,
    #                       runner=runner,
    #                       resampler=resampler,
    #                       boundary_conditions=None,
    #                       work_mapper=map,
    #                       reporter= reporter)
    # n_steps = 100
    # n_cycles = 2

    # # run a simulation with the manager for n_steps cycles of length 1000 each
    # steps = [ n_steps for i in range(n_cycles)]
    # print("Running simulation")
    # sim_manager.run_simulation(n_cycles,

    #                            steps,
    #                            debug_prints=True)

    # # your data should be in the 'wepy_results.h5'
    
    
    
    
 
