"""This is a exmaple of random walke simultion using the REVO
resampler.
"""
import sys
import os
import os.path as osp
from pathlib import Path

import numpy as np

from wepy.resampling.resamplers.revo import REVOResampler

from wepy.resampling.distances.randomwalk import RandomWalkDistance
from wepy.runners.randomwalk import RandomWalkRunner, UNIT_NAMES
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.mapper import Mapper

from wepy.sim_manager import Manager
from wepy.walker import Walker, WalkerState

from wepy.runners.randomwalk import RandomWalkRunner

from wepy_tools.sim_makers.toys.randomwalk import RandomwalkProfiler


ON = True
OFF = False
# the maximum weight allowed for a walker
PMAX = 0.1
# the minimum weight allowed for a walker
PMIN = 1e-100

# set the value of distance exponent
DIST_EXPONENT = 4
# the merge distance value
MERGE_DIST = 2.5
# field in the HDF5
SAVE_FIELDS = ('positions')
# Name of units of fields in the HDF5
UNITS = UNIT_NAMES
PROBABILITY=0.25
WEIGHTS=ON


outputs_dir = Path('_output')

if not osp.exists(outputs_dir):
    os.makedirs(outputs_dir)

# sets the input paths
hdf5_filename = 'rw_results.wepy.h5'
reporter_filename = 'randomwalk_revo.org'

hdf5_path= outputs_dir / hdf5_filename
reporter_path = outputs_dir / reporter_filename


def get_char_distance(dimension, num_walkers):
    """Calculate the characteristic value.
    Runs one cycle simulation and calculates the characteristic
    distance value.

    Parameters
    ----------
    dimension: int
        The dimension of the random walk space.


    num_walkers: int
        The number of walkers.

    Returns
    -------
    characteristic distance : float
        The characteristic distance value.

    """
    # set up initial state for walkers
    positions = np.zeros((1, dimension))

    init_state = WalkerState(positions=positions, time=0.0)

    # set up  the distance function
    rw_distance = RandomWalkDistance();

    # set up the  REVO Resampler with the parameters
    resampler = REVOResampler(distance=rw_distance,
                              pmin=PMIN, pmax=PMAX,
                              init_state=init_state,
                              char_dist=1,
                              merge_dist=MERGE_DIST
                              )

    # create list of init_walkers
    initial_weight = 1/num_walkers

    init_walkers = [Walker(init_state, initial_weight)
                    for i in range(num_walkers)]

    # set up raunner for system
    runner = RandomWalkRunner(probability=PROBABILITY)

    n_steps = 10
    mapper = Mapper()
    # running the simulation
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=mapper)


    print("Running simulation")
    #runs for one cycle
    sim_manager.init(num_walkers)

    new_walkers = sim_manager.run_segment(init_walkers, n_steps, 0)

    dist_matrix, _ = resampler._all_to_all_distance(new_walkers)


    return np.average(dist_matrix)



if __name__=="__main__":

    if sys.argv[1] == "--help" or sys.argv[1] == '-h':
        print("arguments: n_cycles, n_walkers, dimension")
    else:

        n_runs = int(sys.argv[1])
        n_cycles = int(sys.argv[2])
        n_walkers = int(sys.argv[3])
        dimension = int(sys.argv[4])


    dimension = 5

    # set up initial state for walkers
    position_coords = np.zeros((1, dimension))

    init_state = WalkerState(positions=position_coords, time=0.0)

    # set up  the distance function
    distance = RandomWalkDistance();


    char_dist = get_char_distance(dimension, n_walkers)
    # set up the Revo Resampler with the parameters
    resampler = REVOResampler(distance=distance,
                              pmin=PMIN,
                              pmax=PMAX,
                              dist_exponent=DIST_EXPONENT,
                              init_state=init_state,
                              char_dist=char_dist,
                              merge_dist=MERGE_DIST,
                              weights=WEIGHTS)


    # set up a RandomWalkProfilier
    rw_profiler = RandomwalkProfiler(resampler,
                                     dimension,
                                     hdf5_filename=str(hdf5_path),
                                     reporter_filename=str(reporter_path))

    # runs the simulations and gets the result
    rw_profiler.run(num_runs=n_runs, num_cycles=n_cycles,
                    num_walkers=n_walkers)
