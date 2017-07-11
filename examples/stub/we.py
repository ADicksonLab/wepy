from scoop import futures

from wepy.walker import Walker
from wepy.sim_manager import Manager
from wepy.runner import NoRunner
from wepy.resampling import NoResampler

# create walkers for your simulation with equal weight
n_walkers = 8
init_weight = 1.0 / n_walkers
init_walkers = []
for i in range(n_walkers):
    init_walkers.append(Walker(i, init_weight))

# create a simulation manager
manager = Manager(init_walkers,
                  
                  work_mapper=futures.map)

num_cycles = 5
segment_lengths = [0 for i in range(num_cycles)]
walker_history, resampling_record = manager.run_simulation(num_cycles,
                                                           segment_lengths)
