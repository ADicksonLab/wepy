from scoop import futures

from wepy.sim_manager import Manager
from wepy.runner import NoRunner
from wepy.resampling import NoResampler

num_walkers = 8
num_workers = 8
manager = Manager(NoRunner, num_walkers, num_workers,
                  decision_model=NoCloneMerge,
                  work_mapper=futures.map)

num_cycles = 5
segment_lengths = [0 for i in range(num_cycles)]
walker_history, resampling_record = manager.run_simulation(num_cycles,
                                                           segment_lengths)
