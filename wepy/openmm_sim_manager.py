import sys
from wepy.sim_manager import Manager
class OpenmmManager(Manager):
        def __init__(self, init_walkers, num_workers,
                 runner = NoRunner(), 
                 resampler = NoResampler(),
                 work_mapper = map):
            super.__init__(init_walkers, num_workers,
                           runner = NoRunner(), 
                           resampler = NoResampler(),
                           work_mapper = map):

