import sys
from collections import namedtuple

import numpy as np
import pandas as pd


from wepy.sim_manager import Manager
from wepy.resampling.resampler import NoResampler
from wepy.runner import NoRunner

class OpenmmManager(Manager):
    def __init__(self, init_walkers, num_workers,
                 runner = NoRunner(), 
                 resampler = NoResampler(),
                 work_mapper = map):
        super().__init__(init_walkers, num_workers,
                        runner, 
                        resampler,
                        work_mapper)

    def run_simulation(self, n_cycles, segment_lengths, debug_prints=False):
        """Run a simulation for a given number of cycles with specified
        lengths of MD segments in between.

        Can either return results in memory or write to a file.
        """

        if debug_prints:
            sys.stdout.write("Starting simulation\n")
        walkers = self.init_walkers

        hdf5_handler = pd.HDFStore('/mnt/home/nazanin/projects/wepy/examples/sEH_TPPU_NewMapper/resampling_records.h5')
        for cycle_idx in range(n_cycles):
            if debug_prints:
                sys.stdout.write("Begin cycle {}\n".format(cycle_idx))

            # run the segment
            new_walkers = self.run_segment(walkers, segment_lengths[cycle_idx],
                                           debug_prints=debug_prints)


            if debug_prints:
                sys.stdout.write("End cycle {}\n".format(cycle_idx))

            # record changes in state of the walkers
            #walker_records.append(new_walkers)

            # resample based walkers
            resampled_walkers, cycle_resampling_records = self.resampler.resample(new_walkers,debug_prints=debug_prints)
            # save resampling records in a hdf5 file
            self.save_resampling_records(hdf5_handler, cycle_idx, cycle_resampling_records)
            # prepare resampled walkers for running new state changes
            walkers = resampled_walkers
        hdf5_handler.close()    
        if debug_prints:
            self.read_resampling_data()

         
    def save_resampling_records(self, hdf5_handler, cycle_idx, cycle_resampling_records):


        DFResamplingRecord = namedtuple("DFResamplingRecord", ['cycle_idx', 'step_idx', 'walker_idx',
                                              'decision', 'instruction'])
        df_recs = []
        hdf = hdf5_handler
        for step_idx, step in enumerate(cycle_resampling_records):
            for walker_idx, rec in enumerate(step):
                df_rec = DFResamplingRecord(cycle_idx=cycle_idx,
                                            step_idx=step_idx,
                                            walker_idx=walker_idx,
                                            decision=rec.decision.name,
                                            instruction=rec.instruction)
                df_recs.append(df_rec)

        resampling_df = pd.DataFrame(df_recs)
 
        hdf.put('re_record{}'.format(cycle_idx), resampling_df, data_columns= True)
        # reding
              
    def read_resampling_data(self,):

        hdf = pd.HDFStore('/mnt/home/nazanin/projects/wepy/examples/sEH_TPPU_NewMapper/resampling_records.h5',
                            mode ='r')
        keys = list (hdf.keys())
        for key in keys:
            df = hdf.get(key)
            sys.stdout.write (df)

 
