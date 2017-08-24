import sys
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import h5py 

from wepy.sim_manager import Manager
from wepy.resampling.resampler import NoResampler
from wepy.runner import NoRunner
from wepy.boundary import NoBC


class OpenmmManager(Manager):
    def __init__(self, init_walkers, num_workers,
                 runner = NoRunner(), 
                 resampler = NoResampler(),
                 ubc = NoBC(),
                 work_mapper = map):
        super().__init__(init_walkers, num_workers,
                        runner, 
                        resampler,
                        work_mapper)
        self.ubc = ubc 

    def run_simulation(self, n_cycles, segment_lengths, debug_prints=False):
        """Run a simulation for a given number of cycles with specified
        lengths of MD segments in between.

        Can either return results in memory or write to a file.
        """


        if debug_prints:
            sys.stdout.write("Starting simulation\n")
        walkers = self.init_walkers

        resampling_handler = pd.HDFStore(os.getcwd()+'/resampling_records.h5',mode='w')
        walker_handler = h5py.File(os.getcwd()+'/walkers_records.h5',mode='w')
        dist_handler = h5py.File(os.getcwd()+'/dist_records.h5',mode='w')
        #save initial state
        self.save_walker_records(walker_handler,-1, walkers)
        for cycle_idx in range(n_cycles):
            if debug_prints:
                sys.stdout.write("Begin cycle {}\n".format(cycle_idx))
                
            sys.stdout.write("Begin cycle {}\n".format(cycle_idx))    
            # run the segment

            walkers = self.run_segment(walkers, segment_lengths[cycle_idx],
                                           debug_prints=debug_prints)


            # calls wexplore2 ubinding boundary conditions
            if debug_prints:
                sys.stdout.write("Start  boundary Conditions")
            
            resampled_walkers, warped_walkers_idx = self.ubc.warp_walkers(walkers)
            
            # record changes in state of the walkers
            if debug_prints:
                sys.stdout.write("End  BoundaryConditions")
                print ('warped_walkers=',warped_walkers_idx)


            # resample based walkers
            resampled_walkers, cycle_resampling_records, distance_matrix, spreads = self.resampler.resample(resampled_walkers,
                                                                                                  debug_prints=debug_prints)
            self.save_dist_records(dist_handler,cycle_idx, distance_matrix, spreads)
            # save resampling records in a hdf5 file
            if debug_prints:
                sys.stdout.write("Start Resampling")
            
            self.save_resampling_records(resampling_handler, cycle_idx, cycle_resampling_records, warped_walkers_idx)
            if debug_prints:
                sys.stdout.write("End  Resampling")
            
            # prepare resampled walkers for running new state changes
            # save walkers positions in a hdf5 file
            
            self.save_walker_records(walker_handler, cycle_idx, resampled_walkers)
            walkers = resampled_walkers.copy()
            
        resampling_handler.close()
        walker_handler.close()
        dist_handler.close()
        if debug_prints:
            sys.stdout.write("End cycle {}\n".format(cycle_idx))
        
        
    
        if debug_prints:
            self.read_resampling_data()
#            self.read_walker_data()

    def mdtraj_positions(self, openmm_positions):
        
        n_atoms = len (openmm_positions)

        xyz = np.zeros(( n_atoms, 3))
        


        for i in range(n_atoms):
            xyz[i,:] = ([openmm_positions[i]._value[0], openmm_positions[i]._value[1],
                                                    openmm_positions[i]._value[2]])

        return xyz
         
    def save_resampling_records(self, hdf5_handler, cycle_idx, cycle_resampling_records, warped_walkers_idx):

        # save resampling records in table format in a hdf5 file
        DFResamplingRecord = namedtuple("DFResamplingRecord", ['cycle_idx', 'step_idx', 'walker_idx',
                                                               'decision','instruction','warped_walker'])
        df_recs = []
        
        for step_idx, step in enumerate(cycle_resampling_records):
            for walker_idx, rec in enumerate(step):
                
                if  walker_idx  in warped_walkers_idx:
                    decision = True
                else:
                    decision = False
                df_rec = DFResamplingRecord(cycle_idx=cycle_idx,
                                            step_idx=step_idx,
                                            walker_idx=walker_idx,
                                            decision=rec.decision.name,
                                            instruction = rec.value,
                                            warped_walker = decision)
                df_recs.append(df_rec)

        resampling_df = pd.DataFrame(df_recs)
 
        hdf5_handler.put('cycle_{:0>5}'.format(cycle_idx), resampling_df, data_columns= True)
        hdf5_handler.flush(fsync=True)
                         
                     
        
      
    def save_walker_records(self, walker_handler, cycle_idx, resampled_walkers):

        walker_handler.create_dataset('cycle_{:0>5}/time'.format(cycle_idx), data=resampled_walkers[0].time._value)        
        for walker_idx, walker in enumerate(resampled_walkers):
            walker_handler.create_dataset('cycle_{:0>5}/walker_{:0>5}/positions'.format(cycle_idx,walker_idx),
                                          data = self.mdtraj_positions(walker.positions))
            box_vector = np.array(((walker.box_vectors._value[0],walker.box_vectors._value[1],walker.box_vectors._value[2])))
            walker_handler.create_dataset('cycle_{:0>5}/walker_{:0>5}/box_vectors'.format(cycle_idx,walker_idx), data=box_vector)
            walker_handler.create_dataset('cycle_{:0>5}/walker_{:0>5}/weight'.format(cycle_idx,walker_idx), data=walker.weight)
            
            
            walker_handler.flush()
                          
                            
    def read_resampling_data(self,):

        hdf = pd.HDFStore(os.getcwd()+'/resampling_records.h5',
                            mode ='r')
        keys = list (hdf.keys())
        for key in keys:
            df = hdf.get(key)
            print (df)
        hdf.close()
        
    def save_dist_records(self, dist_handler,cycle_idx, distance_matrix, spreads):
        dist_handler.create_dataset('cycle_{:0>5}/dist_matrix'.format(cycle_idx), data=distance_matrix)
        dist_handler.create_dataset('cycle_{:0>5}/spreads'.format(cycle_idx), data=np.array(spreads))        
        
        
        
        

 
