import os
import sys
import json

import numpy as np
import pandas as pd
import h5py
import scoop.futures

import mdtraj as mdj

from wepy.resampling.resamplers.resampler import NoResampler
from wepy.work_mapper.mapper import Mapper
from wepy.hdf5 import RunCycleSlice
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.sim_manager import Manager
from wepy.walker import Walker, WalkerState
from wepy.runners.randomwalk import RandomWalkRunner, UNIT_NAMES
from wepy.hdf5 import WepyHDF5

class RandomwalkProfiler(object):

    def __init__(self, resampler, move_prob=0.25, hdf5_reporter=None):
        self.resampler = resampler
        self.probability = move_prob
        self.hdf5_reporter = hdf5_reporter

        if self.hdf5_reporter is None:
            self.hdf5_reporter = "rw_results.h5"

    def generate_topology(self):
        n_atoms = 1
        data = []
        for i in range(n_atoms):
            data.append(dict(serial=i, name="H", element="H",
                             resSeq=i + 1, resName="UNK", chainID=0))

        data = pd.DataFrame(data)

        xyz = np.zeros((1, 1, 3))
        unitcell_lengths = 0.943 * np.ones((1, 3))
        unitcell_angles = 90 * np.ones((1, 3))

        top = mdj.Topology.from_dataframe(data, bonds=np.zeros((0, 2), dtype='int'))
        traj = mdj.Trajectory(xyz, top,
                              unitcell_lengths=unitcell_lengths,
                              unitcell_angles=unitcell_angles)
        traj.save_hdf5("tmp_mdtraj_system.h5")
        #traj.save_pdb("randowalk.pdb")
        # we need a JSON string for now in the topology section of the
        # HDF5 so we just load the topology from the hdf5 file
        top_h5 = h5py.File("tmp_mdtraj_system.h5")
        # it is in bytes so we need to decode to a string, which is in JSON format
        json_top_str = top_h5['topology'][0].decode()
        top_h5.close()
        os.remove("tmp_mdtraj_system.h5")

        return json_top_str

    def run_test(self, num_walkers=200, num_cycles=100, dimension=5,
                 debug_prints=False):

        hdf5_filepath = self._runtest(num_walkers, num_cycles,
                                      dimension, debug_prints=debug_prints)

        return self.analyse(hdf5_filepath, num_walkers)

    def _runtest(self, num_walkers, num_cycles, dimension, debug_prints=False):
        print("Random walk simulation with: ")
        print("Dimension =", dimension)
        print("Probability =", self.probability)
        print("Number of Walkers", num_walkers)
        print("Number of Cycles", num_cycles)

        # set up initial state for walkers
        positions = np.zeros((1, dimension))

        init_state = WalkerState(positions=positions, time=0.0)


        # create list of init_walkers
        initial_weight = 1/num_walkers
        init_walkers = []

        # init_walkers, n_cycles = get_final_state(path, num_walkers)
        init_walkers = [Walker(init_state, initial_weight) for i in range(num_walkers)]



        # set up raunner for system
        runner = RandomWalkRunner(dimension=dimension, probability=self.probability)

        units = dict(UNIT_NAMES)
        # instantiate a revo unbindingboudaryconditiobs
        segment_length = 10

        # set up the reporter
        report_path = self.hdf5_reporter
        randomwalk_system_top_json = self.generate_topology()
        reporter = WepyHDF5Reporter(report_path, mode='w',
                                    save_fields=['positions', 'weights'],
                                    decisions=self.resampler.DECISION.ENUM,
                                    instruction_dtypes=self.resampler.DECISION.instruction_dtypes(),
                                    topology=randomwalk_system_top_json,
                                    units=units,
                                    n_dims=dimension)


        # running the simulation
        sim_manager = Manager(init_walkers,
                              runner=runner,
                              resampler=self.resampler,
                              work_mapper=Mapper(),
                              reporters=[reporter])


        # run a simulation with the manager for n_steps cycles of length 1000 each
        steps = [segment_length for i in range(num_cycles)]
        print("Start simulation")

        sim_manager.run_simulation(num_cycles, steps, debug_prints=debug_prints)

        print("Finished Simulation")
        return report_path

    # implements kronecker_delta function
    def kronecker_delta(self, x):
        if x == 0:
            return 1
        else:
            return 0

    # Measure accuracy
    def accuracy(self, x, Px):

        if Px == 0:
            return 0
        elif np.log(Px) > 2* np.log(self.Pt(x)):
            return 1 + np.abs( np.log(self.Pt(x)) - np.log(Px)) / np.log(self.Pt(x))
        else :
            return 0

    # claculating target probability of position x
    def Pt(self, x):
        return (2/3) * np.power(1/3, x)


    def find_max_range(self, wepy_h5, num_walkers):

        max_ranges = []

        hd = wepy_h5.h5
        walker_ranges = []
        for walker_idx in range(num_walkers):
            max_ranges = np.amax(hd['runs/0/trajectories/{}/positions'.format(walker_idx)][:], axis=0)
            walker_ranges.append(max_ranges[0])


        run_data = np.array(np.amax(np.array(walker_ranges), axis=0))
        # print("max1=", np.max(runs_data))
        # print(runs_data)
        # outfile = "ranges_" + str(dimension)+'.csv'
        # np.savetxt(outfile, runs_data, delimiter=",", fmt='%d')

        return np.max(run_data), run_data


    # calculating  sum of probability for one cycle
    def prob_of_cycle(self, traj_data, max_x):

        weight = traj_data['weights']
        positions = traj_data['positions']

        n_walkers = positions.shape[0]
        dimension = positions.shape[2]
        pcycles = np.zeros((max_x))

        for x in range(max_x):
            p  = 0
            for walker_idx in range(n_walkers):
                for dim_idx in range(dimension):
                    p += weight[walker_idx] * \
                         self.kronecker_delta(x - positions[walker_idx, 0, dim_idx])
            pcycles[x] = p

        return pcycles

    def analyse(self, result_h5file_path, num_walkers):



        wepy_h5 = WepyHDF5(result_h5file_path, mode='r')
        wepy_h5.open()

        max_x, max_range = self.find_max_range(wepy_h5, num_walkers)

        hd = wepy_h5.h5
        n_cycles = hd['/runs/0/trajectories/0/positions'].shape[0]
        dimension =  hd['/runs/0/trajectories/0/positions'].shape[2]

        cycle_idxs = [i for i in range(n_cycles)]

        # set up the RunCycleSlice object for each run
        rcs = RunCycleSlice(0, cycle_idxs, wepy_h5)

        data_list = rcs.compute_observable(self.prob_of_cycle, ['positions', 'weights'], int(max_x),
                                           map_func=scoop.futures.map, debug_prints=True)

        cycles_sum = np.zeros(int(max_x))

        for data in data_list:
            cycles_sum += data

        prob_x = 1 / (n_cycles * dimension) * cycles_sum

        #calculating the accuracy of x
        acc = 0

        for x in range(int(max_x)):
            acc += self.accuracy(x , prob_x[x])

        # print test data
        print ("Max x= {}".format(max_x))
        print("Max range = {}".format(max_range))
        print("Probability of x ={}".format(prob_x))
        print("accuracy = {}".format(acc))

        results = {"max_x": max_x, "max_range": max_range,
                   "px":prob_x, "accuracy":acc}

        return results

if __name__=="__main__":

    from wepy.resampling.resamplers.wexplore import WExploreResampler
    from wepy.resampling.distances.randomwalk import RandomWalkDistance

    # set up  the distance function
    distance = RandomWalkDistance()

    # set up the Revo Resampler with the parameters
    # resampler = REVOResampler(pmax=0.1, pmin=10e-100,
    #                                    scorer=scorer)

    # # set up the Wexplore Resampler with the parameters
    resampler = WExploreResampler(pmax=0.1, pmin=10e-100,
                                       cellsize=[16, 4, 1,0.25], distance=distance)

    # set up a RandomWalkProfilier
    rw_profiler = RandomwalkProfiler(resampler)
    rw_profiler.run_test(num_walkers=200, num_cycles=100, dimension=5, debug_prints=False)
