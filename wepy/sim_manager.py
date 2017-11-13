import sys
import time
import os
import os.path as osp
import pickle

from wepy.resampling.resampler import NoResampler
from wepy.runner import NoRunner

class CountingIterator(object):
    def __init__(self, iterable):
        self.indices = []
        self.iterable = self.count(iterable)

    def count(self, iterable):
        xs = []
        for i, x in enumerate(iterable):
            self.indices.append(i)
            xs.append(x)
        return (x for x in xs)

class Manager(object):

    def __init__(self, init_walkers, num_workers=None,
                 runner = None,
                 resampler = None,
                 boundary_conditions = None,
                 work_mapper = map,
                 reporter = None,
                 backup_freq = None):

        self.init_walkers = init_walkers
        self.n_init_walkers = len(init_walkers)

        # the number of cores to use
        self.num_workers = num_workers
        # the runner is the object that runs dynamics
        self.runner = runner
        # the resampler
        self.resampler = resampler
        # object for boundary conditions
        self.boundary_conditions = boundary_conditions
        # the function for running work on the workers
        self.map = work_mapper
        # the method for writing output
        self.reporter = reporter
        # the frequency for which to pkl all walkers
        self.backup_freq = backup_freq

    def run_segment(self, walkers, segment_length, debug_prints=False):
        """Run a time segment for all walkers using the available workers. """

        num_walkers = len(walkers)

        if debug_prints:
            sys.stdout.write("Starting segment\n")

        new_walkers = list(self.map(self.runner.run_segment,
                                    walkers,
                                    (segment_length for i in range(num_walkers))
                                   )
                          )
        if debug_prints:
            sys.stdout.write("Ending segment\n")

        return new_walkers

    def run_simulation(self, n_cycles, segment_lengths, debug_prints=False):
        """Run a simulation for a given number of cycles with specified
        lengths of MD segments in between.

        Can either return results in memory or write to a file.
        """

        if debug_prints:
            result_template_str = "|".join(["{:^10}" for i in range(self.n_init_walkers + 1)])
            sys.stdout.write("Starting simulation\n")

        # init the reporter
        self.reporter.init()

        walkers = self.init_walkers
        # the main cycle loop
        for cycle_idx in range(n_cycles):

            if debug_prints:
                sys.stdout.write("Begin cycle {}\n".format(cycle_idx))
                start_time = time.time()
                
            # run the segment
            new_walkers = self.run_segment(walkers, segment_lengths[cycle_idx],
                                           debug_prints=debug_prints)

            if debug_prints:
                sys.stdout.write("End cycle {}\n".format(cycle_idx))
                end_time = time.time()
                sys.stdout.write("Time spent on dynamics {}\n".format(end_time-start_time))

            # boundary conditions should be optional;

            # initialize the warped walkers to the new_walkers and
            # change them later if need be
            warped_walkers = new_walkers
            warp_records = []
            warp_aux_data = []
            bc_records = []
            bc_aux_data = []
            if self.boundary_conditions is not None:

                # apply rules of boundary conditions and warp walkers through space
                bc_results  = self.boundary_conditions.warp_walkers(new_walkers,
                                                            debug_prints=debug_prints)

                # warping results
                warped_walkers = bc_results[0]
                warp_records = bc_results[1]
                warp_aux_data = bc_results[2]

                if debug_prints:
                    if len(warp_records) > 0:
                        print("Returned warp record in cycle {}".format(cycle_idx))

                    if len(warp_aux_data) > 0:
                        print("Returned warp aux_data in cycle {}".format(cycle_idx))


                # boundary conditions checking results
                bc_records = bc_results[3]
                bc_aux_data = bc_results[4]


            if debug_prints:
                start_time = time.time()
                
            # resample walkers
            resampled_walkers, resampling_records, resampling_aux_data =\
                           self.resampler.resample(warped_walkers,
                                                   debug_prints=debug_prints)

            if debug_prints:
                end_time = time.time()
                sys.stdout.write("Resampling time {}\n".format(end_time-start_time))
                
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

            # report results to the reporter
            self.reporter.report(cycle_idx, new_walkers,
                                 warp_records, warp_aux_data,
                                 bc_records, bc_aux_data,
                                 resampling_records, resampling_aux_data,
                                 debug_prints=debug_prints)

            # prepare resampled walkers for running new state changes
            walkers = resampled_walkers

            if self.backup_freq != None:
                if cycle_idx % self.backup_freq == 0:
                    # pkl walkers
                    pklname = "walker_backup_" + str(cycle_idx) + ".pkl"
                    with open(pklname, 'wb') as f:
                        pickle.dump(walkers, f, pickle.HIGHEST_PROTOCOL)

                    # remove old pickle (but keep two at all times)
                    idx = cycle_idx - 2*self.backup_freq
                    oldpkl = "walker_backup_" + str(idx) + ".pkl"
                    try:
                        os.remove(oldpkl)
                    except OSError:
                        pass

        # cleanup things associated with the reporter
        self.reporter.cleanup()
