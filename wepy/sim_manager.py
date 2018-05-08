import sys
import time

from wepy.work_mapper.mapper import Mapper

class Manager(object):

    def __init__(self, init_walkers,
                 runner = None,
                 resampler = None,
                 boundary_conditions = None,
                 reporters = None,
                 work_mapper = None
    ):

        self.init_walkers = init_walkers
        self.n_init_walkers = len(init_walkers)

        # the runner is the object that runs dynamics
        self.runner = runner
        # the resampler
        self.resampler = resampler
        # object for boundary conditions
        self.boundary_conditions = boundary_conditions

        # the method for writing output
        if reporters is None:
            self.reporters = []
        else:
            self.reporters = reporters

        self.work_mapper = work_mapper


    def run_segment(self, walkers, segment_length, debug_prints=False):
        """Run a time segment for all walkers using the available workers. """

        num_walkers = len(walkers)

        if debug_prints:
            sys.stdout.write("Starting segment\n")

        new_walkers = list(self.work_mapper.map(walkers,
                                                (segment_length for i in range(num_walkers)),
                                                debug_prints=debug_prints
                                               )
                          )
        if debug_prints:
            sys.stdout.write("Ending segment\n")

        return new_walkers

    def run_cycle(self, walkers, segment_length, cycle_idx,
                  debug_prints=False):

            if debug_prints:
                sys.stdout.write("Begin cycle {}\n".format(cycle_idx))

            # run the segment
            start = time.time()
            new_walkers = self.run_segment(walkers, segment_length,
                                           debug_prints=debug_prints)
            end = time.time()
            runner_time = end - start

            if debug_prints:
                sys.stdout.write("End cycle {}\n".format(cycle_idx))

            # boundary conditions should be optional;

            # initialize the warped walkers to the new_walkers and
            # change them later if need be
            warped_walkers = new_walkers
            warp_data = []
            bc_data = []
            progress_data = []
            bc_time = 0.0
            if self.boundary_conditions is not None:

                # apply rules of boundary conditions and warp walkers through space
                start = time.time()
                bc_results  = self.boundary_conditions.warp_walkers(new_walkers,
                                                                    cycle_idx,
                                                                    debug_prints=debug_prints)
                end = time.time()
                bc_time = end - start

                # warping results
                warped_walkers = bc_results[0]
                warp_data = bc_results[1]
                bc_data = bc_results[2]
                progress_data = bc_results[3]

                if debug_prints:
                    if len(warp_data) > 0:
                        print("Returned warp record in cycle {}".format(cycle_idx))



            # resample walkers
            start = time.time()
            resampling_results = self.resampler.resample(warped_walkers,
                                                       debug_prints=debug_prints)
            end = time.time()
            resampling_time = end - start

            resampled_walkers = resampling_results[0]
            resampling_data = resampling_results[1]
            resampler_data = resampling_results[2]

            if debug_prints:
                # print results for this cycle
                print("Net state of walkers after resampling:")
                print("--------------------------------------")
                # slots
                slot_str = self.result_template_str.format("walker",
                                                      *[i for i in range(len(resampled_walkers))])
                print(slot_str)
                # weights
                walker_weight_str = self.result_template_str.format("weight",
                    *[round(walker.weight, 3) for walker in resampled_walkers])
                print(walker_weight_str)

            # report results to the reporters
            for reporter in self.reporters:
                reporter.report(cycle_idx, new_walkers,
                                warp_data, bc_data, progress_data,
                                resampling_data, resampler_data,
                                debug_prints=debug_prints,
                                n_steps=segment_length,
                                worker_segment_times=self.work_mapper.worker_segment_times,
                                cycle_runner_time=runner_time,
                                cycle_bc_time=bc_time,
                                cycle_resampling_time=resampling_time)

            # prepare resampled walkers for running new state changes
            walkers = resampled_walkers

            return walkers

    def init(self, num_workers, debug_prints=False):

        if debug_prints:
            self.result_template_str = "|".join(["{:^5}" for i in range(self.n_init_walkers + 1)])
            sys.stdout.write("Starting simulation\n")

        # initialize the work_mapper with the function it will be
        # mapping and the number of workers, this may include things like starting processes
        # etc.
        self.work_mapper.init(self.runner.run_segment, num_workers=num_workers,
                               debug_prints=debug_prints)

        # init the reporter
        for reporter in self.reporters:
            reporter.init()

    def cleanup(self, debug_prints=False):
        # cleanup things associated with the reporter
        for reporter in self.reporters:
            reporter.cleanup()

        # cleanup the mapper
        self.work_mapper.cleanup()

    def run_simulation_by_time(self, run_time, segments_length, num_workers=None,
                               debug_prints=False):
        """Run a simulation for a certain amount of time. This starts timing
        as soon as this is called. If the time before running a new
        cycle is greater than the runtime the run will exit after
        cleaning up. Once a cycle is started it may also run over the
        wall time.

        run_time :: float (in seconds)

        segments_length :: int ; number of iterations performed for
                                 each walker segment for each cycle

        """
        start_time = time.time()
        self.init(num_workers, debug_prints=debug_prints)
        cycle_idx = 0
        walkers = self.init_walkers
        while time.time() - start_time < run_time:

            if debug_prints:
                print("starting cycle {} at time {}".format(cycle_idx, time.time() - start_time))

            walkers = self.run_cycle(walkers, segments_length, cycle_idx,
                                       debug_prints=debug_prints)

            if debug_prints:
                print("ending cycle {} at time {}".format(cycle_idx, time.time() - start_time))

            cycle_idx += 1

        self.cleanup(debug_prints=debug_prints)

        return walkers

    def run_simulation(self, n_cycles, segment_lengths, num_workers=None,
                       debug_prints=False):
        """Run a simulation for a given number of cycles with specified
        lengths of MD segments in between.

        """

        self.init(num_workers, debug_prints=debug_prints)

        walkers = self.init_walkers
        # the main cycle loop
        for cycle_idx in range(n_cycles):
            walkers = self.run_cycle(walkers, segment_lengths[cycle_idx], cycle_idx,
                                         debug_prints=debug_prints)

        self.cleanup(debug_prints=debug_prints)

        return walkers



