import sys
import time
from copy import deepcopy
import logging

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


    def run_segment(self, walkers, segment_length):
        """Run a time segment for all walkers using the available workers. """

        num_walkers = len(walkers)

        logging.info("Starting segment")

        new_walkers = list(self.work_mapper.map(walkers,
                                                (segment_length for i in range(num_walkers)),
                                               )
                          )
        logging.info("Ending segment")

        return new_walkers

    def run_cycle(self, walkers, segment_length, cycle_idx):

        logging.info("Begin cycle {}".format(cycle_idx))

        # run the segment
        start = time.time()
        new_walkers = self.run_segment(walkers, segment_length)
        end = time.time()
        runner_time = end - start

        logging.info("End cycle {}".format(cycle_idx))

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
                                                                cycle_idx)
            end = time.time()
            bc_time = end - start

            # warping results
            warped_walkers = bc_results[0]
            warp_data = bc_results[1]
            bc_data = bc_results[2]
            progress_data = bc_results[3]

            if len(warp_data) > 0:
                logging.info("Returned warp record in cycle {}".format(cycle_idx))



        # resample walkers
        start = time.time()
        resampling_results = self.resampler.resample(warped_walkers)
        end = time.time()
        resampling_time = end - start

        resampled_walkers = resampling_results[0]
        resampling_data = resampling_results[1]
        resampler_data = resampling_results[2]

        # log the weights of the walkers after resampling
        result_template_str = "|".join(["{:^5}" for i in range(self.n_init_walkers + 1)])
        walker_weight_str = result_template_str.format("weight",
            *[round(walker.weight, 3) for walker in resampled_walkers])
        logging.info(walker_weight_str)

        # report results to the reporters
        for reporter in self.reporters:
            reporter.report(cycle_idx, new_walkers,
                            warp_data, bc_data, progress_data,
                            resampling_data, resampler_data,
                            n_steps=segment_length,
                            worker_segment_times=self.work_mapper.worker_segment_times,
                            cycle_runner_time=runner_time,
                            cycle_bc_time=bc_time,
                            cycle_resampling_time=resampling_time,
                            resampled_walkers=resampled_walkers)

        # prepare resampled walkers for running new state changes
        walkers = resampled_walkers


        # we also return a list of the "filters" which are the
        # classes that are run on the initial walkers to produce
        # the final walkers. THis is to satisfy a future looking
        # interface in which the order and components of these
        # filters are completely parametrizable. This may or may
        # not be implemented in a future release of wepy but this
        # interface is assumed by the orchestration classes for
        # making snapshots of the simulations. The receiver of
        # these should perform the copy to make sure they aren't
        # mutated. We don't do this here for efficiency.
        filters = [self.runner, self.boundary_conditions, self.resampler]

        return walkers, filters

    def init(self, num_workers=None, continue_run=None):


        logging.info("Starting simulation")

        # initialize the work_mapper with the function it will be
        # mapping and the number of workers, this may include things like starting processes
        # etc.
        self.work_mapper.init(segment_func=self.runner.run_segment,
                              num_workers=num_workers)

        # init the reporter
        for reporter in self.reporters:
            reporter.init(init_walkers=self.init_walkers,
                          runner=self.runner,
                          resampler=self.resampler,
                          boundary_conditions=self.boundary_conditions,
                          work_mapper=self.work_mapper,
                          reporters=self.reporters,
                          continue_run=continue_run)

    def cleanup(self):

        # cleanup the mapper
        self.work_mapper.cleanup()

        # cleanup things associated with the reporter
        for reporter in self.reporters:
            reporter.cleanup(runner=self.runner,
                             work_mapper=self.work_mapper,
                             resampler=self.resampler,
                             boundary_conditions=self.boundary_conditions,
                             reporters=self.reporters)


    def run_simulation_by_time(self, run_time, segments_length, num_workers=None):
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
        self.init(num_workers=num_workers)
        cycle_idx = 0
        walkers = self.init_walkers
        while time.time() - start_time < run_time:

            logging.info("starting cycle {} at time {}".format(cycle_idx, time.time() - start_time))

            walkers, filters = self.run_cycle(walkers, segments_length, cycle_idx)

            logging.info("ending cycle {} at time {}".format(cycle_idx, time.time() - start_time))

            cycle_idx += 1

        self.cleanup()

        return walkers, deepcopy(filters)

    def run_simulation(self, n_cycles, segment_lengths, num_workers=None):
        """Run a simulation for a given number of cycles with specified
        lengths of MD segments in between.

        """

        self.init(num_workers=num_workers)

        walkers = self.init_walkers
        # the main cycle loop
        for cycle_idx in range(n_cycles):
            walkers, filters = self.run_cycle(walkers, segment_lengths[cycle_idx], cycle_idx)

        self.cleanup()

        return walkers, deepcopy(filters)

    def continue_run_simulation(self, run_idx, n_cycles, segment_lengths, num_workers=None):
        """Continue a simulation. All this does is provide a run idx to the
        reporters, which is the run that is intended to be
        continued. This simulation manager knows no details and is
        left up to the reporters to handle this appropriately.

        """

        self.init(num_workers=num_workers,
                  continue_run=run_idx)

        walkers = self.init_walkers
        # the main cycle loop
        for cycle_idx in range(n_cycles):
            walkers, filters = self.run_cycle(walkers, segment_lengths[cycle_idx], cycle_idx)

        self.cleanup()

        return walkers, filters


    def continue_run_simulation_by_time(self, run_idx, run_time, segments_length, num_workers=None):
        """Continue a simulation. All this does is provide a run idx to the
        reporters, which is the run that is intended to be
        continued. This simulation manager knows no details and is
        left up to the reporters to handle this appropriately.

        """

        start_time = time.time()

        self.init(num_workers=num_workers,
                  continue_run=run_idx)

        cycle_idx = 0
        walkers = self.init_walkers
        while time.time() - start_time < run_time:

            logging.info("starting cycle {} at time {}".format(cycle_idx, time.time() - start_time))

            walkers, filters = self.run_cycle(walkers, segments_length, cycle_idx)

            logging.info("ending cycle {} at time {}".format(cycle_idx, time.time() - start_time))

            cycle_idx += 1

        self.cleanup()

        return walkers, filters
