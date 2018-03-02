import sys

from wepy.work_mapper.mapper import Mapper

class Manager(object):

    def __init__(self, init_walkers,
                 runner = None,
                 resampler = None,
                 boundary_conditions = None,
                 reporters = None,
                 work_mapper_type = None,
                 worker_type = None,
                 num_workers = None,
    ):

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

        # the method for writing output
        if reporters is None:
            self.reporters = []
        else:
            self.reporters = reporters

        # the mapping class
        if work_mapper_type is None:
            self.work_mapper_type = Mapper
        else:
            self.work_mapper_type = work_mapper_type

        self.worker_type = worker_type
        self.num_workers = num_workers

        # Create a work_mapper for this sim_manager.
        self._work_mapper = self.work_mapper_type(self.runner.run_segment,
                                                  self.num_workers,
                                                  worker_type=self.worker_type)



    def run_segment(self, walkers, segment_length, debug_prints=False):
        """Run a time segment for all walkers using the available workers. """

        num_walkers = len(walkers)

        if debug_prints:
            sys.stdout.write("Starting segment\n")

        new_walkers = list(self._work_mapper.map(walkers,
                                                (segment_length for i in range(num_walkers)),
                                                debug_prints=debug_prints
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

        # initialize the work_mapper, this may include things like
        # starting processes etc.
        self._work_mapper.init(debug_prints=debug_prints)

        # init the reporter
        for reporter in self.reporters:
            reporter.init()

        walkers = self.init_walkers
        # the main cycle loop
        for cycle_idx in range(n_cycles):

            if debug_prints:
                sys.stdout.write("Begin cycle {}\n".format(cycle_idx))

            # run the segment
            new_walkers = self.run_segment(walkers, segment_lengths[cycle_idx],
                                           debug_prints=debug_prints)

            if debug_prints:
                sys.stdout.write("End cycle {}\n".format(cycle_idx))

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
                                                                    cycle_idx,
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

            # resample walkers
            resampled_walkers, resampling_records, resampling_aux_data =\
                           self.resampler.resample(warped_walkers,
                                                   debug_prints=debug_prints)

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

            # report results to the reporters
            for reporter in self.reporters:
                reporter.report(cycle_idx, new_walkers,
                                     warp_records, warp_aux_data,
                                     bc_records, bc_aux_data,
                                     resampling_records, resampling_aux_data,
                                     debug_prints=debug_prints)

            # prepare resampled walkers for running new state changes
            walkers = resampled_walkers


        # cleanup things associated with the reporter
        for reporter in self.reporters:
            reporter.cleanup()

        # cleanup the mapper
        self._work_mapper.cleanup()
