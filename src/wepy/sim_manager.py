"""Module for the main simulation management class.

All component class interfaces are set by how the manager interacts
with them.

Managers should implement a three phase protocol for running
simulations:

- init
- run_simulation
- cleanup

The separate `init` method is different than the constructor
`__init__` method and instead calls the special `init` method on all
wepy components (runner, resampler, boundary conditions, and
reporters) at runtime.

This allows for a things that need to be done at runtime before a
simulation begins, e.g. opening files, that you don't want done at
construction time.

This is useful for orchestration because the complete simulation
'image' can be made before runtime (and pickled or otherwise
persisted) without producing external effects.

This is used primarily for reporters, which perform I/O, and work
mappers which may spawn processes.

The `cleanup` method should be called either when the simulation ends
normally as well as when the simulation ends abnormally.

This allows file handles to be closed and processes to be killed at
the end of a simulation and upon failure.


The base methods for running simulations is `run_cycle` which runs
a cycle of weighted ensemble given the state of all the components.

The simulation manager should provide multiple ways of running
simulations depending on if the number of cycles is known up front or
to be determined adaptively (e.g. according to some time limit).

"""

import sys
import time
from copy import deepcopy
import logging

class Manager(object):
    """The class that coordinates wepy simulations.

    The Manager class is the lynchpin of wepy simulations and is where
    all the different components are composed.

    Strictly speaking the Manager defines the interfaces each
    component must provide to function.

    Developers can call `run_cycle` directly but the following
    convenience functions are provided to run many cycles in
    succession as a single 'run' with consecutive cycle idxs:

    - run_simulation_by_time
    - run_simulation

    The corresponding 'continue' run methods will simply pass a run
    index to reporters indicating that the run continues another.

    For these run methods the `init` method is called followed by
    iterative calls to `run_cycle` and finally with a call to
    `cleanup`.

    The order of application of wepy components are:

    - runner
    - boundary_conditions
    - resampler
    - reporters

    """


    REPORT_ITEM_KEYS = ('cycle_idx', 'n_segment_steps',
                        'new_walkers', 'resampled_walkers',
                        'warp_data', 'bc_data', 'progress_data',
                        'resampling_data', 'resampler_data',
                        'worker_segment_times', 'cycle_runner_time',
                        'cycle_bc_time', 'cycle_resampling_time',)
    """Keys of values that will be passed to reporters.

    This indicates the values that the reporters will have access to.
    """

    def __init__(self, init_walkers,
                 runner = None,
                 work_mapper = None,
                 resampler = None,
                 boundary_conditions = None,
                 reporters = None):
        """Constructor for Manager.

        Arguments
        ---------

        init_walkers : list of walkers
            The list of the initial walkers that will be run.

        runner : object implementing the Runner interface
            The runner to be used for propagating sampling segments of walkers.

        work_mapper : object implementing the WorkMapper interface
            The object that will be used to perform a set of runner
            segments in a cycle.

        resampler : object implementing the Resampler interface
            The resampler to be used in the simulation

        boundary_conditions : object implementing BoundaryCondition interface, optional
            The boundary conditions to apply to walkers

        reporters : list of objects implenting the Reporter interface, optional
            Reporters to be used. You should provide these if you want to keep data.

        Warnings
        --------

        While reporters are strictly optional, you probably want to
        provide some because the simulation manager provides no
        utilities for saving data from the simulations except for the
        walkers at the end of a cycle or simulation.

        See Also
        --------
        wepy.reporter.hdf5 : The standard reporter for molecular simulations in wepy.

        wepy.orchestration.orchestrator.Orchestrator : for running simulations with
            checkpointing, restarting, reporter localization, and configuration hotswapping
            with command line interface.

        """

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
        """Run a time segment for all walkers using the available workers.

        Maps the work for running each segment for each walker using
        the work mapper.

        Walkers will have the same weights but different states.

        Parameters
        ----------
        walkers : list of walkers
        segment_length : int
            Number of steps to run in each segment.

        Returns
        -------

        new_walkers : list of walkers
           The walkers after the segment of sampling simulation.

        """

        num_walkers = len(walkers)

        logging.info("Starting segment")

        new_walkers = list(self.work_mapper.map(walkers,
                                                (segment_length for i in range(num_walkers)),
                                               )
                          )
        logging.info("Ending segment")

        return new_walkers

    def run_cycle(self, walkers, n_segment_steps, cycle_idx):
        """Run a full cycle of weighted ensemble simulation using each
        component.

        The order of application of wepy components are:

        - runner
        - boundary_conditions
        - resampler
        - reporters

        The `init` method should have been called before this or
        components may fail.

        This method is not idempotent and will alter the state of wepy
        components.

        The cycle is not kept as a state variable of the simulation
        manager and so myst be provided here. This motivation for this
        is that a cycle index is really a property of a run and runs
        can be composed in many ways and is then handled by
        higher-level methods calling run_cycle.

        Each component should implement its respective interface to be
        called in this method, the order and names of the methods
        called are as follows:

        1. runner.pre_cycle
        2. run_segment -> work_mapper.map(runner.run_segment)
        3. runner.post_cycle
        4. boundary_conditions.warp_walkers (if present)
        5. resampler.resample
        6. reporter.report for all reporters

        The pre and post cycle calls to the runner allow for a parity
        of one call per cycle to the runner.

        The boundary_conditions component is optional, as are the
        reporters (although it won't be very useful to run this
        without any).

        Parameters
        ----------
        walkers : list of walkers

        n_segment_steps : int
            Number of steps to run in each segment.

        cycle_idx : int
            The index of this cycle.

        Returns
        -------

        new_walkers : list of walkers
            The resulting walkers of the cycle

        sim_components : list
            The runner, resampler, and boundary conditions
            objects at the end of the cycle.

        See Also
        --------
        run_simulation : To run a simulation by the number of cycles
        run_simulation_by_time

        """


        # this one is called to just easily be able to catch all the
        # errors from it so we can cleanup if an error is caught

        try:
            return self._run_cycle(walkers, n_segment_steps, cycle_idx)
        except Exception as err:

            # if we catch any error we want to make sure that run the
            # cleanup for everything. By policy this should make sure
            # all running processes are killed (i.e. does not actually
            # kill processes and the modules should implement this
            # themselves in their cleanup method)
            self.cleanup()

            # then reraise the error
            raise err

    def _run_cycle(self, walkers, n_segment_steps, cycle_idx):
        """See run_cycle."""

        logging.info("Begin cycle {}".format(cycle_idx))

        # run the pre-cycle hook
        self.runner.pre_cycle(walkers=walkers,
                              n_segment_steps=n_segment_steps,
                              cycle_idx=cycle_idx)

        # run the segment
        start = time.time()
        new_walkers = self.run_segment(walkers, n_segment_steps)
        end = time.time()
        runner_time = end - start

        # run post-cycle hook
        self.runner.post_cycle()


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

        # make a dictionary of all the results that will be reported


        report = {'cycle_idx' : cycle_idx,
                  'new_walkers' : new_walkers,
                  'warp_data' : warp_data,
                  'bc_data' : bc_data,
                  'progress_data' : progress_data,
                  'resampling_data' : resampling_data,
                  'resampler_data' : resampler_data,
                  'n_segment_steps' : n_segment_steps,
                  'worker_segment_times' : self.work_mapper.worker_segment_times,
                  'cycle_runner_time' : runner_time,
                  'cycle_bc_time' : bc_time,
                  'cycle_resampling_time' : resampling_time,
                  'resampled_walkers' : resampled_walkers}

        # check that all of the keys that are specified for this sim
        # manager are present
        assert all([True if rep_key in report else False
                    for rep_key in self.REPORT_ITEM_KEYS])

        # report results to the reporters
        for reporter in self.reporters:
            reporter.report(**report)

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
        """Initialize wepy configuration components for use at runtime.

        This `init` method is different than the constructor
        `__init__` method and instead calls the special `init` method
        on all wepy components (runner, resampler, boundary
        conditions, and reporters) at runtime.

        This allows for a things that need to be done at runtime before a
        simulation begins, e.g. opening files, that you don't want done at
        construction time.

        It calls the `init` methods on:

        - work_mapper
        - reporters

        Passes the segment_func of the runner and the number of
        workers to the work_mapper.

        Passes the following things to each reporter `init` method:

        - init_walkers
        - runner
        - resampler
        - boundary_conditions
        - work_mapper
        - reporters
        - continue_run

        Parameters
        ----------
        num_workers : int
            The number of workers to use in the work mapper.
             (Default value = None)
        continue_run : int
            Index of a run this one is continuing.
             (Default value = None)

        """


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
        """Perform cleanup actions for wepy configuration components.

        Allow components to perform actions before ending the main
        simulation manager process.

        Calls the `cleanup` method on:

        - work_mapper
        - reporters

        Passes nothing to the work mapper.

        Passes the following to each reporter:

        - runner
        - work_mapper
        - resampler
        - boundary_conditions
        - reporters

        """

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
        """Run a simulation for a certain amount of time.

        This starts timing as soon as this is called. If the time
        before running a new cycle is greater than the runtime the run
        will exit after cleaning up. Once a cycle is started it may
        also run over the wall time.

        All this does is provide a run idx to the reporters, which is
        the run that is intended to be continued. This simulation
        manager knows no details and is left up to the reporters to
        handle this appropriately.

        Parameters
        ----------
        run_time : float
            The time to run in seconds.

        segments_length : int
            The number of steps for each runner segment.

        num_workers : int
            The number of workers to use for the work mapper.
             (Default value = None)

        Returns
        -------
        new_walkers : list of walkers
            The resulting walkers of the cycle

        sim_components : list
            Deep copies of the runner, resampler, and boundary
            conditions objects at the end of the cycle.

        See Also
        --------
        wepy.orchestration.orchestrator.Orchestrator : for running simulations with
            checkpointing, restarting, reporter localization, and configuration hotswapping
            with command line interface.

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
        """Run a simulation for an explicit number of cycles.

        Parameters
        ----------
        n_cycles : int
            Number of cycles to perform.

        segment_lengths : int
            The number of steps for each runner segment.

        num_workers : int
            The number of workers to use for the work mapper.
             (Default value = None)

        Returns
        -------
        new_walkers : list of walkers
            The resulting walkers of the cycle

        sim_components : list
            Deep copies of the runner, resampler, and boundary
            conditions objects at the end of the cycle.

        See Also
        --------
        wepy.orchestration.orchestrator.Orchestrator : for running simulations with
            checkpointing, restarting, reporter localization, and configuration hotswapping
            with command line interface.

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

        Parameters
        ----------
        run_idx : int
            Index of the run you are continuing.

        n_cycles : int
            Number of cycles to perform.

        segment_lengths : int
            The number of steps for each runner segment.

        num_workers : int
            The number of workers to use for the work mapper.
             (Default value = None)

        Returns
        -------
        new_walkers : list of walkers
            The resulting walkers of the cycle

        sim_components : list
            Deep copies of the runner, resampler, and boundary
            conditions objects at the end of the cycle.

        See Also
        --------
        wepy.orchestration.orchestrator.Orchestrator : for running simulations with
            checkpointing, restarting, reporter localization, and configuration hotswapping
            with command line interface.

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
        """Continue a simulation with a separate run by time.

        This starts timing as soon as this is called. If the time
        before running a new cycle is greater than the runtime the run
        will exit after cleaning up. Once a cycle is started it may
        also run over the wall time.

        All this does is provide a run idx to the reporters, which is
        the run that is intended to be continued. This simulation
        manager knows no details and is left up to the reporters to
        handle this appropriately.

        Parameters
        ----------
        run_idx : int
            Deep copies of the runner, resampler, and boundary
            conditions objects at the end of the cycle.


        See Also
        --------
        wepy.orchestration.orchestrator.Orchestrator : for running simulations with
            checkpointing, restarting, reporter localization, and configuration hotswapping
            with command line interface.

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
