import pickle
from hashlib import md5

class SimApparatus():
    """The simulation apparatus are the components needed for running a
    simulation without the initial conditions for starting the simulation.

    A runner is strictly necessary but a resampler and boundary
    conditions are not.

    """

    def __init__(self, runner, resampler=None, boundary_conditions=None):
        self.runner = runner
        self.resampler = resampler
        self.boundary_conditions = boundary_conditions

class SimSnapshot():

    def __init__(self, ):
        pass

class Orchestrator():

    def __init__(self, runner,
                 resampler=None,
                 boundary_conditions=None,
                 work_mapper=None,
                 reporters=None):

        self._runner = runner
        self._work_mapper = work_mapper

        self._extra_reporters = reporters
        self._resampler = resampler
        self._boundary_conditions = boundary_conditions


        # the status of each run recorded in this object
        self._run_states = {}


        # construct the apparatus that will be used to run all
        # subsequent simulations given initial conditions
        self._apparatus = SimApparatus(self._runner, self._resampler, self._boundary_conditions)

        # make a pickle of the apparatus as a string and make a hash
        # of the pickle string for checking data integrity later
        self._apparatus_pkl_md5 = md5(pickle.dumps(self.apparatus)).hexdigest()


    def _gen_sim_start(self, init_walkers, resampler=None, boundary_conditions=None):

        # make a SimSnapshot object using the initial walkers and
        # optionally replacing the filter states
        sim_start = SimSnapshot(self._apparatus, resampler=resampler,
                                boundary_conditions=boundary_conditions)

        # save the snapshot and its hash in the dictionary of snapshots
        sim_start_md5 = self._add_snapshot(sim_start_md5, sim_start)

        return sim_start_md5

    def _add_snapshot(self, sim_snapshot):

        # get the hash of the snapshot
        snapshot_md5 = md5(pickle.dumps(sim_snapshot)).hexdigest()

        self._snapshots[snapshot_md5] = sim_snapshot

        return snapshot_md5



    def gen_sim_manager(self):

        # copy all of the objects before construction
        reporters = [deepcopy(reporter) for reporter in self.reporters]
        restart_walkers = deepcopy(self.restart_walkers)
        runner = deepcopy(self.runner)
        resampler = deepcopy(self.resampler)
        boundary_conditions = deepcopy(self.boundary_conditions)
        work_mapper = deepcopy(self.work_mapper)


        # TODO
        # add in orchestrated reporters with appropriate modifications
        pass

        # construct the sim manager
        sim_manager = Manager(restart_walkers,
                              runner=runner,
                              resampler=resampler,
                              boundary_conditions=boundary_conditions,
                              work_mapper=work_mapper,
                              reporters=reporters)

        return sim_manager


    def new_run_by_time(self, init_walkers, run_time, n_steps, n_workers=None):
        """Start a new run that will go for a certain amount of time given a
        new set of initial conditions. """

        sim_manager = self.gen_sim_manager()

        sim_manager.run_simulation_by_time(run_time, n_steps, n_workers=n_workers)

    def continue_run(self, run_id):
        """For a finished run continue it but resetting all the state of the
        resampler and boundary conditions"""

        pass

    def restart_run(self, run_id):
        """For a finished run continue it and don't reset the state of the
        resampler and boundary conditions."""

        pass

    def recover_run(self, run_id):
        """For a run that ended in a bad state recover it and restart it. """

        pass

    def run_status(self, run_id):
        """Return the status of a run. """

        pass
