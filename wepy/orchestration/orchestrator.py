from copy import copy, deepcopy
import pickle
from hashlib import md5
from warnings import warn

class SimApparatus():
    """The simulation apparatus are the components needed for running a
    simulation without the initial conditions for starting the simulation.

    A runner is strictly necessary but a resampler and boundary
    conditions are not.

    """

    def __init__(self, runner, resampler=None, boundary_conditions=None):
        self.runner = deepcopy(runner)
        self.resampler = deepcopy(resampler)
        self.boundary_conditions = deepcopy(boundary_conditions)

class SimSnapshot():

    def __init__(self, init_walkers, apparatus):

        self._init_walkers = deepcopy(init_walkers)
        self._apparatus = deepcopy(apparatus)

    @property
    def init_walkers(self):
        return self._init_walkers

    @property
    def apparatus(self):
        return self._apparatus


class OrchestratorError(Exception):
    pass

class Orchestrator():

    def __init__(self, sim_apparatus):

        # the apparatus for the simulation. This is the configuration
        # and initial conditions independent components necessary for
        # running a simulation. The primary (and only necessary)
        # component is the runner. The other components when passed
        # here are used as the defaults when unspecified later
        # (e.g. resampler and boundary conditions)
        self._apparatus = deepcopy(sim_apparatus)

        # make a pickle of the apparatus as a string and make a hash
        # of the pickle string for checking data integrity later
        self._apparatus_pkl_md5 = md5(pickle.dumps(self.apparatus)).hexdigest()

        # the main dictionary of snapshots keyed by their hashes
        self._snapshots = {}

        # list of "segments" which are 2-tuples of hashes, where the
        # first hash is the predecessor to the second hash. The second
        # hash may be an end of a run or a checkpoint of a run.
        self._segments = []

        # the list of "runs" which are tuples of hashes for the starts
        # and ends of runs. THis just excludes the checkpoints, this
        # really is for convenience so you can ignore all the
        # checkpoints when reconstructing full run continuations etc.
        self._runs = []


        # the list of "mutations". Again a 2-tuple of hashes
        # indicating an input and output snapshot. Mutations are when
        # the the state of the walkers in the snapshot do not change
        # but the state associated with the filters (e.g. resampler,
        # boundary conditions, runner) or the number and weights of
        # the walkers is subject to change
        self._mutations = []

        # TODO: default work mapper and reporters set in constructor

        # computing environment specific configuration components
        #self._work_mapper = work_mapper
        #self._extra_reporters = reporters


    @property
    def apparatus(self):
        return self._apparatus

    def add_snapshot(self, sim_snapshot):

        # get the hash of the snapshot
        snapshot_md5 = md5(pickle.dumps(sim_snapshot)).hexdigest()

        # check that the hash is not already in the snapshots
        if any([True if snapshot_md5 == md5 else False for md5 in self._snapshots.keys()]):

            # raise an error that the snapshot is already in the
            # collection
            raise OrchestratorError(
                "snapshot {} already in orchestrator snapshots, ignoring".format(snapshot_md5))

        self._snapshots[snapshot_md5] = sim_snapshot

        return snapshot_md5


    def gen_sim_start(self, init_walkers, resampler=None, boundary_conditions=None):

        # make a SimSnapshot object using the initial walkers and
        # optionally replacing the filter states
        sim_start = SimSnapshot(init_walkers, self._apparatus,
                                resampler=resampler,
                                boundary_conditions=boundary_conditions)

        # save the snapshot and its hash in the dictionary of snapshots
        sim_start_md5 = self._add_snapshot(sim_start_md5, sim_start)

        return sim_start_md5



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
