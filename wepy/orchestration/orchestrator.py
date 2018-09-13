from copy import copy, deepcopy
import pickle
from hashlib import md5
from warnings import warn
import time
import os
import os.path as osp

from wepy.sim_manager import Manager
from wepy.orchestration.configuration import Configuration

class SimApparatus():
    """The simulation apparatus are the components needed for running a
    simulation without the initial conditions for starting the simulation.

    A runner is strictly necessary but a resampler and boundary
    conditions are not.

    """

    def __init__(self, filters):
        self._filters = deepcopy(filters)

    @property
    def filters(self):
        return self._filters


class WepySimApparatus(SimApparatus):

    def __init__(self, runner, resampler=None, boundary_conditions=None):

        # add them in the order they are done in Wepy
        filters = [runner]
        if boundary_conditions is not None:
            filters.append(boundary_conditions)
        if resampler is not None:
            filters.append(resampler)

        super().__init__(filters)

    @property
    def work_mapper(self):
        return self._work_mapper

    @property
    def reporters(self):
        return self._reporters

class SimSnapshot():

    def __init__(self, walkers, apparatus):

        self._walkers = deepcopy(walkers)
        self._apparatus = deepcopy(apparatus)

    @property
    def walkers(self):
        return self._walkers

    @property
    def apparatus(self):
        return self._apparatus


class OrchestratorError(Exception):
    pass

class Orchestrator():


    DEFAULT_WORKDIR = Configuration.DEFAULT_WORKDIR
    DEFAULT_CONFIG_NAME = Configuration.DEFAULT_CONFIG_NAME
    DEFAULT_NARRATION = Configuration.DEFAULT_NARRATION
    DEFAULT_MODE = Configuration.DEFAULT_MODE

    CHECKPOINT_FILENAME_TEMPLATE = "{run_start_hash}_{checkpoint_hash}.chk.pkl"
    DEFAULT_CHECKPOINT_DIR = 'checkpoints'
    ORCH_FILENAME_TEMPLATE = "{config}{narration}.orch.pkl"
    DEFAULT_ORCHESTRATION_MODE = 'xb'

    def __init__(self, sim_apparatus, default_configuration=None,
                 default_work_dir=None):
        # the main dictionary of snapshots keyed by their hashes
        self._snapshots = {}

        # the main dictionary for the apparatuses keyed by their hashes
        self._apparatuses = {}

        # list of "segments" which are 2-tuples of hashes, where the
        # first hash is the predecessor to the second hash. The second
        # hash may be an end of a run or a checkpoint of a run.
        self._segments = set()

        # the list of "runs" which are tuples of hashes for the starts
        # and ends of runs. THis just excludes the checkpoints, this
        # really is for convenience so you can ignore all the
        # checkpoints when reconstructing full run continuations etc.
        self._runs = set()

        # a dictionary of the checkpoints associated with each run
        self._run_checkpoints = {}

        # the list of "mutations". Again a 2-tuple of hashes
        # indicating an input and output snapshot. Mutations are when
        # the the state of the walkers in the snapshot do not change
        # but the state associated with the filters (e.g. resampler,
        # boundary conditions, runner) or the number and weights of
        # the walkers is subject to change
        self._mutations = set()


        # the apparatus for the simulation. This is the configuration
        # and initial conditions independent components necessary for
        # running a simulation. The primary (and only necessary)
        # component is the runner. The other components when passed
        # here are used as the defaults when unspecified later
        # (e.g. resampler and boundary conditions)
        self._apparatus = deepcopy(sim_apparatus)

        # then add the apparatus to the apparatuses
        self._apparatus_hash = self.add_apparatus(self._apparatus)

        # if a configuration was given use this as the default
        # configuration, if none is given then we will use the default
        # one
        if default_configuration is not None:
            self._configuration = deepcopy(default_configuration)
        # TODO: make a default configuration class here
        else:
            self._configuration = None


        # if a default work dir was not given we set it as the current
        # directory for wherever the process calling a run function of
        # the orchestrator is. That is the full real path will be
        # determined at runtime when a run is called, not on creation
        # of this object
        if default_work_dir is None:
            self._work_dir = "."
        # otherwise we just set it to wherever the path said to. This
        # can contain things that will be realized to a path later
        # with the osp.realpath() function such as "~/dir/for/running"
        # will get expanded to the full path
        else:
            self._work_dir = work_dir



    def hash_snapshot(self, snapshot):

        return md5(pickle.dumps(snapshot)).hexdigest()

    def hash_apparatus(self, apparatus):

        return md5(pickle.dumps(apparatus)).hexdigest()

    def get_snapshot(self, snapshot_hash):
        """Returns a copy of a snapshot."""

        return deepcopy(self._snapshots[snapshot_hash])

    @property
    def snapshots(self):
        return deepcopy(list(self._snapshots.values()))

    @property
    def snapshot_hashes(self):
        return list(self._snapshots.keys())

    def get_apparatus(self, apparatus_hash):
        """Returns a copy of a apparatus."""

        return deepcopy(self._apparatuses[apparatus_hash])

    @property
    def apparatuses(self):
        return deepcopy(list(self._apparatuses.values()))

    @property
    def apparatus_hashes(self):
        return list(self._apparatuses.keys())

    @property
    def default_apparatus(self):
        return self._apparatus

    @property
    def default_apparatus_hash(self):
        return self._apparatus_hash

    @property
    def default_configuration(self):
        return self._configuration

    @property
    def default_work_dir(self):
        return self._default_work_dir

    def snapshot_registered(self, snapshot):

        snapshot_md5 = self.hash_snapshot(snapshot)
        if any([True if snapshot_md5 == h else False for h in self.snapshot_hashes]):
            return True
        else:
            return False

    def apparatus_registered(self, apparatus):

        apparatus_md5 = self.hash_apparatus(apparatus)
        if any([True if apparatus_md5 == h else False for h in self.apparatus_hashes]):
            return True
        else:
            return False

    def snapshot_hash_registered(self, snapshot_hash):

        if any([True if snapshot_hash == h else False for h in self.snapshot_hashes]):
            return True
        else:
            return False

    def apparatus_hash_registered(self, apparatus_hash):

        if any([True if apparatus_hash == h else False for h in self.apparatus_hashes]):
            return True
        else:
            return False

    @property
    def segments(self):
        return deepcopy(self._segments)

    @property
    def runs(self):
        return list(deepcopy(self._runs))

    @property
    def mutations(self):
        return deepcopy(self._mutations)

    def get_run_checkpoints(self, run_id):
        return deepcopy(self._run_checkpoints[run_id])

    def add_snapshot(self, snapshot):

        # copy the snapshot
        snapshot = deepcopy(snapshot)


        # add the apparatus to the apparatuses if it isn't there
        # already
        apparatus_hash = self.add_apparatus(snapshot.apparatus)

        # get the hash of the snapshot
        snapshot_md5 = self.hash_snapshot(snapshot)

        # check that the hash is not already in the snapshots
        if any([True if snapshot_md5 == md5 else False for md5 in self.snapshot_hashes]):

            # just skip the rest of the function and return the hash
            return snapshot_md5

        self._snapshots[snapshot_md5] = snapshot

        return snapshot_md5

    def add_apparatus(self, apparatus):

        apparatus = deepcopy(apparatus)

        apparatus_md5 = self.hash_apparatus(apparatus)

        # check that the hash is not already in the snapshots
        if any([True if apparatus_md5 == md5 else False for md5 in self.apparatus_hashes]):

            # just skip the rest and return the hash
            return apparatus_md5

        self._apparatuses[apparatus_md5] = apparatus

        return apparatus_md5

    # TODO: not sure how to actually implement this but here is the
    # bookkeeping stuff.
    def mutate_apparatus(self, apparatus_hash):

        # mutated_apparatus =

        mutated_hash = self.add_apparatus(mutated_apparatus)

        self.register_mutation(apparatus_hash, mutated_hash)


    def gen_start_snapshot(self, init_walkers, apparatus_hash=None):

        # if the apparatus_hash is None we will use the default one
        if apparatus_hash is None:
            apparatus_hash = self.default_apparatus_hash

        # get the apparatus
        apparatus = self.get_apparatus(apparatus_hash)

        # make a SimSnapshot object using the initial walkers and
        # optionally replacing the apparatus with mutated values
        start_snapshot = SimSnapshot(init_walkers, apparatus)

        # save the snapshot, and generate its hash
        sim_start_md5 = self.add_snapshot(start_snapshot)

        return sim_start_md5

    def gen_sim_manager(self, start_snapshot, configuration=None):

        if configuration is None:
            configuration = deepcopy(self.default_configuration)

        # copy the snapshot to use for the sim_manager
        start_snapshot = deepcopy(start_snapshot)

        # construct the sim manager, in a wepy specific way
        sim_manager = Manager(start_snapshot.walkers,
                              runner=start_snapshot.apparatus.filters[0],
                              boundary_conditions=start_snapshot.apparatus.filters[1],
                              resampler=start_snapshot.apparatus.filters[2],
                              # configuration options
                              work_mapper=configuration.work_mapper,
                              reporters=configuration.reporters)

        return sim_manager

    # TODO: add in checking to make sure you don't add a mutation to
    # either run or segments and vice versa adding in segments to
    # mutations. So don't hurt yourself.

    def register_run(self, start_hash, end_hash, checkpoints=[]):

        # make sure it is a segment first
        self.register_segment(start_hash, end_hash)

        # then add it to the set of runs too
        self._runs.add((start_hash, end_hash))

        # add the checkpoints to the dictionary of run checkpoints
        self._run_checkpoints[(start_hash, end_hash)] = checkpoints

    def register_segment(self, start_hash, end_hash):

        # check that the hashes are for snapshots in the orchestrator
        # if one is not registered raise an error
        if not self.snapshot_hash_registered(start_hash):
            raise OrchestratorError(
                "snapshot start_hash {} is not registered with the orchestrator".format(
                start_hash))
        if not self.snapshot_hash_registered(end_hash):
            raise OrchestratorError(
                "snapshot end_hash {} is not registered with the orchestrator".format(
                end_hash))

        # if they both are registered register the segment
        self._segments.add((start_hash, end_hash))

    def register_mutation(self, start_hash, end_hash):

        # check that the hashes are for apparatuss in the orchestrator
        # if one is not registered raise an error
        if not self.apparatus_hash_registered(start_hash):
            raise OrchestratorError(
                "apparatus start_hash {} is not registered with the orchestrator".format(
                start_hash))
        if not self.apparatus_hash_registered(end_hash):
            raise OrchestratorError(
                "apparatus end_hash {} is not registered with the orchestrator".format(
                end_hash))

        # if the hashes are the same there is no mutation so we don't
        # add it
        if start_hash == end_hash:
            raise OrchestratorError("The hashes are the same, so no mutation detected")

        self._mutations.add((start_hash, end_hash))

    def save_segment(self, start_snapshot_hash, end_snapshot):

        # add the snapshot
        end_hash = self.add_snapshot(end_snapshot)

        # register it as a segment
        self.register_segment(start_snapshot_hash, end_hash)

        # get the hashes of the apparatuses to check if there was a mutation
        start_apparatus_hash = self.hash_apparatus(self.get_snapshot(start_snapshot_hash).apparatus)
        end_apparatus_hash = self.hash_apparatus(end_snapshot.apparatus)

        # register the mutation if it has mutated
        if start_apparatus_hash != end_apparatus_hash:
            self.register_mutation(start_apparatus_hash, end_apparatus_hash)

        return end_hash

    def new_run_by_time(self, init_walkers, run_time, n_steps,
                        apparatus_hash=None, configuration=None,
                        checkpoint_freq=None,
                        checkpoint_dir=None):
        """Start a new run that will go for a certain amount of time given a
        new set of initial conditions. """

        # make the starting snapshot from the walkers
        start_hash = self.gen_start_snapshot(init_walkers,
                                             apparatus_hash=apparatus_hash)

        # then perform a run with the checkpoints etc using the
        # dedicated method that works on snapshots
        return self.run_snapshot_by_time(start_hash, run_time, n_steps,
                                         configuration=configuration,
                                         checkpoint_freq=checkpoint_freq,
                                         checkpoint_dir=checkpoint_dir)


    def run_snapshot_by_time(self, snapshot_hash, run_time, n_steps,
                             checkpoint_freq=None,
                             checkpoint_dir=None,
                             configuration=None,
                             mode=None):
        """For a finished run continue it but resetting all the state of the
        resampler and boundary conditions"""

        # check that the directory for checkpoints exists, and create
        # it if it doesn't and isn't already created
        if checkpoint_dir is not None:
            checkpoint_dir = osp.realpath(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

        if mode is None:
            mode = self.DEFAULT_ORCHESTRATION_MODE

        start_time = time.time()

        # get the snapshot
        snapshot = self.get_snapshot(snapshot_hash)

        # the initial apparatus hash
        initial_apparatus_hash = self.hash_apparatus(snapshot.apparatus)

        # generate the simulation manager given the snapshot and the
        # configuration
        sim_manager = self.gen_sim_manager(snapshot, configuration=configuration)

        # run the init subroutine
        sim_manager.init()

        # keep a running list of the checkpoints for this run
        run_checkpoint_hashes = []

        # run each cycle manually creating checkpoints when necessary
        walkers = snapshot.walkers
        cycle_idx = 0
        while time.time() - start_time < run_time:

            # run the cycle
            walkers, filters = sim_manager.run_cycle(walkers, n_steps, cycle_idx)

            # check to see if a checkpoint is necessary
            if (checkpoint_freq is not None):
                if (cycle_idx % checkpoint_freq == 0):

                    # make the checkpoint snapshot
                    checkpoint_snapshot = SimSnapshot(walkers, SimApparatus(filters))

                    # save the checkpoint
                    checkpoint_hash = self.save_segment(snapshot_hash, checkpoint_snapshot)

                    # add the checkpoint to the list of checkpoints for this run
                    run_checkpoint_hashes.append(checkpoint_hash)

                    if checkpoint_dir is not None:

                        # construct the checkpoint filename from the
                        # template using the hashes for the start and the checkpoint
                        checkpoint_filename = self.CHECKPOINT_FILENAME_TEMPLATE.format(
                            run_start_hash=snapshot_hash,
                            checkpoint_hash=checkpoint_hash)

                        checkpoint_path = osp.join(checkpoint_dir, checkpoint_filename)

                        # write out the pickle to the file
                        with open(checkpoint_path, mode) as wf:
                            pickle.dump(checkpoint_snapshot, wf)

            cycle_idx += 1

        # run the segment given the sim manager and run parameters
        end_snapshot = SimSnapshot(walkers, SimApparatus(filters))

        # run the cleanup subroutine
        sim_manager.cleanup()

        # then we get the right stuff to register the segment, run,
        # and mutations

        # add the mutated apparatus
        mutated_apparatus_hash = self.add_apparatus(end_snapshot.apparatus)

        # register the mutation if it has mutated
        if initial_apparatus_hash != mutated_apparatus_hash:
            self.register_mutation(initial_apparatus_hash, mutated_apparatus_hash)

        # add the snapshot and the run for it
        end_hash = self.add_snapshot(end_snapshot)

        self.register_run(snapshot_hash, end_hash, checkpoints=run_checkpoint_hashes)

        return (snapshot_hash, end_hash), (initial_apparatus_hash, mutated_apparatus_hash)

    def orchestrate_snapshot_run_by_time(self, snapshot_hash, run_time, n_steps,
                                         apparatus_hash=None,
                                         checkpoint_freq=None,
                                         checkpoint_dir=None,
                                         orchestrator_path=None,
                                         configuration=None,
                                         mode=None):

        if mode is None:
            mode = self.DEFAULT_ORCHESTRATION_MODE

        run_tup, mutation_tup = self.run_snapshot_by_time(snapshot_hash, run_time, n_steps,
                                         checkpoint_freq=checkpoint_freq,
                                         checkpoint_dir=checkpoint_dir,
                                         configuration=configuration)

        # then pickle thineself
        with open(orchestrator_path, mode) as wf:
            pickle.dump(self, wf)

        return run_tup, mutation_tup

    def orchestrate_run_by_time(self, init_walkers, run_time, n_steps,
                                # non-default orchestration components
                                apparatus_hash=None,
                                checkpoint_freq=None,
                                checkpoint_dir=None,
                                orchestrator_path=None,
                                configuration=None,
                                # these can reparametrize the paths
                                # for both the orchestrator produced
                                # files as well as the configuration
                                work_dir=None,
                                config_name=None,
                                narration=None,
                                mode=None,
                                # extra kwargs will be passed to the
                                # configuration.reparametrize method
                                **kwargs):

        # there are two possible uses for the path reparametrizations:
        # the configuration and the orchestrator file paths. If both
        # of those are explicitly specified by passing in the whole
        # configuration object or both of checkpoint_dir,
        # orchestrator_path then those reparametrization kwargs will
        # not be used. As this is likely not the intention of the user
        # we will raise an error. If there is even one use for them no
        # error will be raised.

        # first check if any reparametrizations were even requested
        parametrizations_requested = (True if work_dir is not None else False,
                                      True if config_name is not None else False,
                                      True if narration is not None else False,
                                      True if mode is not None else False)

        # check if there are any available targets for reparametrization
        reparametrization_targets = (True if configuration is None else False,
                                     True if checkpoint_dir is None else False,
                                     True if orchestrator_path is None else False)

        # if paramatrizations were requested and there are no targets
        # we need to raise an error
        if any(parametrizations_requested) and not any(reparametrization_targets):

            raise OrchestratorError("Reparametrizations were requested but none are possible,"
                                    " due to all possible targets being already explicitly given")


        # if any paths were not given and no defaults for path
        # parameters we want to fill in the defaults for them. This
        # will also fill in any missing parametrizations with defaults

        # we do this by just setting the path parameters if they
        # aren't set, then later the parametrization targets will be
        # tested for if they have been set or not, and if they haven't
        # then these will be used to generate paths for them.
        if work_dir is None:
            work_dir = self.DEFAULT_WORKDIR
        if config_name is None:
            config_name = self.DEFAULT_CONFIG_NAME
        if narration is None:
            narration = self.DEFAULT_NARRATION
        if mode is None:
            mode = self.DEFAULT_MODE

        # if no configuration was specified use the default one
        if configuration is None:
            configuration = self.default_configuration

            # reparametrize the configuration with the given path
            # parameters and anything else in kwargs. If they are none
            # this will have no effect anyhow
            configuration = configuration.reparametrize(work_dir=work_dir,
                                                        config_name=config_name,
                                                        narration=narration,
                                                        mode=mode,
                                                        **kwargs)

        # make parametric paths for the checkpoint directory and the
        # orchestrator pickle to be made, unless they are explicitly given

        if checkpoint_dir is None:

            # the checkpoint directory will be a directory called
            # 'checkpoints' that will be placed in the work dir
            checkpoint_dir = osp.join(work_dir, self.DEFAULT_CHECKPOINT_DIR)

        if orchestrator_path is None:

            # the orchestrator pickle will be of similar form to the
            # reporters having the config name, and narration if
            # given, and an identifying compound file extension
            orch_narration = "_{}".format(narration) if len(narration) > 0 else ""
            orch_filename = self.ORCH_FILENAME_TEMPLATE.format(config=config_name,
                                                               narration=orch_narration)
            orchestrator_path = osp.join(work_dir, orch_filename)



        # make the starting snapshot from the walkers
        start_hash = self.gen_start_snapshot(init_walkers,
                                             apparatus_hash=apparatus_hash)

        # for the pickles in the orchestrator we want to use bytes
        # mode for the pickles because that will be faster and smaller
        # so unless bytes is already specified in the mode we will add it
        if 'b' not in mode:
            orch_mode = mode + 'b'

        # orchestrate from the snapshot
        return self.orchestrate_snapshot_run_by_time(start_hash, run_time, n_steps,
                                                     checkpoint_freq=checkpoint_freq,
                                                     checkpoint_dir=checkpoint_dir,
                                                     orchestrator_path=orchestrator_path,
                                                     configuration=configuration,
                                                     mode=orch_mode)


    def restart_snapshot(self, snapshot_hash):
        """For a finished run continue it and don't reset the state of the
        resampler and boundary conditions."""

        pass

    def recover_run(self, run_id):
        """For a run that ended in a bad state recover it and restart it. """

        pass

    def snapshot_graph(self):
        """Return a NetworkX graph of the segments """
        pass

    def run_graph(self):
        """Return a NetworkX graph of the runs """
        pass

    def mutation_graph(self):
        """Return a NetworkX graph of the apparatus mutations """
        pass


def reconcile_orchestrators(orch_a, orch_b):

    # check that the hashes of the two apparatuses are the same
    if orch_a.default_apparatus_hash != orch_b.default_apparatus_hash:
        raise ValueError("orchestrator_a and orchestrator_b do not have the same default apparatus")

    # make a new orchestrator
    new_orch = Orchestrator(orch_a.default_apparatus,
                            default_configuration=orch_a.default_configuration)

    # add in all snapshots from each orchestrator
    for snapshot in orch_a.snapshots + orch_b.snapshots:
        snaphash = new_orch.add_snapshot(snapshot)

    # register all the segments in each
    for segment in list(orch_a.segments) + list(orch_b.segments):
        new_orch.register_segment(*segment)

    # register all the runs in each
    for run in list(orch_a.runs) + list(orch_b.runs):
        new_orch.register_run(*run)

    # register all the mutations in each
    for mutation in list(orch_a.mutations) + list(orch_b.mutations):
        new_orch.register_mutation(*mutation)

    return new_orch




def orchestrate_run_by_time(orchestrator_pkl_path, start_hash,
                            run_time, n_steps,
                            apparatus_hash=None, configuration=None,
                            checkpoint_freq=None,
                            checkpoint_dir=None,
                            orchestrator_path=None,
                            n_workers=None):

    # load the orchestrator
    with open(orchestrator_pkl_path, 'rb') as rf:
        orchestrator = pickle.load(rf)


    # run the snapshot
    return orchestrator.orchestrate_snapshot_run_by_time(start_hash, run_time, n_steps,
                                                         apparatus_hash=apparatus_hash,
                                                         configuration=configuration,
                                                         checkpoint_freq=checkpoint_freq,
                                                         checkpoint_dir=checkpoint_dir,
                                                         orchestrator_path=orchestrator_path,
                                                         n_workers=n_workers)


if __name__ == "__main__":

    # orchestrate a simulation from an orchestrator pickle
    pass
