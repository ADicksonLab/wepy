from copy import copy, deepcopy
from hashlib import md5
from warnings import warn
import time
import os
import os.path as osp
from base64 import b64encode, b64decode
from zlib import compress, decompress

# instead of pickle we use dill
import dill

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

    # we freeze the pickle protocol for making hashes, because we care
    # more about stability than efficiency of newer versions
    HASH_PICKLE_PROTOCOL = 2

    DEFAULT_WORKDIR = Configuration.DEFAULT_WORKDIR
    DEFAULT_CONFIG_NAME = Configuration.DEFAULT_CONFIG_NAME
    DEFAULT_NARRATION = Configuration.DEFAULT_NARRATION
    DEFAULT_MODE = Configuration.DEFAULT_MODE

    CHECKPOINT_FILENAME_TEMPLATE = "{run_start_hash}_{checkpoint_hash}.chk.pkl"
    DEFAULT_CHECKPOINT_DIR = 'checkpoints'
    ORCH_FILENAME_TEMPLATE = "{config}{narration}.orch"
    DEFAULT_ORCHESTRATION_MODE = 'xb'

    def __init__(self, sim_apparatus,
                 default_init_walkers=None,
                 default_configuration=None,
                 default_work_dir=None):
        # the main dictionary of snapshots keyed by their hashes
        self._snapshots = {}

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

        # the apparatus for the simulation. This is the configuration
        # and initial conditions independent components necessary for
        # running a simulation. The primary (and only necessary)
        # component is the runner. The other components when passed
        # here are used as the defaults when unspecified later
        # (e.g. resampler and boundary conditions)
        self._apparatus = deepcopy(sim_apparatus)

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

        # if initial walkers were given we save them and also make a
        # snapshot for them
        if default_init_walkers is not None:

            self._init_walkers = default_init_walkers

            self._start_hash = self.gen_start_snapshot(self._init_walkers)

    def serialize(self):

        serial_str = dill.dumps(self, recurse=True)
        return serial_str

    @classmethod
    def deserialize(cls, serial_str):

        orch = dill.loads(serial_str)
        return orch

    @classmethod
    def load(cls, filepath, mode='rb'):

        with open(filepath, mode) as rf:
            orch = cls.deserialize(rf.read())

        return orch


    def dump(self, filepath, mode=None):

        if mode is None:
            mode = self.DEFAULT_ORCHESTRATION_MODE

        with open(filepath, mode) as wf:
            wf.write(self.serialize())

    @classmethod
    def encode(cls, obj):

        # used for both snapshots and apparatuses even though they
        # themselves have different methods in the API

        # we use dill to dump to a string and we always do a deepcopy
        # of the object to avoid differences in the resulting pickle
        # object from having multiple references, then we encode in 64 bit
        return b64encode(compress(dill.dumps(deepcopy(obj),
                                             protocol=cls.HASH_PICKLE_PROTOCOL,
                                             recurse=True)))

    @classmethod
    def decode(cls, encoded_str):

        return dill.loads(decompress(b64decode(encoded_str)))

    @classmethod
    def hash(cls, serial_str):
        return md5(serial_str).hexdigest()

    def serialize_snapshot(self, snapshot):
        return self.encode(snapshot)

    def hash_snapshot(self, snapshot):

        serialized_snapshot = self.serialize_snapshot(snapshot)
        return self.hash(serialized_snapshot)

    def get_snapshot(self, snapshot_hash):
        """Returns a copy of a snapshot."""

        return deepcopy(self._snapshots[snapshot_hash])

    @property
    def snapshots(self):
        return deepcopy(list(self._snapshots.values()))

    @property
    def snapshot_hashes(self):
        return list(self._snapshots.keys())

    @property
    def default_snapshot_hash(self):
        return self._start_hash

    @property
    def default_snapshot(self):
        return self.get_snapshot(self.default_snapshot_hash)

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

    def snapshot_hash_registered(self, snapshot_hash):

        if any([True if snapshot_hash == h else False for h in self.snapshot_hashes]):
            return True
        else:
            return False

    @property
    def segments(self):
        return deepcopy(self._segments)

    @property
    def runs(self):
        return list(deepcopy(self._runs))

    def get_run_checkpoints(self, run_id):
        return deepcopy(self._run_checkpoints[run_id])

    def _add_snapshot(self, snaphash, snapshot):

        # check that the hash is not already in the snapshots
        if any([True if snaphash == md5 else False for md5 in self.snapshot_hashes]):

            # just skip the rest of the function and return the hash
            return snaphash

        self._snapshots[snaphash] = snapshot

        return snaphash


    def add_snapshot(self, snapshot):

        # copy the snapshot
        snapshot = deepcopy(snapshot)

        # get the hash of the snapshot
        snaphash = self.hash_snapshot(snapshot)

        return self._add_snapshot(snaphash, snapshot)

    def gen_start_snapshot(self, init_walkers):

        # make a SimSnapshot object using the initial walkers and
        start_snapshot = SimSnapshot(init_walkers, self.default_apparatus)

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

    def save_segment(self, start_snapshot_hash, end_snapshot):

        # add the snapshot
        end_hash = self.add_snapshot(end_snapshot)

        # register it as a segment
        self.register_segment(start_snapshot_hash, end_hash)

        return end_hash

    def new_run_by_time(self, init_walkers, run_time, n_steps,
                        configuration=None,
                        checkpoint_freq=None,
                        checkpoint_dir=None):
        """Start a new run that will go for a certain amount of time given a
        new set of initial conditions. """

        # make the starting snapshot from the walkers
        start_hash = self.gen_start_snapshot(init_walkers)

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
            mode = self.DEFAULT_MODE

        if 'b' not in mode:
            dump_mode = mode + 'b'

        start_time = time.time()

        # get the snapshot
        snapshot = self.get_snapshot(snapshot_hash)

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
                        with open(checkpoint_path, dump_mode) as wf:
                            wf.write(self.serialize_snapshot(checkpoint_snapshot))

            cycle_idx += 1

        # run the segment given the sim manager and run parameters
        end_snapshot = SimSnapshot(walkers, SimApparatus(filters))

        # run the cleanup subroutine
        sim_manager.cleanup()

        # add the snapshot and the run for it
        end_hash = self.add_snapshot(end_snapshot)

        self.register_run(snapshot_hash, end_hash, checkpoints=run_checkpoint_hashes)

        return (snapshot_hash, end_hash)

    def orchestrate_snapshot_run_by_time(self, snapshot_hash, run_time, n_steps,
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


        # for writing the orchestration files we set the default mode
        # if mode is not given
        if mode is None:
            # the orchestrator mode is used for pickling the
            # orchestrator and so must be in bytes mode
            orch_mode = self.DEFAULT_ORCHESTRATION_MODE

        elif 'b' not in mode:
            # add a bytes to the end of the mode for the orchestrator pickleization
            mode = mode + 'b'

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




        run_tup = self.run_snapshot_by_time(snapshot_hash, run_time, n_steps,
                                            checkpoint_freq=checkpoint_freq,
                                            checkpoint_dir=checkpoint_dir,
                                            configuration=configuration,
                                            mode=mode)

        # then serialize thineself
        self.dump(orchestrator_path, mode=orch_mode)

        return run_tup

    def orchestrate_run_by_time(self, init_walkers, run_time, n_steps,
                                **kwargs):


        # make the starting snapshot from the walkers and the
        # apparatus if given, otherwise the default will be used
        start_hash = self.gen_start_snapshot(init_walkers)


        # orchestrate from the snapshot
        return self.orchestrate_snapshot_run_by_time(start_hash, run_time, n_steps,
                                                     **kwargs)


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


def serialize_orchestrator(orchestrator):

    return orchestrator.serialize()

def deserialize_orchestrator(serial_str):

    return Orchestrator.deserialize(serial_str)

def dump_orchestrator(orchestrator, filepath, mode='wb'):

    orchestrator.dump(filepath, mode=mode)

def load_orchestrator(filepath, mode='rb'):

    return Orchestrator.load(filepath, mode=mode)

def encode(obj):

    return Orchestrator.encode(obj)

def decode(encoded_str):

    return Orchestrator.decode(encoded_str)

def reconcile_orchestrators(orch_a, orch_b):

    # check that the hashes of the two apparatuses are the same
    if orch_a.default_apparatus_hash != orch_b.default_apparatus_hash:
        raise ValueError("orchestrator_a and orchestrator_b do not have the same default apparatus")

    # make a new orchestrator
    new_orch = Orchestrator(orch_a.default_apparatus,
                            default_configuration=orch_a.default_configuration)

    # add in all snapshots from each orchestrator, by the hash not the
    # snapshots themselves
    for snaphash in orch_a.snapshot_hashes:
        snapshot = orch_a.get_snapshot(snaphash)
        new_orch._add_snapshot(snaphash, snapshot)

    for snaphash in orch_b.snapshot_hashes:
        snapshot = orch_b.get_snapshot(snaphash)
        new_orch._add_snapshot(snaphash, snapshot)


    # register all the segments in each
    for segment in list(orch_a.segments) + list(orch_b.segments):
        new_orch.register_segment(*segment)

    # register all the runs in each
    for run in list(orch_a.runs) + list(orch_b.runs):
        new_orch.register_run(*run)

    return new_orch


if __name__ == "__main__":

    pass
