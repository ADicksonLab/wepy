from copy import copy, deepcopy
import sqlite3
from hashlib import md5
import time
import os
import os.path as osp
from base64 import b64encode, b64decode
from zlib import compress, decompress
import itertools as it
import logging

# instead of pickle we use dill, so we can save dynamically defined
# classes
import dill

from wepy.sim_manager import Manager
from wepy.orchestration.configuration import Configuration
from wepy.util.kv import KV

class SimApparatus():
    """The simulation apparatus are the components needed for running a
    simulation without the initial conditions for starting the simulation.
    
    A runner is strictly necessary but a resampler and boundary
    conditions are not.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, filters):
        self._filters = deepcopy(filters)

    @property
    def filters(self):
        """ """
        return self._filters



class WepySimApparatus(SimApparatus):
    """ """

    def __init__(self, runner, resampler=None, boundary_conditions=None):

        # add them in the order they are done in Wepy
        filters = [runner]
        if boundary_conditions is not None:
            filters.append(boundary_conditions)
        if resampler is not None:
            filters.append(resampler)

        super().__init__(filters)

class SimSnapshot():
    """ """

    def __init__(self, walkers, apparatus):

        self._walkers = deepcopy(walkers)
        self._apparatus = deepcopy(apparatus)

    @property
    def walkers(self):
        """ """
        return self._walkers

    @property
    def apparatus(self):
        """ """
        return self._apparatus



class OrchestratorError(Exception):
    """ """
    pass



class Orchestrator():
    """ """

    # we freeze the pickle protocol for making hashes, because we care
    # more about stability than efficiency of newer versions
    HASH_PICKLE_PROTOCOL = 3

    DEFAULT_WORKDIR = Configuration.DEFAULT_WORKDIR
    DEFAULT_CONFIG_NAME = Configuration.DEFAULT_CONFIG_NAME
    DEFAULT_NARRATION = Configuration.DEFAULT_NARRATION
    DEFAULT_MODE = Configuration.DEFAULT_MODE

    DEFAULT_CHECKPOINT_FILENAME = "checkpoint.chk"
    ORCH_FILENAME_TEMPLATE = "{config}{narration}.orch"

    # the default way to oepn up the whole parent database
    DEFAULT_ORCHESTRATION_MODE = 'x'

    # mode to open the individual kv stores on the parent database
    KV_MODE = 'r+'

    # default timeout for connecting to a database
    SQLITE3_DEFAULT_TIMEOUT = 5

    # the fields to return (and their order) as a record for a run
    # query
    RUN_SELECT_FIELDS = ('last_cycle_idx', 'config_hash')

    def __init__(self, orch_path=":memory:",
                 mode='x',
    ):

        self.orch_path = orch_path
        self._mode = mode

        # initialize or open each of the separate KV-stores (tables in
        # the same SQLite3 database)

        # we open the first one with the create mode and let the KV
        # module handle the opening and creation if necessary

        # metadata: default init walkers, default apparatus, default
        # configuration
        self.metadata_kv = KV(db_url=self.orch_path,
                              table='meta',
                              mode=mode,
                              value_types=None)

        # snapshots
        self.snapshot_kv = KV(db_url=self.orch_path,
                              table='snapshots',
                              primary_key='snaphash',
                              value_name='snapshot',
                              mode=self.KV_MODE)

        # configurations
        self.configuration_kv = KV(db_url=self.orch_path,
                                   table='configurations',
                                   primary_key='config_hash',
                                   value_name='config',
                                   mode=self.KV_MODE)

        # run table: start_hash, end_hash, num_cycles, configuration_id

        # get a raw connection to the database
        self._db = sqlite3.connect(self.orch_path, timeout=self.SQLITE3_DEFAULT_TIMEOUT)
        self._db.isolation_level = None

        self._closed = False

        # we make a table for the run data

        create_run_table_query = """
            CREATE TABLE IF NOT EXISTS runs
             (start_hash TEXT NOT NULL,
             end_hash TEXT NOT NULL,
             config_hash NOT NULL,
             last_cycle_idx INTEGER NOT NULL,
             PRIMARY KEY (start_hash, end_hash))

        """

        # do the create as a transaction
        self._db.cursor().execute('BEGIN IMMEDIATE TRANSACTION')
        # run the query
        self._db.cursor().execute(create_run_table_query)
        self._db.cursor().execute('COMMIT')

    def close(self):

        if self._closed == True:
            raise IOError("The database connection is already closed")

        else:
            # close all the connections
            self.metadata_kv.close()
            self.configuration_kv.close()
            self.snapshot_kv.close()
            self._db.close()
            self._closed = True


    @classmethod
    def serialize(cls, snapshot):
        """Serialize a snapshot to a compressed, encoded, pickle string
        representation.

        Currently uses the dill module for pickling because the base
        pickle module is inadequate. However, it is mostly compatible
        and can be read natively with pickle but this usage is
        officially not supported. Instead use the deserialize_snapshot.

        Also compresses with default zlib compression and is encoded
        in base64.

        The object will always have a deepcopy performed on it so that
        all of the extraneous references to it are avoided since there
        is no (AFAIK) way to make sure all references to an object are
        deleted.

        NOTE: Perhaps there is a way and that should be done (and
        tested) to see if it provides stable pickles (i.e. pickles
        that always hash to the same value). To avoid the overhead of
        copying large objects.

        Parameters
        ----------
        snapshot : SimSnapshot object
            The snapshot of the simulation you want to serialize.

        Returns
        -------
        serial_str : str
            Serialized string of the snapshot object

        """

        serial_str = b64encode(
                        compress(
                            dill.dumps(
                                deepcopy(snapshot),
                                protocol=cls.HASH_PICKLE_PROTOCOL,
                                recurse=True)
                        )
                     )

        return serial_str


    # core methods for serializing python objects, used for snapshots,
    # apparatuses, configurations, and the initial walker list

    @classmethod
    def deserialize(cls, serial_str):
        """Deserialize an unencoded string snapshot to an object.

        Parameters
        ----------
        serial_str : str
            Serialized string of the snapshot object

        Returns
        -------
        snapshot : SimSnapshot object
            Simulation snapshot object

        """

        return dill.loads(decompress(b64decode(serial_str)))


    # defaults getters and setters
    def set_default_sim_apparatus(self, sim_apparatus):

        # serialize the apparatus and then set it
        serial_app = self.serialize(sim_apparatus)

        with self.metadata_kv.lock():
            self.metadata_kv['default_sim_apparatus'] = serial_app

    def set_default_init_walkers(self, init_walkers):

        # serialize the apparatus and then set it
        serial_walkers = self.serialize(init_walkers)

        with self.metadata_kv.lock():
            self.metadata_kv['default_init_walkers'] = serial_walkers

    def set_default_configuration(self, configuration):

        # serialize the apparatus and then set it
        serial_config = self.serialize(configuration)

        config_hash = self.hash_snapshot(serial_config)

        with self.metadata_kv.lock():
            self.metadata_kv['default_configuration_hash'] = config_hash

        with self.configuration_kv.lock():
            self.configuration_kv[config_hash] = serial_config

    def set_default_snapshot(self, snapshot):

        snaphash = self.add_snapshot(snapshot)

        # then save the hash in the metadata
        with self.metadata_kv.lock():
            self.metadata_kv['default_snapshot_hash'] = snaphash

        return snaphash

    def gen_default_snapshot(self):

        # generate the snapshot
        sim_start_hash = self.gen_start_snapshot(self.get_default_init_walkers())

        # then save the hash in the metadata
        with self.metadata_kv.lock():
            self.metadata_kv['default_snapshot_hash'] = sim_start_hash

        return sim_start_hash


    def get_default_sim_apparatus(self):

        return self.deserialize(self.metadata_kv['default_sim_apparatus'])

    def get_default_init_walkers(self):

        return self.deserialize(self.metadata_kv['default_init_walkers'])

    def get_default_configuration(self):

        config_hash = self.metadata_kv['default_configuration_hash']

        return self.get_configuration(config_hash)

    def get_default_configuration_hash(self):

        return self.metadata_kv['default_configuration_hash']


    def get_default_snapshot(self):

        start_hash = self.metadata_kv['default_snapshot_hash']

        return self.get_snapshot(start_hash)

    def get_default_snapshot_hash(self):

        return self.metadata_kv['default_snapshot_hash']


    @classmethod
    def hash_snapshot(cls, serial_str):
        """

        Parameters
        ----------
        serial_str :
            

        Returns
        -------

        """
        return md5(serial_str).hexdigest()


    def get_snapshot(self, snapshot_hash):
        """Returns a copy of a snapshot.

        Parameters
        ----------
        snapshot_hash :
            

        Returns
        -------

        """

        return self.deserialize(self.snapshot_kv[snapshot_hash])


    def get_configuration(self, config_hash):
        """Returns a copy of a snapshot.

        Parameters
        ----------
        config_hash :
            

        Returns
        -------

        """

        return self.deserialize(self.configuration_kv[config_hash])


    @property
    def snapshot_hashes(self):
        """ """

        # iterate over the snapshot kv
        return list(self.snapshot_kv.keys())

    @property
    def config_hashes(self):
        """ """

        # iterate over the snapshot kv
        return list(self.config_kv.keys())

    @property
    def configuration_hashes(self):
        """ """

        # iterate over the snapshot kv
        return list(self.configuration_kv.keys())


    def add_snapshot(self, snapshot):
        """

        Parameters
        ----------
        snapshot :

        Returns
        -------

        """

        # serialize the snapshot using the protocol for doing so
        serialized_snapshot = self.serialize(snapshot)

        # get the hash of the snapshot
        snaphash = self.hash_snapshot(serialized_snapshot)

        # check that the hash is not already in the snapshots
        if any([True if snaphash == md5 else False for md5 in self.snapshot_hashes]):

            # just skip the rest of the function and return the hash
            return snaphash

        # save the snapshot in the KV store
        with self.snapshot_kv.lock():
            self.snapshot_kv[snaphash] = serialized_snapshot

        return snaphash

    def gen_start_snapshot(self, init_walkers):
        """

        Parameters
        ----------
        init_walkers :
            

        Returns
        -------

        """

        # make a SimSnapshot object using the initial walkers and
        start_snapshot = SimSnapshot(init_walkers, self.get_default_sim_apparatus())

        # save the snapshot, and generate its hash
        sim_start_md5 = self.add_snapshot(start_snapshot)

        return sim_start_md5

    @property
    def default_snapshot_hash(self):
        """ """
        return self.metadata_kv['default_snapshot_hash']

    @property
    def default_snapshot(self):
        """ """
        return self.get_snapshot(self.default_snapshot_hash)

    def snapshot_registered(self, snapshot):
        """Check whether a snapshot is already in the database, based on the
        hash of it.

        This serializes the snapshot so may be slow.

        Parameters
        ----------
        snapshot : SimSnapshot object
            The snapshot object you want to query for.

        Returns
        -------

        """

        # serialize and hash the snapshot
        snaphash = self.hash_snapshot(self.serialize(snapshot))

        # then check it
        return self.snapshot_hash_registered(snaphash)

    def snapshot_hash_registered(self, snapshot_hash):
        """Check whether a snapshot hash is already in the database.

        Parameters
        ----------
        snapshot_hash : str
            The string hash of the snapshot.

        Returns
        -------

        """

        if any([True if snapshot_hash == h else False for h in self.snapshot_hashes]):
            return True
        else:
            return False


    def configuration_hash_registered(self, config_hash):
        """Check whether a snapshot hash is already in the database.

        Parameters
        ----------
        snapshot_hash : str
            The string hash of the snapshot.

        Returns
        -------

        """

        if any([True if config_hash == h else False for h in self.configuration_hashes]):
            return True
        else:
            return False


    ### run methods

    def add_configuration(self, configuration):

        serialized_config = self.serialize(configuration)

        config_hash = self.hash_snapshot(serialized_config)

        # check that the hash is not already in the snapshots
        if any([True if config_hash == md5 else False for md5 in self.configuration_hashes]):

            # just skip the rest of the function and return the hash
            return config_hash

        # save the snapshot in the KV store
        with self.configuration_kv.lock():
            self.configuration_kv[config_hash] = serialized_config

        return config_hash

    def _add_run_record(self, start_hash, end_hash, configuration_hash, cycle_idx):

        add_run_row_query = """
        INSERT INTO runs (start_hash, end_hash, config_hash, last_cycle_idx)
        VALUES (?, ?, ?, ?)
        """

        params = (start_hash, end_hash, configuration_hash, cycle_idx)

        # do it as a transaction
        self._db.cursor().execute('BEGIN IMMEDIATE TRANSACTION')

        # run the insert
        self._db.cursor().execute(add_run_row_query, params)

        self._db.cursor().execute('COMMIT')


    def register_run(self, start_hash, end_hash, config_hash, cycle_idx):
        """

        Parameters
        ----------
        start_hash :
            
        end_hash :
            
        config_hash :
            
        cycle_idx : int
            The cycle of the simulation run the checkpoint was generated for.

        Returns
        -------

        """

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

        if not self.configuration_hash_registered(config_hash):
            raise OrchestratorError(
                "config hash {} is not registered with the orchestrator".format(
                config_hash))


        # save the configuration and get it's id

        self._add_run_record(start_hash, end_hash, config_hash, cycle_idx)

    def get_run_records(self):

        get_run_record_query = """
        SELECT *
        FROM runs
        """.format(fields=', '.join(self.RUN_SELECT_FIELDS))

        cursor = self._db.cursor()
        cursor.execute(get_run_record_query)
        records = cursor.fetchall()

        return records

    def get_run_record(self, start_hash, end_hash):

        get_run_record_query = """
        SELECT {fields}
        FROM runs
        WHERE start_hash=? AND end_hash=?
        """.format(fields=', '.join(self.RUN_SELECT_FIELDS))

        params = (start_hash, end_hash)

        cursor = self._db.cursor()
        cursor.execute(get_run_record_query, params)
        record = cursor.fetchone()

        return record

    def run_last_cycle_idx(self, start_hash, end_hash):

        record = self.get_run_record(start_hash, end_hash)

        last_cycle_idx = record[self.RUN_SELECT_FIELDS.index('last_cycle_idx')]

        return last_cycle_idx


    def run_configuration(self, start_hash, end_hash):

        record = self.get_run_record(start_hash, end_hash)

        config_hash = record[self.RUN_SELECT_FIELDS.index('config_hash')]

        # get the configuration object and deserialize it
        return self.deserialize(self.configuration_kv[config_hash])


    def run_hashes(self):

        return [(rec[0], rec[1]) for rec in self.get_run_records()]

    def run_continues(self, start_hash, end_hash):
        """

        Parameters
        ----------
        start_hash :
            
        end_hash :
            

        Returns
        -------
        type
            

        """

        # loop through the runs in this orchestrator until we find one
        # where the start_hash matches the end hash
        runs = self.run_hashes()
        run_idx = 0
        while True:

            run_start_hash, run_end_hash = runs[run_idx]

            # if the start hash of the queried run is the same as the
            # end hash for this run we have found it
            if start_hash == run_end_hash:

                return (run_start_hash, run_end_hash)

            run_idx += 1

            # if the index is over the number of runs we quit and
            # return None as no match
            if run_idx >= len(runs):
                return None



    def _gen_checkpoint_orch(self, start_hash, checkpoint_snapshot, configuration, cycle_idx):
        """

        Parameters
        ----------
        start_hash :
            
        checkpoint_snapshot :
            
        configuration :
            
        cycle_idx : int
            The cycle of the simulation run the checkpoint was generated for.

        Returns
        -------

        """
        # make an orchestrator with the only run going from the start
        # hash snapshot to the checkpoint
        start_snapshot = self.get_snapshot(start_hash)
        checkpoint_orch = type(self)(start_snapshot.apparatus)

        # add the the starting snapshot to the orchestrator, we do
        # this the sneaky way because I am worried about hash
        # stability and we need to preserve the intial hash, so we
        # force the hash to be the start_hash and add the object regardless
        checkpoint_orch._add_snapshot(start_hash, deepcopy(start_snapshot))

        # then add a run to this checkpoint orchestrator by adding the
        # checkpoint snapshot and registering the run
        checkpoint_hash = checkpoint_orch.add_snapshot(checkpoint_snapshot)

        # register the run with the two hashes and the configuration
        checkpoint_orch.register_run(start_hash, checkpoint_hash, configuration, cycle_idx)

        return checkpoint_orch

    def _save_checkpoint(self, start_hash, checkpoint_snapshot, configuration,
                         checkpoint_dir, cycle_idx,
                         mode='wb'):
        """

        Parameters
        ----------
        start_hash :
            
        checkpoint_snapshot :
            
        configuration :
            
        checkpoint_dir :
            
        mode :
             (Default value = 'wb')

        Returns
        -------

        """

        if 'b' not in mode:
            mode = mode + 'b'

        # make a checkpoint object which is an orchestrator with only
        # 1 run in it which is the start and the checkpoint as its end
        checkpoint_orch = self._gen_checkpoint_orch(start_hash, checkpoint_snapshot,
                                                    configuration, cycle_idx)

        # construct the checkpoint filename from the template using
        # the hashes for the start and the checkpoint, we add a "new""
        # at the end of the file to indicate that it was just written
        # and if the other checkpoint is not removed you will be able
        # to tell them apart, this will get renamed without the new
        # once the other checkpoint is deleted successfully
        new_checkpoint_filename = self.DEFAULT_CHECKPOINT_FILENAME + "new"

        new_checkpoint_path = osp.join(checkpoint_dir, new_checkpoint_filename)

        # write out the pickle to the file
        with open(new_checkpoint_path, mode) as wf:
            wf.write(checkpoint_orch.serialize())

        # the path that the checkpoint should be existing
        checkpoint_path = osp.join(checkpoint_dir, self.DEFAULT_CHECKPOINT_FILENAME)

        # only after the writing is complete do we delete the old
        # checkpoint, if there are any to delete
        if osp.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # then rename the one with "new" at the end to the final path
        os.rename(new_checkpoint_path, checkpoint_path)

    def gen_sim_manager(self, snapshot_hash, configuration):
        """

        Parameters
        ----------
        start_snapshot :
            
        configuration :
            

        Returns
        -------

        """

        # copy the snapshot to use for the sim_manager
        start_snapshot = self.get_snapshot(snapshot_hash)

        # construct the sim manager, in a wepy specific way
        sim_manager = Manager(start_snapshot.walkers,
                              runner=start_snapshot.apparatus.filters[0],
                              boundary_conditions=start_snapshot.apparatus.filters[1],
                              resampler=start_snapshot.apparatus.filters[2],
                              # configuration options
                              work_mapper=configuration.work_mapper,
                              reporters=configuration.reporters)

        return sim_manager


    def run_snapshot_by_time(self, start_hash, run_time, n_steps,
                             checkpoint_freq=None,
                             checkpoint_dir=None,
                             configuration=None,
                             mode=None):
        """For a finished run continue it but resetting all the state of the
        resampler and boundary conditions

        Parameters
        ----------
        start_hash :
            
        run_time :
            
        n_steps :
            
        checkpoint_freq :
             (Default value = None)
        checkpoint_dir :
             (Default value = None)
        configuration :
             (Default value = None)
        mode :
             (Default value = None)

        Returns
        -------

        """

        # check that the directory for checkpoints exists, and create
        # it if it doesn't and isn't already created
        if checkpoint_dir is not None:
            checkpoint_dir = osp.realpath(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

        if mode is None:
            mode = self.DEFAULT_MODE

        dump_mode = mode
        if 'b' not in dump_mode:
            dump_mode = mode + 'b'

        if configuration is None:
            configuration = deepcopy(self.default_configuration)



        # get the snapshot
        start_snapshot = self.get_snapshot(start_hash)

        # generate the simulation manager given the snapshot and the
        # configuration
        sim_manager = self.gen_sim_manager(start_snapshot, configuration=configuration)

        # run the init subroutine
        sim_manager.init()

        # keep a running list of the checkpoints for this run
        self._curr_run_checkpoints = []

        # run each cycle manually creating checkpoints when necessary
        walkers = start_snapshot.walkers
        cycle_idx = 0
        start_time = time.time()
        while time.time() - start_time < run_time:

            # run the cycle
            walkers, filters = sim_manager.run_cycle(walkers, n_steps, cycle_idx)

            # check to see if a checkpoint is necessary
            if (checkpoint_freq is not None):
                if (cycle_idx % checkpoint_freq == 0):

                    # make the checkpoint snapshot
                    checkpoint_snapshot = SimSnapshot(walkers, SimApparatus(filters))

                    # save the checkpoint (however that is implemented)
                    self._save_checkpoint(start_hash, checkpoint_snapshot,
                                          configuration,
                                          checkpoint_dir,
                                          cycle_idx,
                                          mode=dump_mode)

            cycle_idx += 1

        # run the segment given the sim manager and run parameters
        end_snapshot = SimSnapshot(walkers, SimApparatus(filters))

        # run the cleanup subroutine
        sim_manager.cleanup()

        # add the snapshot and the run for it
        end_hash = self.add_snapshot(end_snapshot)

        self.register_run(start_hash, end_hash, configuration, cycle_idx)

        # clear the object variable for the current checkpoints
        del self._curr_run_checkpoints

        return start_hash, end_hash

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
        """

        Parameters
        ----------
        snapshot_hash :
            
        run_time :
            
        n_steps :
            
        checkpoint_freq :
             (Default value = None)
        checkpoint_dir :
             (Default value = None)
        orchestrator_path :
             (Default value = None)
        configuration :
             (Default value = None)
        # these can reparametrize the paths# for both the orchestrator produced# files as well as the configurationwork_dir :
             (Default value = None)
        config_name :
             (Default value = None)
        narration :
             (Default value = None)
        mode :
             (Default value = None)
        # extra kwargs will be passed to the# configuration.reparametrize method**kwargs :
            

        Returns
        -------

        """


        # for writing the orchestration files we set the default mode
        # if mode is not given
        if mode is None:
            # the orchestrator mode is used for pickling the
            # orchestrator and so must be in bytes mode
            orch_mode = self.DEFAULT_ORCHESTRATION_MODE

        elif 'b' not in mode:
            # add a bytes to the end of the mode for the orchestrator pickleization
            orch_mode = mode + 'b'
        else:
            orch_mode = mode

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
                                      True if mode is not None else False,)

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

            # the checkpoint directory will be in the work dir
            checkpoint_dir = work_dir

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



    # @classmethod
    # def load(cls, filepath, mode='rb'):
    #     """

    #     Parameters
    #     ----------
    #     filepath :

    #     mode :
    #          (Default value = 'rb')

    #     Returns
    #     -------

    #     """

    #     with open(filepath, mode) as rf:
    #         orch = cls.deserialize(rf.read())

    #     return orch


    # def dump(self, filepath, mode=None):
    #     """

    #     Parameters
    #     ----------
    #     filepath :
            
    #     mode :
    #          (Default value = None)

    #     Returns
    #     -------

    #     """

    #     if mode is None:
    #         mode = self.DEFAULT_ORCHESTRATION_MODE

    #     with open(filepath, mode) as wf:
    #         wf.write(self.serialize())


def reconcile_orchestrators(host_path, *orchestrator_paths):
    """

    Parameters
    ----------
    template_orchestrator :
        
    *orchestrators :
        

    Returns
    -------

    """

    if not osp.exists(host_path):
        assert len(orchestrator_paths) > 1, \
            "If the host path is a new orchestrator, must give at least 2 orchestrators to merge."

    # open the host orchestrator at the location which will have all
    # of the new things put into it from the other orchestrators. If
    # it doesn't already exist it will be created otherwise open
    # read-write.
    new_orch = Orchestrator(orch_path=host_path,
                            mode='a')

    # if this is an existing orchestrator copy the default
    # sim_apparatus and init_walkers
    try:
        default_app = new_orch.get_default_sim_apparatus()
    except KeyError:
        # no default apparatus, that is okay
        pass
    else:
        # set it
        new_orch.set_default_sim_apparatus(default_app)

    # same for the initial walkers
    try:
        default_walkers = new_orch.get_default_init_walkers()
    except KeyError:
        # no default apparatus, that is okay
        pass
    else:
        # set it
        new_orch.set_default_sim_apparatus(default_walkers)


    for orch_path in orchestrator_paths:

        # open it in read-write fail if doesn't exist
        orch = Orchestrator(orch_path=orch_path,
                            mode='r+')

        # add in all snapshots from each orchestrator, by the hash not the
        # snapshots themselves, we trust they are correct
        for snaphash in orch.snapshot_hashes:

            # check that the hash is not already in the snapshots
            if any([True if snaphash == md5 else False for md5 in new_orch.snapshot_hashes]):

                # skip it and move on
                continue

            # if it is not copy it over without deserializing
            with new_orch.snapshot_kv.lock():
                new_orch.snapshot_kv[snaphash] = orch.snapshot_kv[snaphash]

        # add in all the configuration from each orchestrator, by the
        # hash not the snapshots themselves, we trust they are correct
        for config_hash in orch.configuration_hashes:

            # check that the hash is not already in the snapshots
            if any([True if config_hash == md5 else False for md5 in new_orch.configuration_hashes]):

                # skip it and move on
                continue

            # if it is not set it
            with new_orch.configuration_kv.lock():
                new_orch.configuration_kv[config_hash] = orch.configuration_kv[config_hash]

        # concatenate the run table with an SQL union from an attached
        # database

        attached_table_name = "other"

        # query to attach the foreign database
        attach_query = """
        ATTACH '{}' AS {}
        """.format(orch_path, attached_table_name)

        # query to update the runs tabel with new unique runs
        union_query = """
        INSERT INTO runs
        SELECT * FROM (
        SELECT * FROM {}.runs
        EXCEPT
        SELECT * FROM runs
        )
        """.format(attached_table_name)

        # query to detach the table
        detach_query = """
        DETACH {}
        """.format(attached_table_name)

        # DEBUG
        print(attach_query)
        print(union_query)
        print(detach_query)

        # then run the queries

        c = new_orch._db.cursor()
        c.execute('BEGIN IMMEDIATE TRANSACTION')
        c.execute(attach_query)
        c.execute(union_query)
        c.execute('COMMIT')
        c.execute(detach_query)




    return new_orch

def recover_run_by_time(start_orch, checkpoint_orch,
                        run_time, n_steps,
                        **kwargs):
    """

    Parameters
    ----------
    start_orch :
        
    checkpoint_orch :
        
    run_time :
        
    n_steps :
        
    **kwargs :
        

    Returns
    -------

    """

    # reconcile the checkpoint orchestrator with the master the
    # original orchestrator, we put the original orch first so that it
    # preserves the defaults
    new_orch = reconcile_orchestrators(start_orch, checkpoint_orch)

    # now we need to get the hash of the checkpoint at the end of
    # the checkpoint orch to start from that, a checkpoint orch
    # should only have one run and the checkpoint will be the end
    # of that run.
    checkpoint_hash = checkpoint_orch.runs[0][-1]

    # then all we need to do is orchestrate from this checkpoint
    run_tup = new_orch.orchestrate_snapshot_run_by_time(checkpoint_hash, run_time, n_steps,
                                                        **kwargs)

    return new_orch, run_tup
