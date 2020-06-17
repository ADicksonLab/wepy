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
from wepy.orchestration.snapshot import SimApparatus, SimSnapshot

from wepy.util.kv import KV, SQLITE3_INMEMORY_URI, gen_uri


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

    DEFAULT_CHECKPOINT_FILENAME = "checkpoint.orch.sqlite"
    ORCH_FILENAME_TEMPLATE = "{config}{narration}.orch.sqlite"

    # the default way to oepn up the whole parent database
    DEFAULT_ORCHESTRATION_MODE = 'x'

    # mode to open the individual kv stores on the parent database
    KV_MODE = 'r+'

    # default timeout for connecting to a database
    SQLITE3_DEFAULT_TIMEOUT = 5

    # the fields to return (and their order) as a record for a run
    # query
    RUN_SELECT_FIELDS = ('last_cycle_idx', 'config_hash')

    def __init__(self, orch_path=None,
                 mode='x',
                 append_only=False,
    ):

        self._mode = mode
        self._append_only = append_only

        # handle the path and convert to a proper URI for the database
        # given the path and the mode
        self._db_uri = gen_uri(orch_path, mode)

        # run table: start_hash, end_hash, num_cycles, configuration_id

        # get a raw connection to the database
        self._db = sqlite3.connect(self.db_uri, uri=True,
                                   timeout=self.SQLITE3_DEFAULT_TIMEOUT)
        self._closed = False

        # set isolation level to autocommit
        self._db.isolation_level = None

        # we can use read_uncommited only in append_only mode (no
        # updates) because you never have to worry about dirty reads
        # since you can't update
        if self.append_only:
            self._db.execute("PRAGMA read_uncommited=1")

        # we make a table for the run data, if it doesn't already
        # exist
        c = self._db.cursor().execute(self.create_run_table_query)


        # initialize or open each of the separate KV-stores (tables in
        # the same SQLite3 database)

        # change the mode for the KV stores since we already created the database

        # metadata: default init walkers, default apparatus, default
        # configuration
        self.metadata_kv = KV(db_url=self.db_uri,
                              table='meta',
                              mode='a',
                              value_types=None,
                              append_only=self.append_only)



        # snapshots
        self.snapshot_kv = KV(db_url=self.db_uri,
                              table='snapshots',
                              primary_key='snaphash',
                              value_name='snapshot',
                              mode='a',
                              append_only=self.append_only)

        # configurations
        self.configuration_kv = KV(db_url=self.db_uri,
                                   table='configurations',
                                   primary_key='config_hash',
                                   value_name='config',
                                   mode='a',
                                   append_only=self.append_only)

    @property
    def mode(self):
        return self._mode

    @property
    def append_only(self):
        return self._append_only

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

    @property
    def db_uri(self):
        return self._db_uri

    @property
    def orch_path(self):

        # if it is not an in-memory database we parse off the path and
        # return that
        if self.db_uri == SQLITE3_INMEMORY_URI:
            return None
        else:

            # URIs have the following form: protocol:url?query
            # destructure the URI
            _, tail = self.db_uri.split(':')

            if len(tail.split('?')) > 1:
                url, _ = tail.split('?')
            else:
                url = tail

            return url


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

        self.metadata_kv['default_sim_apparatus'] = serial_app

    def set_default_init_walkers(self, init_walkers):

        # serialize the apparatus and then set it
        serial_walkers = self.serialize(init_walkers)

        self.metadata_kv['default_init_walkers'] = serial_walkers

    def set_default_configuration(self, configuration):

        # serialize the apparatus and then set it
        serial_config = self.serialize(configuration)

        config_hash = self.hash_snapshot(serial_config)

        self.metadata_kv['default_configuration_hash'] = config_hash

        self.configuration_kv[config_hash] = serial_config

    def set_default_snapshot(self, snapshot):

        snaphash = self.add_snapshot(snapshot)

        # then save the hash in the metadata
        self.metadata_kv['default_snapshot_hash'] = snaphash

        return snaphash

    def gen_default_snapshot(self):

        # generate the snapshot
        sim_start_hash = self.gen_start_snapshot(self.get_default_init_walkers())

        # then save the hash in the metadata
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
        self.snapshot_kv[snaphash] = serialized_snapshot

        return snaphash

    def add_serial_snapshot(self, serial_snapshot):

        # get the hash of the snapshot
        snaphash = self.hash_snapshot(serial_snapshot)

        # check that the hash is not already in the snapshots
        if any([True if snaphash == md5 else False for md5 in self.snapshot_hashes]):

            # just skip the rest of the function and return the hash
            return snaphash

        # save the snapshot in the KV store
        self.snapshot_kv[snaphash] = serial_snapshot

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
        self.configuration_kv[config_hash] = serialized_config

        return config_hash

    def add_serial_configuration(self, serial_configuration):

        # get the hash of the configuration
        snaphash = self.hash_snapshot(serial_configuration)

        # check that the hash is not already in the configurations
        if any([True if snaphash == md5 else False for md5 in self.configuration_hashes]):

            # just skip the rest of the function and return the hash
            return snaphash

        # save the configuration in the KV store
        self.configuration_kv[snaphash] = serial_configuration

        return snaphash


    @property
    def create_run_table_query(self):
        create_run_table_query = """
            CREATE TABLE IF NOT EXISTS runs
             (start_hash TEXT NOT NULL,
             end_hash TEXT NOT NULL,
             config_hash NOT NULL,
             last_cycle_idx INTEGER NOT NULL,
             PRIMARY KEY (start_hash, end_hash))

        """

        return create_run_table_query

    @property
    def add_run_record_query(self):

        add_run_row_query = """
        INSERT INTO runs (start_hash, end_hash, config_hash, last_cycle_idx)
        VALUES (?, ?, ?, ?)
        """

        return add_run_row_query

    @property
    def update_run_record_query(self):

        q = """
        UPDATE runs
        SET config_hash = ?,
            last_cycle_idx = ?
        WHERE start_hash=? AND end_hash=?
        """

        return q

    @property
    def delete_run_record_query(self):

        q = """
        DELETE FROM runs
        WHERE start_hash=? AND end_hash=?
        """

        return q


    def _add_run_record(self, start_hash, end_hash, configuration_hash, cycle_idx):

        params = (start_hash, end_hash, configuration_hash, cycle_idx)

        # do it as a transaction
        c = self._db.cursor()

        # run the insert
        c.execute(self.add_run_record_query, params)

    def _delete_run_record(self, start_hash, end_hash):

        params = (start_hash, end_hash)

        cursor = self._db.cursor()

        cursor.execute(self.delete_run_record_query, params)

    def _update_run_record(self, start_hash, end_hash, new_config_hash, new_last_cycle_idx):

        params = (new_config_hash, new_last_cycle_idx, start_hash, end_hash)

        # do it as a transaction
        c = self._db.cursor()

        # run the update
        c.execute(self.update_run_record_query, params)



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

    def run_configuration_hash(self, start_hash, end_hash):

        record = self.get_run_record(start_hash, end_hash)

        config_hash = record[self.RUN_SELECT_FIELDS.index('config_hash')]

        return config_hash

    def run_hashes(self):

        return [(rec[0], rec[1]) for rec in self.get_run_records()]

    def run_continues(self, start_hash, end_hash):
        """Given a start hash and end hash for a run, find the run that this
        continues.

        Parameters
        ----------
        start_hash :
            
        end_hash :
            

        Returns
        -------
        run_id

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


    def _init_checkpoint_db(self, start_hash, configuration, checkpoint_dir, mode='x'):

        logging.debug("Initializing checkpoint orch database")

        # make the checkpoint with the default filename at the checkpoint directory
        checkpoint_path = osp.join(checkpoint_dir, self.DEFAULT_CHECKPOINT_FILENAME)

        # create a new database in the mode specified
        logging.debug("Creating checkpoint database")
        checkpoint_orch = Orchestrator(checkpoint_path, mode=mode)

        # add the starting snapshot, bypassing the serialization stuff
        logging.debug("Setting the starting snapshot")
        checkpoint_orch.snapshot_kv[start_hash] = self.snapshot_kv[start_hash]

        # if we have a new configuration at runtime serialize and
        # hash it
        serialized_config = self.serialize(configuration)
        config_hash = self.hash_snapshot(serialized_config)

        # save the configuration as well
        checkpoint_orch.configuration_kv[config_hash] = serialized_config

        checkpoint_orch.close()
        logging.debug("closing connection to checkpoint database")

        return checkpoint_path, config_hash


    def _save_checkpoint(self, checkpoint_snapshot, config_hash,
                         checkpoint_db_path, cycle_idx,
                         ):
        """

        Parameters
        ----------
        checkpoint_snapshot :
            
        config_hash :
            
        checkpoint_db_path :
            
        mode :
             (Default value = 'wb')

        Returns
        -------

        """

        # orchestrator wrapper to the db
        logging.debug("Opening the checkpoint orch database")
        checkpoint_orch = Orchestrator(checkpoint_db_path, mode='r+')

        # connection to the db
        cursor = checkpoint_orch._db.cursor()

        # we replicate the code for adding the snapshot here because
        # we want it to occur transactionally the delete and add

        # serialize the snapshot using the protocol for doing so
        serialized_snapshot = self.serialize(checkpoint_snapshot)

        # get the hash of the snapshot
        snaphash = self.hash_snapshot(serialized_snapshot)

        # the queries for deleting and inserting the new run record
        delete_query = """
        DELETE FROM runs
        WHERE start_hash=?
          AND end_hash=?
        """

        insert_query = """
        INSERT INTO runs (start_hash, end_hash, config_hash, last_cycle_idx)
        VALUES (?, ?, ?, ?)
        """

        # if there are any runs in the checkpoint orch remove the
        # final snapshot
        delete_params = None
        if len(checkpoint_orch.run_hashes()) > 0:
            start_hash, old_checkpoint_hash = checkpoint_orch.run_hashes()[0]

            delete_params = (start_hash, old_checkpoint_hash)
        else:
            start_hash = list(checkpoint_orch.snapshot_kv.keys())[0]

        # the config should already be in the orchestrator db
        insert_params = (start_hash, snaphash, config_hash, cycle_idx)


        # start this whole process as a transaction so we don't get
        # something weird in between
        logging.debug("Starting transaction for updating run table in checkpoint")
        cursor.execute("BEGIN TRANSACTION")

        # add the new one, using a special method for setting inside
        # of a transaction
        logging.debug("setting the new checkpoint snapshot into the KV")
        cursor = checkpoint_orch.snapshot_kv.set_in_tx(cursor, snaphash, serialized_snapshot)
        logging.debug("finished")

        # if we need to delete the old end of the run snapshot and the
        # run record for it
        if delete_params is not None:

            logging.debug("Old run record needs to be removed")

            # remove the old run from the run table
            logging.debug("Deleting the old run record")
            cursor.execute(delete_query, delete_params)
            logging.debug("finished")

        # register the new run in the run table
        logging.debug("Inserting the new run record")
        cursor.execute(insert_query, insert_params)
        logging.debug("finished")

        # end the transaction
        logging.debug("Finishing transaction")
        cursor.execute("COMMIT")
        logging.debug("Transaction committed")

        # we do the removal of the old snapshot outside of the
        # transaction since it is slow and can cause timeouts to
        # occur. Furthermore, it is okay if it is in the checkpoint as
        # the run record is what matters as long as the new checkpoint
        # is there.

        # delete the old snapshot if we need to
        if delete_params is not None:
            logging.debug("Deleting the old snapshot")
            del checkpoint_orch.snapshot_kv[old_checkpoint_hash]
            logging.debug("finished")


        checkpoint_orch.close()
        logging.debug("closed the checkpoint orch connection")


    @staticmethod
    def gen_sim_manager(start_snapshot, configuration):
        """

        Parameters
        ----------
        start_snapshot :
            
        configuration :
            

        Returns
        -------

        """

        # construct the sim manager, in a wepy specific way
        sim_manager = Manager(start_snapshot.walkers,
                              runner=start_snapshot.apparatus.filters[0],
                              boundary_conditions=start_snapshot.apparatus.filters[1],
                              resampler=start_snapshot.apparatus.filters[2],
                              # configuration options
                              work_mapper=configuration.work_mapper,
                              reporters=configuration.reporters,
                              sim_monitor=configuration.monitor,
        )

        return sim_manager


    def run_snapshot_by_time(self, start_hash, run_time, n_steps,
                             checkpoint_freq=None,
                             checkpoint_dir=None,
                             configuration=None,
                             configuration_hash=None,
                             checkpoint_mode='x'):
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
        configuration_hash :
             (Default value = None)
        checkpoint_mode :
             (Default value = None)

        Returns
        -------

        """


        # you must have a checkpoint dir if you ask for a checkpoint
        # frequency
        if checkpoint_freq is not None and checkpoint_dir is None:
            raise ValueError("Must provide a directory for the checkpoint file "
                             "is a frequency is specified")


        if configuration_hash is not None and configuration is not None:
            raise ValueError("Cannot specify both a hash of an existing configuration"
                             "and provide a runtime configuration")

        # if no configuration was specified we use the default one, oth
        elif (configuration is None) and (configuration_hash is None):
            configuration = self.get_default_configuration()

        # if a configuration hash was given only then we retrieve that
        # configuration since we must pass configurations to the
        # checkpoint DB initialization
        elif configuration_hash is not None:
            configuration = self.configuration_kv[configuration_hash]


        # check that the directory for checkpoints exists, and create
        # it if it doesn't and isn't already created
        if checkpoint_dir is not None:
            checkpoint_dir = osp.realpath(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)


        # if the checkpoint dir is not specified don't create a
        # checkpoint db orch
        checkpoint_db_path = None
        if checkpoint_dir is not None:
            logging.debug("Initialization of checkpoint database is requested")
            checkpoint_db_path, configuration_hash = self._init_checkpoint_db(start_hash,
                                                                              configuration,
                                                                              checkpoint_dir,
                                                                              mode=checkpoint_mode)
            logging.debug("finished initializing checkpoint database")


        # get the snapshot and the configuration to use for the sim_manager
        start_snapshot = self.get_snapshot(start_hash)

        # generate the simulation manager given the snapshot and the
        # configuration
        sim_manager = self.gen_sim_manager(start_snapshot, configuration)

        # handle and process the optional arguments for running simulation
        if 'runner' in configuration.apparatus_opts:
            runner_opts = configuration.apparatus_opts['runner']
        else:
            runner_opts = None

        # run the init subroutine for the simulation manager
        logging.debug("Running sim_manager.init")
        sim_manager.init()

        # run each cycle manually creating checkpoints when necessary
        logging.debug("Starting run loop")
        walkers = sim_manager.init_walkers
        cycle_idx = 0
        start_time = time.time()
        while time.time() - start_time < run_time:

            logging.debug("Running cycle {}".format(cycle_idx))
            # run the cycle
            walkers, filters = sim_manager.run_cycle(
                walkers,
                n_steps,
                cycle_idx,
                runner_opts=runner_opts,
            )

            # check to see if a checkpoint is necessary
            if (checkpoint_freq is not None):
                if (cycle_idx % checkpoint_freq == 0):
                    logging.debug("Checkpoint is required for this cycle")

                    # make the checkpoint snapshot
                    logging.debug("Generating the simulation snapshot")
                    checkpoint_snapshot = SimSnapshot(walkers, SimApparatus(filters))

                    # save the checkpoint (however that is implemented)
                    logging.debug("saving the checkpoint to the database")
                    self._save_checkpoint(checkpoint_snapshot,
                                          configuration_hash,
                                          checkpoint_db_path,
                                          cycle_idx)
                    logging.debug("finished saving the checkpoint to the database")

            # increase the cycle index for the next cycle
            cycle_idx += 1

        logging.debug("Finished the run cycle")

        # the cycle index was set for the next cycle which didn't run
        # so we decrement it
        last_cycle_idx = cycle_idx - 1

        logging.debug("Running sim_manager.cleanup")
        # run the cleanup subroutine
        sim_manager.cleanup()

        # run the segment given the sim manager and run parameters
        end_snapshot = SimSnapshot(walkers, SimApparatus(filters))

        logging.debug("Run finished")
        # return the things necessary for saving to the checkpoint if
        # that is what is wanted later on
        return end_snapshot, configuration_hash, checkpoint_db_path, last_cycle_idx

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
            configuration = self.get_default_configuration()

            # reparametrize the configuration with the given path
            # parameters and anything else in kwargs. If they are none
            # this will have no effect anyhow
            logging.debug("Reparametrizing the configuration")
            configuration = configuration.reparametrize(work_dir=work_dir,
                                                        config_name=config_name,
                                                        narration=narration,
                                                        mode=mode,
                                                        **kwargs)

        # make parametric paths for the checkpoint directory and the
        # orchestrator pickle to be made, unless they are explicitly given

        if checkpoint_dir is None:

            # the checkpoint directory will be in the work dir
            logging.debug("checkpoint directory defaulted to the work_dir")
            checkpoint_dir = work_dir


        logging.debug("In the orchestrate run, calling to run_snapshot by time")
        # then actually run the simulation with checkpointing. This
        # returns the end snapshot and doesn't write out anything to
        # orchestrators other than the checkpointing
        (end_snapshot, configuration_hash, checkpoint_db_path, last_cycle_idx) =\
            self.run_snapshot_by_time(snapshot_hash, run_time, n_steps,
                                            checkpoint_freq=checkpoint_freq,
                                            checkpoint_dir=checkpoint_dir,
                                            configuration=configuration,
                                            checkpoint_mode=orch_mode)

        logging.debug("Finished running snapshot by time")

        # if the last cycle in the run was a checkpoint skip this step
        # of saving a checkpoint
        do_final_checkpoint = True

        # make sure the checkpoint_freq is defined before testing it
        if checkpoint_freq is not None:
            if checkpoint_freq % last_cycle_idx == 0:
                logging.debug("Last cycle saved a checkpoint, no need to save one")
                do_final_checkpoint = False

        if do_final_checkpoint:

            logging.debug("Saving a final checkpoint for the end of the run")
            # now that it is finished we save the final snapshot to the
            # checkpoint file. This is done transactionally using the
            # SQLite transaction functionality (either succeeds or doesn't
            # happen) that way we don't have worry about data integrity
            # loss. Here we also don't have to worry about other processes
            # interacting with the checkpoint which makes it isolated.
            self._save_checkpoint(end_snapshot, configuration_hash,
                                  checkpoint_db_path, last_cycle_idx)
            logging.debug("Finished saving the final checkpoint for the run")

        # then return the final orchestrator
        logging.debug("Getting a connection to that orch to retun")
        checkpoint_orch = Orchestrator(checkpoint_db_path,
                                       mode='r+',
                                       append_only=True)

        return checkpoint_orch




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
                            mode='a',
                            append_only=True)

    # TODO deprecate, if there is no defaults we can't set them since
    # the mode is append only, we don't really care about these so
    # don't set them, otherwise do some mode logic to figure this out
    # and open in write mode and set defaults, then change to append
    # only

    # # if this is an existing orchestrator copy the default
    # # sim_apparatus and init_walkers
    # try:
    #     default_app = new_orch.get_default_sim_apparatus()
    # except KeyError:
    #     # no default apparatus, that is okay
    #     pass
    # else:
    #     # set it
    #     new_orch.set_default_sim_apparatus(default_app)

    # # same for the initial walkers
    # try:
    #     default_walkers = new_orch.get_default_init_walkers()
    # except KeyError:
    #     # no default apparatus, that is okay
    #     pass
    # else:
    #     # set it
    #     new_orch.set_default_sim_apparatus(default_walkers)


    for orch_path in orchestrator_paths:

        # open it in read-write fail if doesn't exist
        orch = Orchestrator(orch_path=orch_path,
                            mode='r+',
                            append_only=True)

        # add in all snapshots from each orchestrator, by the hash not the
        # snapshots themselves, we trust they are correct
        for snaphash in orch.snapshot_hashes:

            # check that the hash is not already in the snapshots
            if any([True if snaphash == md5 else False for md5 in new_orch.snapshot_hashes]):

                # skip it and move on
                continue

            # if it is not copy it over without deserializing
            new_orch.snapshot_kv[snaphash] = orch.snapshot_kv[snaphash]

        # add in the configurations for the runs from each
        # orchestrator, by the hash not the snapshots themselves, we
        # trust they are correct
        for run_id in orch.run_hashes():

            config_hash = orch.run_configuration_hash(*run_id)

            # check that the hash is not already in the snapshots
            if any([True if config_hash == md5 else False for md5 in new_orch.configuration_hashes]):

                # skip it and move on
                continue

            # if it is not set it
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

        # then run the queries

        cursor = new_orch._db.cursor()
        try:
            cursor.execute('BEGIN TRANSACTION')
            cursor.execute(attach_query)
            cursor.execute(union_query)
            cursor.execute('COMMIT')
            cursor.execute(detach_query)
        except:
            cursor.execute('COMMIT')
            import pdb; pdb.set_trace()
            cursor.execute("SELECT * FROM (SELECT * FROM other.runs EXCEPT SELECT * FROM runs)")
            recs = cursor.fetchall()

    return new_orch
