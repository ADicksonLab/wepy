import os.path as osp
import logging
import subprocess
from copy import deepcopy

import click

from wepy.orchestration.orchestrator import (
    reconcile_orchestrators,
    Orchestrator
)

from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.hdf5 import WepyHDF5

ORCHESTRATOR_DEFAULT_FILENAME = \
            Orchestrator.ORCH_FILENAME_TEMPLATE.format(config=Orchestrator.DEFAULT_CONFIG_NAME,
                                                       narration=Orchestrator.DEFAULT_NARRATION)

def set_loglevel(loglevel):
    """

    \b
    Parameters
    ----------
    loglevel :
        

    \b
    Returns
    -------

    """

    # try to cast the loglevel as an integer. If that fails interpret
    # it as a string.
    try:
        loglevel_num = int(loglevel)
    except ValueError:
        loglevel_num = getattr(logging, loglevel, None)

    # if no such log level exists in logging the string was invalid
    if loglevel_num is None:
        raise ValueError("invalid log level given")

    logging.basicConfig(level=loglevel_num)

START_HASH = '<start_hash>'
CURDIR = '<curdir>'

def settle_run_options(n_workers=None,
                       job_dir=None,
                       job_name=None,
                       narration=None,
                       monitor_http_port=None,
                       tag=None,
                       configuration=None,
                       start_hash=None,
):
    """

    Parameters
    ----------
    n_workers :
         (Default value = None)
    job_dir :
         (Default value = None)
    job_name :
         (Default value = None)
    narration :
         (Default value = None)

    \b
    Returns
    -------

    """

    # the default for the job name is the start hash if none is given
    if job_name == START_HASH:
        job_name = start_hash

    # if the job_name is given and the default value for the job_dir
    # is given (i.e. not specified by the user) we set the job-dir as
    # the job_name
    if job_name is not None and job_dir == CURDIR:
            job_dir = job_name

    # if the special value for curdir is given we get the systems
    # current directory, this is the default.
    if job_dir == CURDIR:
        job_dir = osp.curdir

    # normalize the job_dir
    job_dir = osp.realpath(job_dir)

    # if a path for a configuration was given we want to use it so we
    # unpickle it and return it, otherwise return None and use the
    # default one in the orchestrator
    config = None
    if configuration is not None:
         with open(configuration, 'rb') as rf:
             config = Orchestrator.deserialize(rf.read())

    ## Monitoring

    # nothing to do, just use the port or tag if its given
    monitor_pkwargs = {
        'tag' : tag,
        'port' : monitor_http_port,
    }

    # we need to reparametrize the configuration here since the
    # orchestrator API will ignore reparametrization values if a
    # concrete Configuration is given.
    if config is not None:

        # if there is a change in the number of workers we need to
        # recalculate all of the partial kwargs

        work_mapper_pkwargs = deepcopy(config.work_mapper_partial_kwargs)

        # if the number of workers has changed update all the relevant
        # fields, otherwise leave it alone
        if work_mapper_pkwargs['num_workers'] != n_workers:

            work_mapper_pkwargs['num_workers'] = n_workers
            work_mapper_pkwargs['device_ids'] = [str(i) for i in range(n_workers)]

        config = config.reparametrize(work_dir=job_dir,
                                      config_name=job_name,
                                      narration=narration,
                                      work_mapper_partial_kwargs=work_mapper_pkwargs,
                                      monitor_partial_kwargs=monitor_pkwargs,
        )

    return job_dir, job_name, narration, config

@click.option('--log', default="WARNING")
@click.option('--n-workers', type=click.INT)
@click.option('--checkpoint-freq', default=None, type=click.INT)
@click.option('--job-dir', default=CURDIR, type=click.Path(writable=True))
@click.option('--job-name', default=START_HASH)
@click.option('--narration', default="")
@click.option('--monitor-http-port', default=9001)
@click.option('--tag', default="None")
@click.argument('n_cycle_steps', type=click.INT)
@click.argument('run_time', type=click.FLOAT)
@click.argument('configuration', type=click.Path(exists=True))
@click.argument('snapshot', type=click.File('rb'))
@click.command()
def run_snapshot(log, n_workers, checkpoint_freq, job_dir, job_name, narration,
                 monitor_http_port, tag,
                 n_cycle_steps, run_time, configuration, snapshot):
    """

    \b
    Parameters
    ----------
    log :
        
    n_workers :
        
    checkpoint_freq :
        
    job_dir :
        
    job_name :
        
    narration :
        
    monitor_http_port :
        
    n_cycle_steps :
        
    run_time :
        
    start_hash :
        
    orchestrator :
        

    \b
    Returns
    -------

    """

    set_loglevel(log)

    logging.info("Loading the starting snapshot file")
    # read the config and snapshot in
    serial_snapshot = snapshot.read()

    logging.info("Creating orchestrating orch database")
    # make the orchestrator for this simulation in memory to start
    orch = Orchestrator()
    logging.info("Adding the starting snapshot to database")
    start_hash = orch.add_serial_snapshot(serial_snapshot)

    # settle what the defaults etc. are for the different options as they are interdependent
    job_dir, job_name, narration, config = settle_run_options(
        # work mapper
        n_workers=n_workers,
        # reporters
        job_dir=job_dir,
        job_name=job_name,
        narration=narration,
        # monitoring
        tag=tag,
        monitor_http_port=monitor_http_port,
        # other
        configuration=configuration,
        start_hash=start_hash,
    )

    # add the parametrized configuration to the orchestrator
    # config_hash = orch.add_serial_configuration(config)

    logging.info("Orchestrator loaded")
    logging.info("Running snapshot by time")
    run_orch = orch.orchestrate_snapshot_run_by_time(start_hash,
                                                     run_time,
                                                     n_cycle_steps,
                                                     checkpoint_freq=checkpoint_freq,
                                                     work_dir=job_dir,
                                                     config_name=job_name,
                                                     narration=narration,
                                                     configuration=config,
                                                     )
    logging.info("Finished running snapshot by time")

    start_hash, end_hash = run_orch.run_hashes()[0]

    run_orch.close()
    logging.info("Closed the resultant orch")

    # write the run tuple out to the log
    run_line_str = "Run start and end hashes: {}, {}".format(start_hash, end_hash)

    # log it
    logging.info(run_line_str)

    # also put it to the terminal
    click.echo(run_line_str)

    orch.close()
    logging.info("closed the orchestrating orch database")

@click.option('--log', default="WARNING")
@click.option('--n-workers', type=click.INT)
@click.option('--checkpoint-freq', default=None, type=click.INT)
@click.option('--job-dir', default=CURDIR, type=click.Path(writable=True))
@click.option('--job-name', default=START_HASH)
@click.option('--narration', default="")
@click.option('--configuration', type=click.Path(exists=True), default=None)
@click.argument('n_cycle_steps', type=click.INT)
@click.argument('run_time', type=click.FLOAT)
@click.argument('start_hash')
@click.argument('orchestrator', type=click.Path(exists=True))
@click.command()
def run_orch(log, n_workers, checkpoint_freq, job_dir, job_name, narration, configuration,
        n_cycle_steps, run_time, start_hash, orchestrator):
    """

    \b
    Parameters
    ----------
    log :
        
    n_workers :
        
    checkpoint_freq :
        
    job_dir :
        
    job_name :
        
    narration :
        
    n_cycle_steps :
        
    run_time :
        
    start_hash :
        
    orchestrator :
        

    \b
    Returns
    -------

    """

    set_loglevel(log)

    # settle what the defaults etc. are for the different options as they are interdependent
    job_dir, job_name, narration, config = settle_run_options(n_workers=n_workers,
                                                                         job_dir=job_dir,
                                                                         job_name=job_name,
                                                                         narration=narration,
                                                                         configuration=configuration,
                                                                         start_hash=start_hash)

    # Open a wrapper around the orchestrator database that provides
    # the inputs for the simulation
    orch = Orchestrator(orchestrator, mode='r')

    logging.info("Orchestrator loaded")

    logging.info("Running snapshot by time")
    run_orch = orch.orchestrate_snapshot_run_by_time(start_hash,
                                                     run_time,
                                                     n_cycle_steps,
                                                     checkpoint_freq=checkpoint_freq,
                                                     work_dir=job_dir,
                                                     config_name=job_name,
                                                     narration=narration,
                                                     configuration=config,
                                                     )
    logging.info("Finished running snapshot by time")

    start_hash, end_hash = run_orch.run_hashes()[0]

    logging.info("Closing the resultant orchestrator")
    run_orch.close()

    # write the run tuple out to the log
    run_line_str = "Run start and end hashes: {}, {}".format(start_hash, end_hash)

    # log it
    logging.info(run_line_str)

    # also put it to the terminal
    click.echo(run_line_str)

    logging.info("Closing the orchestrating orch")
    orch.close()

def combine_orch_wepy_hdf5s(new_orch, new_hdf5_path, run_ids=None):
    """

    \b
    Parameters
    ----------
    new_orch :
        
    new_hdf5_path :
        

    \b
    Returns
    -------

    """

    if run_ids is None:
        run_ids = new_orch.run_hashes()

    # we assume that the run we are interested in is the only run in
    # the WepyHDF5 file so it is index 0
    singleton_run_idx = 0

    # a key-value for the paths for each run
    hdf5_paths = {}

    # go through each run in the new orchestrator
    for run_id in run_ids:

        # get the configuration used for this run
        run_config = new_orch.run_configuration(*run_id)

        # from that configuration find the WepyHDF5Reporters
        for reporter in run_config.reporters:

            if isinstance(reporter, WepyHDF5Reporter):

                # and save the path for that run
                hdf5_paths[run_id] = reporter.file_path

    click.echo("Combining these HDF5 files:")
    click.echo('\n'.join(hdf5_paths.values()))

    # now that we have the paths (or lack of paths) for all
    # the runs we need to start linking them all
    # together.

    # first we need a master linker HDF5 to do this with

    # so load a template WepyHDF5
    template_wepy_h5_path = hdf5_paths[run_ids[singleton_run_idx]]
    template_wepy_h5 = WepyHDF5(template_wepy_h5_path, mode='r')

    # clone it
    with template_wepy_h5:
        master_wepy_h5 = template_wepy_h5.clone(new_hdf5_path, mode='x')

    click.echo("Into a single master hdf5 file: {}".format(new_hdf5_path))

    # then link all the files to it
    run_mapping = {}
    for run_id, wepy_h5_path in hdf5_paths.items():

        # in the case where continuations were done from
        # checkpoints then the runs data will potentially (and
        # most likely) contain extra cycles since checkpoints are
        # typically produced on some interval of cycles. So, in
        # order for us to actually piece together contigs we need
        # to take care of this.

        # There are two ways to deal with this which can both be
        # done at the same time. The first is to keep the "nubs",
        # which are the small leftover pieces after the checkpoint
        # that ended up getting continued, and make a new run from
        # the last checkpoint to the end of the nub, in both the
        # WepyHDF5 and the orchestrator run collections.

        # The second is to generate a WepyHDF5 run that
        # corresponds to the run in the checkpoint orchestrator.

        # To avoid complexity (for now) we opt to simply dispose
        # of the nubs and assume that not much will be lost from
        # this. For the typical use case of making multiple
        # independent and linear contigs this is also the simplest
        # mode, since the addition of multiple nubs will introduce
        # an extra spanning contig in the contig tree.

        # furthermore the nubs provide a source of problems if
        # rnus were abruptly stopped and data is not written some
        # of the frames can be corrupted. SO until we know how to
        # stop this (probably SWMR mode will help) this is also a
        # reason not to deal with nubs.

        # TODO: add option to keep nubs in HDF5, and deal with in
        # orch (you won't be able to have an end snapshot...).

        # to do this we simply check whether or not the number of
        # cycles for the run_id are less than the number of cycles
        # in the corresponding WepyHDF5 run dataset.
        orch_run_num_cycles = new_orch.run_last_cycle_idx(*run_id)

        # get the number of cycles that are in the data for the run in
        # the HDF5 to compare to the number in the orchestrator run
        # record
        wepy_h5 = WepyHDF5(wepy_h5_path, mode='r')
        with wepy_h5:
            h5_run_num_cycles = wepy_h5.num_run_cycles(singleton_run_idx)

        # sanity check for if the number of cycles in the
        # orchestrator is greater than the HDF5
        if orch_run_num_cycles > h5_run_num_cycles:
            raise ValueError("Number of cycles in orch run is more than HDF5."\
                             "This implies missing data")

        # copy the run (with the slice)
        with master_wepy_h5:

            # TODO: this was the old way of combining where we would
            # just link, however due to the above discussion this is
            # not tenable now. In the future there might be some more
            # complex options taking linking into account but for now
            # we just don't use it and all runs will be copied by this
            # operation

            # # we just link the whole file then sort out the
            # # continuations later since we aren't necessarily doing
            # # this in a logical order
            # new_run_idxs = master_wepy_h5.link_file_runs(wepy_h5_path)

            # extract the runs from the file (there should only be
            # one). This means copy the run, but if we only want a
            # truncation of it we will use the run slice to only get
            # part of it

            # so first we generate the run slices for this file using
            # the number of cycles recorded in the orchestrator
            run_slices = {singleton_run_idx : (0, orch_run_num_cycles)}

            click.echo("Extracting Run: {}".format(run_id))
            click.echo("Frames 0 to {} out of {}".format(
                orch_run_num_cycles, h5_run_num_cycles))

            # then perform the extraction, which will open the other
            # file on its own
            new_run_idxs = master_wepy_h5.extract_file_runs(wepy_h5_path,
                                                            run_slices=run_slices)

            # map the hash id to the new run idx created. There should
            # only be one run in an HDF5 if we are following the
            # orchestration workflow.
            assert len(new_run_idxs) < 2, \
                "Cannot be more than 1 run per HDF5 file in orchestration workflow"

            run_mapping[run_id] = new_run_idxs[0]

            click.echo("Set as run: {}".format(new_run_idxs[0]))

    click.echo("Done extracting runs, setting continuations")

    with master_wepy_h5:

        # now that they are all linked we need to add the snapshot
        # hashes identifying the runs as metadata. This is so we can
        # map the simple run indices in the HDF5 back to the
        # orchestrator defined runs. This will be saved as metadata on
        # the run. Also:

        # We need to set the continuations correctly betwen the runs
        # in different files, so for each run we find the run it
        # continues in the orchestrator
        for run_id, run_idx in run_mapping.items():

            # set the run snapshot hash metadata except for if we have
            # already done it
            try:
                master_wepy_h5.set_run_start_snapshot_hash(run_idx, run_id[0])
            except AttributeError:
                # it was already set so just move on
                pass
            try:
                master_wepy_h5.set_run_end_snapshot_hash(run_idx, run_id[1])
            except AttributeError:
                # it was already set so just move on
                pass

            # find the run_id that this one continues
            continued_run_id = new_orch.run_continues(*run_id)

            # if a None is returned then there was no continuation
            if continued_run_id is None:
                # so we go to the next run_id and don't log any
                # continuation
                continue

            # get the run_idx in the HDF5 that corresponds to this run
            continued_run_idx = run_mapping[continued_run_id]

            click.echo("Run {} continued by {}".format(continued_run_id, run_idx))

            # add the continuation
            master_wepy_h5.add_continuation(run_idx, continued_run_idx)

@click.command()
@click.argument('orchestrator', nargs=1, type=click.Path(exists=True))
@click.argument('hdf5', nargs=1, type=click.Path(exists=False))
@click.argument('run_ids', nargs=-1)
def reconcile_hdf5(orchestrator, hdf5, run_ids):
    """For an orchestrator with multiple runs combine the HDF5 results
    into a single one.

    This requires that the paths inside of the reporters for the
    configurations used for a run still have valid paths to the HDF5
    files.

    \b
    Parameters
    ----------
    orchestrator : Path
        The orchestrator to retrieve HDF5s for

    hdf5 : Path
        Path to the resultant HDF5.

    run_ids : str
        String specifying a run as start and end hash
        e.g. 'd0cb2e6fbcc8c2d66d67c845120c7f6b,b4b96580ae57f133d5f3b6ce25affa6d'

    \b
    Returns
    -------

    """

    # parse the run ids
    run_ids = [tuple(run_id.split(',')) for run_id in run_ids]

    orch = Orchestrator(orchestrator, mode='r')

    hdf5_path = osp.realpath(hdf5)

    click.echo("Combining the HDF5s together, saving to:")
    click.echo(hdf5_path)

    # combine the HDF5 files from those orchestrators
    combine_orch_wepy_hdf5s(orch, hdf5_path, run_ids=run_ids)


@click.command()
@click.option('--hdf5', type=click.Path(exists=False))
@click.argument('output', nargs=1, type=click.Path(exists=False))
@click.argument('orchestrators', nargs=-1, type=click.Path(exists=True))
def reconcile_orch(hdf5, output, orchestrators):
    """ 

    \b
    Parameters
    ----------
    hdf5 : Path
        Path to the resultant HDF5.
    output : Path
        Path to the resultant orchestrator that is created.
    orchestrators : Path
        Paths to the orchestrators to reconcile.

    \b
    Returns
    -------

    """

    new_orch = reconcile_orchestrators(output, *orchestrators)

    # if a path for an HDF5 file is given
    if hdf5 is not None:
        hdf5_path = osp.realpath(hdf5)

        click.echo("Combining the HDF5s together, saving to:")
        click.echo(hdf5_path)

        # combine the HDF5 files from those orchestrators
        combine_orch_wepy_hdf5s(new_orch, hdf5_path)



def hash_listing_formatter(hashes):
    """

    \b
    Parameters
    ----------
    hashes :
        

    \b
    Returns
    -------

    """
    hash_listing_str = '\n'.join(hashes)
    return hash_listing_str

@click.argument('orchestrator', type=click.Path(exists=True))
@click.command()
def ls_snapshots(orchestrator):
    """

    \b
    Parameters
    ----------
    orchestrator :
        

    \b
    Returns
    -------

    """

    orch = Orchestrator(orch_path=orchestrator, mode='r')

    message = hash_listing_formatter(orch.snapshot_hashes)

    orch.close()

    click.echo(message)

@click.argument('orchestrator', type=click.Path(exists=True))
@click.command()
def ls_runs(orchestrator):
    """

    \b
    Parameters
    ----------
    orchestrator :
        

    \b
    Returns
    -------

    """

    orch = Orchestrator(orch_path=orchestrator, mode='r')

    runs = orch.run_hashes()

    orch.close()

    hash_listing_str = "\n".join(["{}, {}".format(start, end) for start, end in runs])

    click.echo(hash_listing_str)

@click.argument('orchestrator', type=click.Path(exists=True))
@click.command()
def ls_configs(orchestrator):
    """

    \b
    Parameters
    ----------
    orchestrator :
        

    \b
    Returns
    -------

    """

    orch = Orchestrator(orch_path=orchestrator, mode='r')

    message = hash_listing_formatter(orch.configuration_hashes)

    orch.close()

    click.echo(message)



@click.command()
@click.option('--no-expand-external', is_flag=True)
@click.argument('source', type=click.Path(exists=True))
@click.argument('target', type=click.Path(exists=False))
def hdf5_copy(no_expand_external, source, target):
    """Copy a WepyHDF5 file, except links to other runs will optionally be
    expanded and truly duplicated if symbolic inter-file links are present."""


    # arg clusters to pass to subprocess for the files
    input_f_args = ['-i', source]
    output_f_args = ['-o', target]

    # each invocation calls a different group since we can't call the
    # toplevel '/' directly
    settings_args = ['-s', '/units', '-d', '/units']
    settings_args = ['-s', '/_settings', '-d', '/_settings']
    topology_args = ['-s', '/topology', '-d', '/topology']
    runs_args = ['-s', '/runs', '-d', '/runs']

    # by default expand the external links
    flags_args = ['-f', 'ext']

    # if the not expand external flag is given get rid of those args
    if no_expand_external:
        flags_args = []

    common_args = input_f_args + output_f_args + flags_args

    settings_output = subprocess.check_output(['h5copy'] + common_args + settings_args)

    topology_output = subprocess.check_output(['h5copy'] + common_args + topology_args)

    runs_output = subprocess.check_output(['h5copy'] + common_args + runs_args)


@click.option('-O', '--output', type=click.Path(exists=False), default=None)
@click.argument('snapshot_hash')
@click.argument('orchestrator', type=click.Path(exists=True))
@click.command()
def get_snapshot(output, snapshot_hash, orchestrator):

    # first check if the output is None, if it is we automatically
    # generate a file in the cwd that is the hash of the snapshot
    if output is None:
        output = "{}.snap.dill.pkl".format(snapshot_hash)

        # check that it doesn't exist, and fail if it does, since we
        # don't want to implicitly overwrite stuff
        if osp.exists(output):
            raise OSError("No output path was specified and default alredy exists, exiting.")

    orch = Orchestrator(orchestrator, mode='r')

    serial_snapshot = orch.snapshot_kv[snapshot_hash]

    with open(output, 'wb') as wf:
        wf.write(serial_snapshot)

    orch.close()

@click.option('-O', '--output', type=click.Path(exists=False), default=None)
@click.argument('config_hash')
@click.argument('orchestrator', type=click.Path(exists=True))
@click.command()
def get_config(output, config_hash, orchestrator):

    # first check if the output is None, if it is we automatically
    # generate a file in the cwd that is the hash of the snapshot
    if output is None:
        output = "{}.config.dill.pkl".format(config_hash)

        # check that it doesn't exist, and fail if it does, since we
        # don't want to implicitly overwrite stuff
        if osp.exists(output):
            raise OSError("No output path was specified and default alredy exists, exiting.")

    orch = Orchestrator(orchestrator, mode='r')

    serial_snapshot = orch.configuration_kv[config_hash]

    with open(output, 'wb') as wf:
        wf.write(serial_snapshot)

        orch.close()

@click.option('-O', '--output', type=click.Path(exists=False), default=None)
@click.argument('end_hash')
@click.argument('start_hash')
@click.argument('orchestrator', type=click.Path(exists=True))
@click.command()
def get_run(output, end_hash, start_hash, orchestrator):

    # first check if the output is None, if it is we automatically
    # generate a file in the cwd that is the hash of the snapshot
    if output is None:
        output = "{}-{}.orch.sqlite".format(start_hash, end_hash)

        # check that it doesn't exist, and fail if it does, since we
        # don't want to implicitly overwrite stuff
        if osp.exists(output):
            raise OSError("No output path was specified and default alredy exists, exiting.")

    orch = Orchestrator(orchestrator, mode='r')

    start_serial_snapshot = orch.snapshot_kv[start_hash]
    end_serial_snapshot = orch.snapshot_kv[end_hash]


    # get the records values for this run
    rec_d = {field : value for field, value in
           zip(Orchestrator.RUN_SELECT_FIELDS, orch.get_run_record(start_hash, end_hash))}

    config = orch.configuration_kv[rec_d['config_hash']]

    # create a new orchestrator at the output location
    new_orch = Orchestrator(output, mode='w')

    _ = new_orch.add_serial_snapshot(start_serial_snapshot)
    _ = new_orch.add_serial_snapshot(end_serial_snapshot)
    config_hash = new_orch.add_serial_configuration(config)

    new_orch.register_run(start_hash, end_hash, config_hash, rec_d['last_cycle_idx'])

    orch.close()
    new_orch.close()

@click.argument('end_hash')
@click.argument('start_hash')
@click.argument('orchestrator', type=click.Path(exists=True))
@click.command()
def get_run_cycles(end_hash, start_hash, orchestrator):

    orch = Orchestrator(orchestrator, mode='r')

    start_serial_snapshot = orch.snapshot_kv[start_hash]
    end_serial_snapshot = orch.snapshot_kv[end_hash]

    # get the records values for this run
    rec_d = {field : value for field, value in
           zip(Orchestrator.RUN_SELECT_FIELDS, orch.get_run_record(start_hash, end_hash))}

    click.echo(rec_d['last_cycle_idx'])


@click.argument('orchestrator', type=click.Path(exists=False))
@click.command()
def create_orch(orchestrator):

    orch = Orchestrator(orchestrator, mode='x')

    orch.close()

@click.argument('snapshot', type=click.File('rb'))
@click.argument('orchestrator', type=click.Path(exists=True))
@click.command()
def add_snapshot(snapshot, orchestrator):

    orch = Orchestrator(orchestrator, mode='r+')

    serial_snapshot = snapshot.read()

    snaphash = orch.add_serial_snapshot(serial_snapshot)

    orch.close()

    click.echo(snaphash)


@click.argument('configuration', type=click.File('rb'))
@click.argument('orchestrator', type=click.Path(exists=True))
@click.command()
def add_config(configuration, orchestrator):
    orch = Orchestrator(orchestrator, mode='r+')

    serial_config = configuration.read()

    config_hash = orch.add_serial_snapshot(serial_config)

    orch.close()

    click.echo(config_hash)


@click.group()
def cli():
    """ """
    pass

@click.group()
def run():
    """ """
    pass

@click.group()
def get():
    """ """
    pass

@click.group()
def add():
    """ """
    pass

@click.group()
def create():
    """ """
    pass


@click.group()
def ls():
    """ """
    pass

@click.group()
def reconcile():
    """ """
    pass

@click.group()
def hdf5():
    """ """
    pass

# command groupings

# run
run.add_command(run_orch, name='orch')
run.add_command(run_snapshot, name='snapshot')

# ls
ls.add_command(ls_snapshots, name='snapshots')
ls.add_command(ls_runs, name='runs')
ls.add_command(ls_configs, name='configs')

# get
get.add_command(get_snapshot, name='snapshot')
get.add_command(get_config, name='config')
get.add_command(get_run, name='run')
get.add_command(get_run_cycles, name='run-cycles')

# add
add.add_command(add_snapshot, name='snapshot')
add.add_command(add_config, name='config')

# create
create.add_command(create_orch, name='orch')

# reconcile
reconcile.add_command(reconcile_orch, name='orch')
reconcile.add_command(reconcile_hdf5, name='hdf5')

# hdf5
hdf5.add_command(hdf5_copy, name='copy')
# desired commands
# hdf5.add_command(hdf5_copy, name='copy-run')
# hdf5.add_command(hdf5_copy, name='copy-traj')
# hdf5.add_command(hdf5_copy, name='ls-runs')
# hdf5.add_command(hdf5_copy, name='ls-run-hashes')

# subgroups
subgroups = [run, get, add, create, ls, reconcile, hdf5]

for subgroup in subgroups:
    cli.add_command(subgroup)

if __name__ == "__main__":

    cli()
