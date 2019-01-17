import os.path as osp
import logging
import subprocess

import dill

import click

from wepy.orchestration.orchestrator import deserialize_orchestrator, \
                                            reconcile_orchestrators, \
                                            Orchestrator, \
                                            recover_run_by_time

from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.hdf5 import WepyHDF5

ORCHESTRATOR_DEFAULT_FILENAME = \
            Orchestrator.ORCH_FILENAME_TEMPLATE.format(config=Orchestrator.DEFAULT_CONFIG_NAME,
                                                       narration=Orchestrator.DEFAULT_NARRATION)

@click.group()
def cli():
    """ """
    pass

def set_loglevel(loglevel):
    """

    Parameters
    ----------
    loglevel :
        

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
                       configuration=None):
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
             config = dill.load(rf)

    # we need to reparametrize the configuration here since the
    # orchestrator API will ignore reparametrization values if a
    # concrete Configuration is given.
    if config is not None:
        config = config.reparametrize(work_dir=job_dir,
                                      config_name=job_name,
                                      narration=narration,
                                      n_workers=n_workers,
        )

    return n_workers, job_dir, job_name, narration, config

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
@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def run(log, n_workers, checkpoint_freq, job_dir, job_name, narration, configuration,
        n_cycle_steps, run_time, start_hash, orchestrator):
    """

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
        

    Returns
    -------

    """

    set_loglevel(log)

    # settle what the defaults etc. are for the different options as they are interdependent
    n_workers, job_dir, job_name, narration, config = settle_run_options(n_workers=n_workers,
                                                                 job_dir=job_dir,
                                                                 job_name=job_name,
                                                                 narration=narration,
                                                                 configuration=configuration)

    orch = deserialize_orchestrator(orchestrator.read())

    logging.info("Orchestrator loaded")

    start_hash, end_hash = orch.orchestrate_snapshot_run_by_time(start_hash,
                                                                 run_time, n_cycle_steps,
                                                                 checkpoint_freq=checkpoint_freq,
                                                                 work_dir=job_dir,
                                                                 config_name=job_name,
                                                                 narration=narration,
                                                                 configuration=config,
                                                                 n_workers=n_workers)

    # write the run tuple out to the log
    run_line_str = "Run start and end hashes: {}, {}".format(start_hash, end_hash)

    # log it
    logging.info(run_line_str)

    # also put it to the terminal
    click.echo(run_line_str)

@click.option('--log', default="WARNING")
@click.option('--n-workers', type=click.INT)
@click.option('--checkpoint-freq', default=None, type=click.INT)
@click.option('--job-dir', default=CURDIR, type=click.Path(writable=True))
@click.option('--job-name', default=START_HASH)
@click.option('--narration', default="recovery")
@click.option('--configuration', type=click.Path(exists=True), default=None)
@click.argument('n_cycle_steps', type=click.INT)
@click.argument('run_time', type=click.FLOAT)
@click.argument('checkpoint', type=click.File(mode='rb'))
@click.argument('start_hash')
@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def recover(log, n_workers, checkpoint_freq, job_dir, job_name, narration, configuration,
            n_cycle_steps, run_time, checkpoint, start_hash, orchestrator):
    """

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
        
    checkpoint :
        
    start_hash :
        
    orchestrator :
        

    Returns
    -------

    """

    set_loglevel(log)

    n_workers, job_dir, job_name, narration, config = settle_run_options(n_workers=n_workers,
                                                                 job_dir=job_dir,
                                                                 job_name=job_name,
                                                                 narration=narration,
                                                                 configuration=configuration)

    orch = deserialize_orchestrator(orchestrator.read())

    logging.info("Orchestrator loaded")

    checkpoint_orch = deserialize_orchestrator(checkpoint.read())

    logging.info("Checkpoint loadeded")

    # run the continuation from the new orchestrator with the update
    # from the checkpoint
    new_orch, run_tup = recover_run_by_time(orch, checkpoint_orch,
                                            run_time, n_cycle_steps,
                                            checkpoint_freq=checkpoint_freq,
                                            work_dir=job_dir,
                                            config_name=job_name,
                                            narration=narration,
                                            configuration=config,
                                            n_workers=n_workers)

    start_hash, end_hash = run_tup

    # write the run tuple out to the log
    run_line_str = "Run start and end hashes: {}, {}".format(start_hash, end_hash)

    # log it
    logging.info(run_line_str)

    # also put it to the terminal
    click.echo(run_line_str)


def combine_orch_wepy_hdf5s(new_orch, new_hdf5_path):
    """

    Parameters
    ----------
    new_orch :
        
    new_hdf5_path :
        

    Returns
    -------

    """

    # a key-value for the paths for each run
    hdf5_paths = {}

    # go through each run in the new orchestrator
    for run_id in new_orch.runs:

        # get the configuration used for this run
        run_config = new_orch.run_configuration(*run_id)

        # from that configuration find the WepyHDF5Reporters
        for reporter in run_config.reporters:

            if isinstance(reporter, WepyHDF5Reporter):

                # and save the path for that run
                hdf5_paths[run_id] = reporter.file_path

    # now that we have the paths (or lack of paths) for all
    # the runs we need to start linking them all
    # together.

    # first we need a master linker HDF5 to do this with

    # so load a template WepyHDF5
    template_wepy_h5_path = hdf5_paths[new_orch.runs[0]]
    template_wepy_h5 = WepyHDF5(template_wepy_h5_path, mode='r')

    # clone it
    with template_wepy_h5:
        master_wepy_h5 = template_wepy_h5.clone(new_hdf5_path, mode='x')

    with master_wepy_h5:
        # then link all the files to it
        run_mapping = {}
        for run_id, wepy_h5_path in hdf5_paths.items():

            # we just link the whole file then sort out the
            # continuations later since we aren't necessarily doing
            # this in a logical order
            new_run_idxs = master_wepy_h5.link_file_runs(wepy_h5_path)

            # map the hash id to the new run idx created. There should
            # only be one if we are following the orchestration
            # workflow.
            run_mapping[run_id] = new_run_idxs[0]

        # now that they are all linked we need to set the
        # continuations correctly, so for each run we find the run it
        # continues in the orchestrator
        for run_id, run_idx in run_mapping.items():

            # find the run_id that this one continues
            continued_run_id = new_orch.run_continues(*run_id)

            # if a None is returned then there was no continuation
            if continued_run_id is None:
                # so we go to the next run_id and don't log any
                # continuation
                continue

            # get the run_idx in the HDF5 that corresponds to this run
            continued_run_idx = run_mapping[continued_run_id]

            # add the continuation
            master_wepy_h5.add_continuation(run_idx, continued_run_idx)

@click.command()
@click.option('--hdf5', type=click.Path(exists=False))
@click.argument('output', nargs=1, type=click.File(mode='wb'))
@click.argument('orchestrators', nargs=-1, type=click.File(mode='rb'))
def reconcile(hdf5,
              output, orchestrators):
    """

    Parameters
    ----------
    hdf5 :
        
    output :
        
    orchestrators :
        

    Returns
    -------

    """

    # reconcile them one by one as they are big and too expensive to
    # load all into memory at once
    click.echo("Deserializing Orchestrator 1")
    new_orch = deserialize_orchestrator(orchestrators[0].read())
    click.echo("Finished Deserializing Orchestrator 1")

    hash_listing_str = "\n".join(["{}, {}".format(start, end) for start, end in new_orch.runs])
    click.echo("This orchestrator has the following runs:")
    click.echo(hash_listing_str)

    for orch_idx, orchestrator in enumerate(orchestrators[1:]):
        orch_idx += 1

        click.echo("/n")
        click.echo("Deserializing Orchestrator {}", format(orch_idx))
        orch = deserialize_orchestrator(orchestrator.read())
        click.echo("Finished Deserializing Orchestrator {}", format(orch_idx))

        hash_listing_str = "\n".join(["{}, {}".format(start, end) for start, end in new_orch.runs])
        click.echo("This orchestrator has the following runs:")
        click.echo(hash_listing_str)

        # reconcile the two orchestrators
        click.echo("Reconciling this orchestrator to the new orchestrator")
        new_orch = reconcile_orchestrators(new_orch, orch)

        hash_listing_str = "\n".join(["{}, {}".format(start, end) for start, end in new_orch.runs])
        click.echo("The new orchestrator has the following runs:")
        click.echo(hash_listing_str)


    # if a path for an HDF5 file is given
    if hdf5 is not None:
        hdf5_path = osp.realpath(hdf5)
        # combine the HDF5 files from those orchestrators
        combine_orch_wepy_hdf5s(new_orch, hdf5_path)


    # then make and output the orchestrator
    output.write(new_orch.serialize())

def hash_listing_formatter(hashes):
    """

    Parameters
    ----------
    hashes :
        

    Returns
    -------

    """
    hash_listing_str = '\n'.join(hashes)
    return hash_listing_str

@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def ls_snapshots(orchestrator):
    """

    Parameters
    ----------
    orchestrator :
        

    Returns
    -------

    """

    orch = deserialize_orchestrator(orchestrator.read())
    message = hash_listing_formatter(orch.snapshot_hashes)

    click.echo(message)

@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def ls_runs(orchestrator):
    """

    Parameters
    ----------
    orchestrator :
        

    Returns
    -------

    """

    orch = deserialize_orchestrator(orchestrator.read())

    runs = orch.runs

    hash_listing_str = "\n".join(["{}, {}".format(start, end) for start, end in runs])

    click.echo(hash_listing_str)

@click.command()
@click.option('--no-expand-external', is_flag=True)
@click.argument('source', type=click.Path(exists=True))
@click.argument('target', type=click.Path(exists=False))
def copy_h5(no_expand_external, source, target):


    # arg clusters to pass to subprocess for the files
    input_f_args = ['-i', source]
    output_f_args = ['-o', target]

    # each invocation calls a different group since we can't call the
    # toplevel '/' directly
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

# command groupings
cli.add_command(run)
cli.add_command(recover)
cli.add_command(reconcile)
cli.add_command(ls_snapshots)
cli.add_command(ls_runs)
cli.add_command(copy_h5)


if __name__ == "__main__":

    cli()
