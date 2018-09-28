import os.path as osp
import logging

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
    pass

def set_loglevel(loglevel):

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
                       narration=None):

    # the default for the job name is the start hash if none is given
    if job_name == START_HASH:
        job_name = start_hash

    # if the job_name is given and the default value for th job_dir is
    # given we set the job-dir as the job_name
    if job_name is not None and job_dir == CURDIR:
            job_dir = job_name

    # if the special value for curdir is given we get the systems
    # current directory, this is the default.
    if job_dir == CURDIR:
        job_dir = osp.curdir

    # normalize the job_dir
    job_dir = osp.realpath(job_dir)

    return n_workers, job_dir, job_name, narration

@click.option('--log', default="WARNING")
@click.option('--n-workers', type=click.INT)
@click.option('--checkpoint-freq', default=None, type=click.INT)
@click.option('--job-dir', default=CURDIR, type=click.Path(writable=True))
@click.option('--job-name', default=START_HASH)
@click.option('--narration', default="")
@click.argument('n_cycle_steps', type=click.INT)
@click.argument('run_time', type=click.FLOAT)
@click.argument('start_hash')
@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def run(log, n_workers, checkpoint_freq, job_dir, job_name, narration,
        n_cycle_steps, run_time, start_hash, orchestrator):

    set_loglevel(log)

    # settle what the defaults etc. are for the different options as they are interdependent
    n_workers, job_dir, job_name, narration = settle_run_options(n_workers=n_workers,
                                                                 job_dir=job_dir,
                                                                 job_name=job_name,
                                                                 narration=narration)

    orch = deserialize_orchestrator(orchestrator.read())

    logging.info("Orchestrator loaded")

    start_hash, end_hash = orch.orchestrate_snapshot_run_by_time(start_hash,
                                                                 run_time, n_cycle_steps,
                                                                 checkpoint_freq=checkpoint_freq,
                                                                 work_dir=job_dir,
                                                                 config_name=job_name,
                                                                 narration=narration,
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
@click.argument('n_cycle_steps', type=click.INT)
@click.argument('run_time', type=click.FLOAT)
@click.argument('checkpoint', type=click.File(mode='rb'))
@click.argument('start_hash')
@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def recover(log, n_workers, checkpoint_freq, job_dir, job_name, narration,
            n_cycle_steps, run_time, checkpoint, start_hash, orchestrator):

    set_loglevel(log)

    n_workers, job_dir, job_name, narration = settle_run_options(n_workers=n_workers,
                                                                 job_dir=job_dir,
                                                                 job_name=job_name,
                                                                 narration=narration)

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
                                            narration=narration)

    start_hash, end_hash = run_tup

    # write the run tuple out to the log
    run_line_str = "Run start and end hashes: {}, {}".format(start_hash, end_hash)

    # log it
    logging.info(run_line_str)

    # also put it to the terminal
    click.echo(run_line_str)


def combine_orch_wepy_hdf5s(new_orch, new_hdf5_path):

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

    # reconcile them one by one as they are big and too expensive to
    # load all into memory at once
    new_orch = deserialize_orchestrator(orchestrators[0].read())
    for orchestrator in orchestrators[1:]:
        orch = deserialize_orchestrator(orchestrator.read())

        # reconcile the two orchestrators
        new_orch = reconcile_orchestrators(new_orch, orch)

    # if a path for an HDF5 file is given
    if hdf5 is not None:
        hdf5_path = osp.realpath(hdf5)
        # combine the HDF5 files from those orchestrators
        combine_orch_wepy_hdf5s(new_orch, hdf5_path)


    # then make and output the orchestrator
    output.write(new_orch.serialize())

def hash_listing_formatter(hashes):
    hash_listing_str = '\n'.join(hashes)
    return hash_listing_str

@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def ls_snapshots(orchestrator):

    orch = deserialize_orchestrator(orchestrator.read())
    message = hash_listing_formatter(orch.snapshot_hashes)

    click.echo(message)

@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def ls_runs(orchestrator):

    orch = deserialize_orchestrator(orchestrator.read())

    runs = orch.runs

    hash_listing_str = "\n".join(["{}, {}".format(start, end) for start, end in runs])

    click.echo(hash_listing_str)

# command groupings
cli.add_command(run)
cli.add_command(recover)
cli.add_command(reconcile)
cli.add_command(ls_snapshots)
cli.add_command(ls_runs)

if __name__ == "__main__":

    cli()
