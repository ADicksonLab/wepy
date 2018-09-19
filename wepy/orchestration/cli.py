import os.path as osp

import click

from wepy.orchestration.orchestrator import deserialize_orchestrator, \
                                            reconcile_orchestrators, \
                                            Orchestrator, \
                                            recover_run_by_time

ORCHESTRATOR_DEFAULT_FILENAME = \
            Orchestrator.ORCH_FILENAME_TEMPLATE.format(config=Orchestrator.DEFAULT_CONFIG_NAME,
                                                       narration=Orchestrator.DEFAULT_NARRATION)

@click.group()
def cli():
    pass

START_HASH = '<start_hash>'
CURDIR = '<curdir>'

def settle_run_options(job_dir, job_name, narration):

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

    return job_dir, job_name, narration

@click.option('--checkpoint-freq', default=None, type=click.INT)
@click.option('--job-dir', default=CURDIR, type=click.Path(writable=True))
@click.option('--job-name', default=START_HASH)
@click.option('--narration', default="")
@click.argument('n_cycle_steps', type=click.INT)
@click.argument('run_time', type=click.FLOAT)
@click.argument('start_hash')
@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def run(checkpoint_freq, job_dir, job_name, narration,
        n_cycle_steps, run_time, start_hash, orchestrator):

    job_dir, job_name, narration = settle_run_options(job_dir, job_name, narration)

    orch = deserialize_orchestrator(orchestrator.read())

    start_hash, end_hash = orch.orchestrate_snapshot_run_by_time(start_hash,
                                                    run_time, n_cycle_steps,
                                                    checkpoint_freq=checkpoint_freq,
                                                    work_dir=job_dir,
                                                    config_name=job_name,
                                                    narration=narration)

    # write the run tuple out to the log
    run_line_str = "{}, {}".format(start_hash, end_hash)
    click.echo(run_line_str)

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
def recover(checkpoint_freq, job_dir, job_name, narration,
            n_cycle_steps, run_time, checkpoint, start_hash, orchestrator):

    job_dir, job_name, narration = settle_run_options(job_dir, job_name, narration)

    orch = deserialize_orchestrator(orchestrator.read())

    checkpoint_orch = deserialize_orchestrator(checkpoint.read())

    # run the continuation from the new orchestrator with the update
    # from the checkpoint
    new_orch, run_tup = recover_run_by_time(orch, checkpoint_orch,
                                            run_time, n_cycle_steps,
                                            checkpoint_freq=checkpoint_freq,
                                            work_dir=job_dir,
                                            config_name=job_name,
                                            narration=narration)

    # write the run tuple out to the log
    run_line_str = "{}, {}".format(*run_tup)
    click.echo(run_line_str)


@click.command()
@click.option('--hdf5', type=click.Path(exists=False))
@click.argument('orchestrators', nargs=-1, type=click.File(mode='rb'))
@click.argument('output', nargs=1, type=click.File(mode='wb'))
def reconcile(hdf5,
              orchestrators, output):

    orches = []
    for orchestrator in orchestrators:
        orch = deserialize_orchestrator(orchestrator.read())
        orches.append(orch)

    # reconcile the two orchestrators
    new_orch = reconcile_orchestrators(*orchs)

    # combine the HDF5 files from those orchestrators
    if hdf5 is not None:
        # the path the new linker HDF5 will be in
        new_hdf5_path = hdf5

        for run_id in new_orch.runs:

            run_config 
            for reporter in run_config.reporters:
                if isinstance(reporter, WepyHDF5Reporter):
                    hdf5_paths[(orch_idx, run_idx)] = reporter


        # go through each orchestrator and if they have an HDF5
        # reporter use that path to find the HDF5 file they made
        hdf5_paths = {}

        # create the linker HDF5 file
        

    # then make and output the orchestrator
    output.write(new_orch.serialize())

    # output the linker HDF5 if necessary

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
