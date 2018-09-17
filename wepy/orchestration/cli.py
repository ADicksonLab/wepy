import os.path as osp

import click

from wepy.orchestration.orchestrator import deserialize_orchestrator, \
                                            reconcile_orchestrators, \
                                            Orchestrator

ORCHESTRATOR_DEFAULT_FILENAME = \
            Orchestrator.ORCH_FILENAME_TEMPLATE.format(config=Orchestrator.DEFAULT_CONFIG_NAME,
                                                       narration=Orchestrator.DEFAULT_NARRATION)

@click.group()
def cli():
    pass

START_HASH = '<start_hash>'

@click.option('--checkpoint-freq', default=None, type=click.INT)
@click.option('--job-dir', default=osp.realpath(osp.curdir), type=click.Path(writable=True))
@click.option('--job-name', default=START_HASH)
@click.option('--narration', default="")
@click.argument('n_cycle_steps', type=click.INT)
@click.argument('run_time', type=click.FLOAT)
@click.argument('start_hash')
@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def run(checkpoint_freq, job_dir, job_name, narration,
        n_cycle_steps, run_time, start_hash, orchestrator):

    if job_name == START_HASH:
        job_name = start_hash

    # normalize the work dir
    if job_dir is not None:
        job_dir = osp.realpath(job_dir)

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

@click.command()
@click.argument('orchestrator_a', type=click.File(mode='rb'))
@click.argument('orchestrator_b', type=click.File(mode='rb'))
@click.argument('output', type=click.File(mode='wb'))
def reconcile(orchestrator_a, orchestrator_b, output):

    # read in the two orchestrators
    orch_a = deserialize_orchestrator(orchestrator_a.read())
    orch_b = deserialize_orchestrator(orchestrator_b.read())

    # reconcile the two orchestrators
    new_orch = reconcile_orchestrators(orch_a, orch_b)

    # then make and output
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


@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def ls_segments(orchestrator):

    orch = deserialize_orchestrator(orchestrator.read())

    runs = orch.segments

    hash_listing_str = "\n".join(["{}, {}".format(start, end) for start, end in runs])

    click.echo(hash_listing_str)

# command groupings
cli.add_command(run)
cli.add_command(reconcile)
cli.add_command(ls_snapshots)
cli.add_command(ls_runs)
cli.add_command(ls_segments)
#cli.add_command(last_checkpoint)

if __name__ == "__main__":

    cli()
