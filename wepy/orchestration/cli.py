import pickle
import os.path as osp

import click

from wepy.orchestration.orchestrator import orchestrate_run_by_time, \
                                            reconcile_orchestrators, \
                                            Orchestrator

ORCHESTRATOR_DEFAULT_FILENAME = \
            Orchestrator.ORCH_FILENAME_TEMPLATE.format(config=Orchestrator.DEFAULT_CONFIG_NAME,
                                                       narration=Orchestrator.DEFAULT_NARRATION)

@click.group()
def cli():
    pass

START_HASH = '<start_hash>'

@click.option('--run-log', default='-', type=click.File(mode='r+'))
@click.option('--checkpoint-freq', default=10, type=click.INT)
@click.option('--job-dir', default=osp.realpath(osp.curdir), type=click.Path(writable=True))
@click.option('--job-name', default=START_HASH)
@click.option('--narration', default="")
@click.argument('n_cycle_steps', type=click.INT)
@click.argument('run_time', type=click.FLOAT)
@click.argument('start_hash')
@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def run(run_log,
        checkpoint_freq, job_dir, job_name, narration,
        n_cycle_steps, run_time, start_hash, orchestrator):

    if job_name == START_HASH:
        job_name = start_hash

    orch = pickle.load(orchestrator)

    result = orch.orchestrate_snapshot_run_by_time(start_hash,
                                                    run_time, n_cycle_steps,
                                                    checkpoint_freq=checkpoint_freq,
                                                    work_dir=job_dir,
                                                    config_name=job_name,
                                                    narration=narration)

    run_tup, mutation_tup = result

    end_hash = run_tup[1]

    # write the run tuple out to the log
    run_line_str = "{}, {}".format(start_hash, end_hash)
    run_log.write(run_line_str + "\n")

@click.command()
@click.argument('orchestrator_a', type=click.File(mode='rb'))
@click.argument('orchestrator_b', type=click.File(mode='rb'))
@click.argument('output', type=click.File(mode='wb'))
def reconcile(orchestrator_a, orchestrator_b, output):

    # read in the two orchestrator pickles
    orch_a = pickle.load(orchestrator_a)
    orch_b = pickle.load(orchestrator_b)

    # reconcile the two orchestrators
    new_orch = reconcile_orchestrators(orch_a, orch_b)

    # then make and output pickle
    pickle.dump(new_orch, output)

@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def ls_snapshots(orchestrator):
    orch = pickle.load(orchestrator)

    click.echo(str(orch.snapshot_hashes))

@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def ls_apparatuses(orchestrator):
    orch = pickle.load(orchestrator)

    click.echo(str(orch.apparatus_hashes))


@click.argument('orchestrator', type=click.File(mode='rb'))
@click.command()
def ls_runs(orchestrator):
    orch = pickle.load(orchestrator)

    click.echo(str(orch.runs))

# command groupings
cli.add_command(run)
cli.add_command(reconcile)
cli.add_command(ls_snapshots)
cli.add_command(ls_apparatuses)
cli.add_command(ls_runs)
#cli.add_command(ls_segments)
#cli.add_command(last_checkpoint)

if __name__ == "__main__":

    cli()
