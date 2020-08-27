"""Glue all the CLIs together into one interface."""

from wepy.orchestration.cli import cli as orch_cli

cli = orch_cli


# SNIPPET: I was intending to aggregate multiple command lines other
# than the orchestration, but this never materialized or was
# needed. In the future though this can be the place for that.

# @click.group()
# def cli():
#     """ """
#     pass

# # add in the sub-clis
# cli.add_command(orch_cli)

# # the orchestrator stuff we keep in the top-level still though
# for subgroup in orch_subgroups:
#     cli.add_command(subgroup)

if __name__ == "__main__":

    cli()
