"""Glue all the CLIs together into one interface."""

import click

from wepy.orchestration.cli import subgroups as orch_subgroups
from wepy.orchestration.cli import cli as orch_cli

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

    orch_cli()
