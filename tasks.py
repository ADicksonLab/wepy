from invoke import task
import sys

import conda.cli.python_api as Conda

SHELL = "/bin/bash"
ENV = "wepy-dev"
CONDA = "${CONDA_EXE}"

CONDA_DEPS = ['pandoc']
OMNIA_DEPS = ['openmm']

@task
def deps_dev(ctx):
    """Install extra dependencies not available through python package management."""

    ctx.run("conda install {}".format(' '.join(CONDA_DEPS)))

@task
def deps_omnia(ctx):
    """Install dependencies from the omnia conda channel."""

    ctx.run("conda install {}".format(' '.join(OMNIA_DEPS)))

@task
def what_version(ctx):
    """Tell me what version the project is at."""

    # get the current version
    import wepy
    print(wepy.__version__)

@task
def set_version(ctx):
    """Set the version with a custom string."""

    NotImplemented

@task
def inc_version(ctx, level='patch'):
    """Incrementally increase the version number by specifying the bumpversion level."""

    NotImplemented

    # use the bumpversion utility
    ctx.run("bumpversion {}".format(level))

    # tag the git repo
    ctx.run("git tag -a ")


@task
def clean_build(ctx):
    """Remove all build products."""

    ctx.run("rm -rf dist")

@task
def clean_cache(ctx):
    """Remove all of the __pycache__ files in the packages."""

    NotImplemented

@task
def clean_docs(ctx):
    """Remove all documentation build products"""

    NotImplemented

@task
def sdist(ctx):
    ctx.run("python setup.py sdist")

@task(pre=[sdist])
def upload_pypi(ctx):
    ctx.run('twine upload dist/*')

@task
def docs_build(ctx):
    """Buld the documenation"""
    ctx.run("./sphinx/build.sh")

@task
def docs_deploy(ctx):
    """Deploy the documentation onto the server."""
    ctx.run("./sphinx/deploy.sh")


@task
def tests_submodule(ctx):
    """Retrieve the tests submodule if not already cloned."""

    ctx.run("git submodule update --init --recursive")
    ctx.run("git -C wepy-tests checkout master")

@task
def tests_interactive(ctx):
    """Run the interactive tests so we can play with things."""

    ctx.run("pytest -m 'interactive'")

@task
def tests_automatic(ctx):
    """Run the automated tests."""

    ctx.run("pytest -m 'not interactive'")



@task
def tests_tox(ctx):

    NotImplemented

    TOX_PYTHON_DIR=None

    ctx.run("env PATH=\"{}/bin:$PATH\" tox".format(
        TOX_PYTHON_DIR))


@task
def lint(ctx):

    ctx.run("flake8 src/wepy")
