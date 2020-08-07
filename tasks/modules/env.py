from invoke import task

import sys
import os
import os.path as osp
from pathlib import Path
from warnings import warn
import shutil

from ..config import (
    ENV_METHOD,
    DEFAULT_ENV,
    ENVS_DIR,
    PYTHON_VERSION_SOURCE,
    PYTHON_VERSIONS,
)

## user config examples

# SNIPPET:

# # which virtual environment tool to use: pyenv-virtualenv, venv, or
# conda.  pyenv-virtualenv and conda works for all versions, venv only
# works with python3.3 and above ENV_METHOD = 'venv'

# # which env spec to use by default
# DEFAULT_ENV = 'dev'

# # directory where env specs are read from
# ENVS_DIR = 'envs'

# Python version source, this is how we get the different python
# versions

# PYTHON_VERSION_SOURCE = "pyenv"

# which versions to install

# PYTHON_VERSIONS = (
#     '3.8.1',
#     '3.7.6',
# )


## Constants

# directories the actual environments are stored
VENV_DIR = "_venv"
CONDA_ENVS_DIR = "_conda_envs"
# this will be set to the PYENV_PREFIX with the path to the project
# dir
PYENV_DIR = "_pyenv"



# specified names of env specs files
SELF_REQUIREMENTS = 'self.requirements.txt'
"""How to install the work piece"""

# requirements for specific packaging tooling, like pip
ENV_TOOLS_REQUIREMENTS = 'tools.requirements.txt'

PYTHON_VERSION_FILE = 'pyversion.txt'
"""Specify which version of python to use for the env"""

DEV_REQUIREMENTS_LIST = 'dev.requirements.list'
"""Multi-development mode repos to read dependencies from."""

# pip specific
PIP_ABSTRACT_REQUIREMENTS = 'requirements.in'
PIP_COMPILED_REQUIREMENTS = 'requirements.txt'

# conda specific
CONDA_ABSTRACT_REQUIREMENTS = 'env.yaml'
CONDA_COMPILED_REQUIREMENTS = 'env.pinned.yaml'

### Util

def parse_list_format(list_str):

    return [line for line in list_str.split('\n')
            if not line.startswith("#") and line.strip()]


def read_pyversion_file(py_version_path: Path):

    with open(py_version_path, 'r') as rf:
        py_version = rf.read().strip()

    return py_version


def get_current_pyversion():
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

### Dependencies
# managing dependencies for the project at runtime

## pip: things that can be controlled by pip

def deps_pip_pin(cx,
                 path=None,
                 upgrade=False):

    assert path is not None

    path = Path(path)

    # gather any development repos that are colocated on this machine
    # and solve the dependencies together

    specs = [
        str(path / PIP_ABSTRACT_REQUIREMENTS),
        str(path / ENV_TOOLS_REQUIREMENTS)
    ]

    # to get the development repos read the DEV list
    if osp.exists(path / DEV_REQUIREMENTS_LIST):
        with open(path / DEV_REQUIREMENTS_LIST) as rf:
            dev_repo_specs = parse_list_format(rf.read())

        # for each repo spec add this to the list of specs to evaluate for
        for dev_repo_spec in dev_repo_specs:
            # expand env vars
            dev_repo_spec = osp.expandvars(osp.expanduser(dev_repo_spec))

            assert osp.exists(dev_repo_spec), f"Repo spec {dev_repo_spec} doesn't exist"

            specs.append(dev_repo_spec)


    spec_str =  " ".join(specs)

    print("Using simultaneous dev specs:")
    print(spec_str)

    upgrade_str = ''
    if upgrade:
        upgrade_str = "--upgrade"


    cx.run("pip-compile "
           f"{upgrade_str} "
           f"--output-file={path}/{PIP_COMPILED_REQUIREMENTS} "
           f"{spec_str}")

    # SNIPPET: generate hashes is not working right, or just confusing me
    # cx.run("python -m piptools compile "
    #        "--generate-hashes "
    #        "--output-file={PIP_COMPILED_REQUIREMENTS} "
    #        f"{PIP_ABSTRACT_REQUIREMENTS}")


## conda: managing conda dependencies
def deps_conda_pin(cx,
                   path=None,
                   upgrade=False,
                   optional=False,
):

    # STUB: currently upgrade does nothing

    assert path is not None

    env_spec_path = Path(path)

    if not optional:
        assert osp.exists(env_spec_path / CONDA_ABSTRACT_REQUIREMENTS), \
            "There must be an 'env.yaml' file to compile from"

    else:
        if not osp.exists(env_spec_path / CONDA_ABSTRACT_REQUIREMENTS):
            return None

    # delete the pinned file
    if osp.exists(env_spec_path / CONDA_COMPILED_REQUIREMENTS):
        os.remove(env_spec_path / CONDA_COMPILED_REQUIREMENTS)

    # make the environment under a mangled name so we don't screw with
    # the other one
    mangled_name = f"__mangled_tmp_env"

    mangled_env_spec_path = Path(ENVS_DIR) / mangled_name

    # remove if there is one already there
    if osp.exists(mangled_env_spec_path):
        shutil.rmtree(mangled_env_spec_path)

    # make sure the new dir is initialized
    os.makedirs(mangled_env_spec_path,
                exist_ok=True
    )

    # TODO: simultaneous dev requirements from other projects that
    # require a conda install

    # copy the 'env.yaml' and 'pyversion.txt' files to the new env. If
    # we include the requirements.txt file it will screw with the
    # pinning process
    shutil.copyfile(
        env_spec_path / CONDA_ABSTRACT_REQUIREMENTS,
        mangled_env_spec_path / CONDA_ABSTRACT_REQUIREMENTS,
    )

    if osp.exists(env_spec_path / PYTHON_VERSION_FILE):
        shutil.copyfile(
            env_spec_path / PYTHON_VERSION_FILE,
            mangled_env_spec_path / PYTHON_VERSION_FILE,
        )


    # then create the mangled env
    env_dir = conda_env(cx,
                        spec=mangled_env_spec_path,
                        path=Path(CONDA_ENVS_DIR) / mangled_name,
    )

    # then install the packages so we can export them
    with cx.prefix(f'eval "$(conda shell.bash hook)" && conda activate {env_dir}'):

        # only install the declared dependencies
        cx.run(f"conda env update "
               f"--prefix {env_dir} "
               f"--file {env_spec_path}/{CONDA_ABSTRACT_REQUIREMENTS}")

    # pin to a 'env.pinned.yaml' file
    cx.run(f"conda env export "
           f"-p {env_dir} "
           f"-f {env_spec_path}/{CONDA_COMPILED_REQUIREMENTS}")

    # then destroy the temporary mangled env and spec
    shutil.rmtree(env_dir)
    shutil.rmtree(mangled_env_spec_path)

    print("--------------------------------------------------------------------------------")
    print(f"This is an automated process do not attempt to activate the '__mangled' environment")

@task
def deps_pin_path(cx,
                  path=None,
                  upgrade=False):
    """Pin an environment given by the path."""

    deps_pip_pin(cx,
                 path=path,
                 upgrade=False)

    if ENV_METHOD == 'conda':

        # WKRD, FIXME: disabled for now since the end result isn't useful anymore
        pass
        # deps_conda_pin(cx,
        #                path=path,
        #                upgrade=False,
        #                optional=True,)


# altogether
@task
def deps_pin(cx, name=DEFAULT_ENV):
    """Pin an environment in the 'envs' directory."""

    path = Path(ENVS_DIR) / name

    deps_pin_path(cx, path=path)


@task
def deps_pin_update(cx, name=DEFAULT_ENV):
    """Update the pinned environment in the 'envs' directory."""

    path = Path(ENVS_DIR) / name

    deps_pin_path(cx,
                  path=path,
                  upgrade=True,
    )


### Environments

def conda_env(cx,
              spec=None,
              path=None,
):

    # where the specs of the environment are
    env_spec_path = Path(spec)

    # using the local envs dir
    env_dir = Path(path)

    # ensure the directory
    cx.run(f"mkdir -p {env_dir}")


    # clean up old envs if they weren't already
    if osp.exists(env_dir):
        shutil.rmtree(env_dir)

    # figure out which python version to use, if the 'pyversion.txt'
    # file exists read it
    py_version_path = env_spec_path / PYTHON_VERSION_FILE
    if osp.exists(py_version_path):

        print("Using specified python version")

        py_version = read_pyversion_file(py_version_path)

    # otherwise use the one you are currently using
    else:
        print("Using current envs python version")
        py_version = get_current_pyversion()

    print(f"Using python version: {py_version}")

    # create the environment
    cx.run(f"conda create -y "
           f"--prefix {env_dir} "
           f"python={py_version}",
        pty=True)

    with cx.prefix(f'eval "$(conda shell.bash hook)" && conda activate {env_dir}'):

        # install the conda dependencies. choose a specification file
        # based on these priorities of most pinned to least frozen.

        # WKRD, FIXME: this is disabled because this is super-platform
        # dependent and not reliable
        #
        # if osp.exists(env_spec_path / CONDA_COMPILED_REQUIREMENTS):
        #     cx.run(f"conda env update "
        #            f"--prefix {env_dir} "
        #            f"--file {env_spec_path}/{CONDA_COMPILED_REQUIREMENTS}")


        if osp.exists(env_spec_path / CONDA_ABSTRACT_REQUIREMENTS):

            cx.run(f"conda env update "
                   f"--prefix {env_dir} "
                   f"--file {env_spec_path}/{CONDA_ABSTRACT_REQUIREMENTS}")

        else:
            print("No conda dependencies specified")
            # don't do a conda env pin
            pass


        # install the tooling, like pip version etc.
        if osp.exists(env_spec_path / ENV_TOOLS_REQUIREMENTS):
            cx.run(f"{env_dir}/bin/pip install -r {env_spec_path}/{ENV_TOOLS_REQUIREMENTS}")

        # install the extra pip dependencies.

        # this ignores anything already installed by the conda env
        # which can cause problem if this tries to install it again.
        if osp.exists(env_spec_path / PIP_COMPILED_REQUIREMENTS):
            cx.run(f"{env_dir}/bin/pip install "
                   "--ignore-installed "
                   f"-r {env_spec_path}/{PIP_COMPILED_REQUIREMENTS}")

        # install the package itself
        if osp.exists(env_spec_path / SELF_REQUIREMENTS):
            cx.run(f"{env_dir}/bin/pip install -r {env_spec_path}/{SELF_REQUIREMENTS}")

    print("--------------------------------------------------------------------------------")
    print(f"run: conda activate {env_dir}")

    return env_dir


def venv_env(cx,
             spec=None,
             path=None,
):

    assert spec is not None
    assert path is not None

    venv_path = Path(path)
    env_spec_path = Path(spec)

    # ensure the directory
    cx.run(f"mkdir -p {venv_path}")

    py_version_path = env_spec_path / PYTHON_VERSION_FILE

    py_version = get_current_pyversion()
    if osp.exists(py_version_path):

        spec_py_version = read_pyversion_file(py_version_path)
        if spec_py_version != py_version:
            raise ValueError(
                f"Python version {spec_py_version} was specified in {PYTHON_VERSION_FILE} "
                f"but Python {py_version} is activated. For the venv method you must have "
                f"the desired python version already activated"
        )


    print(f"Using python version: {py_version}")

    # create the env requested
    cx.run(f"python -m venv {venv_path}")

    # then install the things we need
    with cx.prefix(f"source {venv_path}/bin/activate"):

        # install the tooling, like pip version etc.
        if osp.exists(env_spec_path / ENV_TOOLS_REQUIREMENTS):
            cx.run(f"{env_dir}/bin/pip install -r {env_spec_path}/{ENV_TOOLS_REQUIREMENTS}")

        # install the pip pinned requirements
        if osp.exists(env_spec_path / PIP_COMPILED_REQUIREMENTS):
            cx.run(f"pip install -r {env_spec_path}/{PIP_COMPILED_REQUIREMENTS}")

        else:
            print("No requirements.txt found")

        # if there is a 'self.requirements.txt' file specifying how to
        # install the package that is being worked on install it
        if osp.exists(env_spec_path / SELF_REQUIREMENTS):
            cx.run(f"pip install -r {env_spec_path}/{SELF_REQUIREMENTS}")

        else:
            print("No self.requirements.txt found")

    print("----------------------------------------")
    print("to activate run:")
    print(f"source {venv_path}/bin/activate")

    return venv_path

def pyenv_env(cx,
              spec=None,
              path=None,
):

    assert spec is not None
    assert path is not None

    # project has its own pyenv root
    pyenv_local_dir = Path(path)

    env_spec_path = Path(spec)

    # ensure the directory
    cx.run(f"mkdir -p {pyenv_local_dir}")


    env_path = f"{pyenv_local_dir}/{name}"
    cx.run("rm -rf {env_path}")

    py_version_path = env_spec_path / PYTHON_VERSION_FILE

    if osp.exists(py_version_path):

        py_version = read_pyversion_file(py_version_path)

    # otherwise use the one you are currently using
    else:
        py_version = get_current_pyversion()

    print(f"Using python version: {py_version}")

    # TODO: use pyenv to make the virtualenv with the right version

    # if you already have pyenv installed and there are versions of
    # python installed there we will preferentially use them while
    # ignoring the envs there since that is a lot of unnecessary files
    # to have all the pythons installed separately.

    pyenv_root = Path(osp.expandvars("$PYENV_ROOT"))

    # go ahead and use the pyenv-virtual-local command
    if pyenv_root.exists():
        cx.run(f"pyenv virtualenv-local \\"
               f"--alt-dir {pyenv_local_dir} \\"
               f"{py_version} \\"
               f"{name}"
        )


    # currently we dont' support installing it
    else:
        raise FileNotFoundError(
            f"pyenv not installed"
            )

    # then install the things we need
    with cx.prefix(f"source {env_path}/bin/activate"):

        # install the tooling, like pip version etc.
        if osp.exists(env_spec_path / ENV_TOOLS_REQUIREMENTS):
            cx.run(f"{env_dir}/bin/pip install -r {env_spec_path}/{ENV_TOOLS_REQUIREMENTS}")

        # install the pinned packages
        if osp.exists(env_spec_path / SELF_REQUIREMENTS):
            cx.run(f"pip install -r {env_spec_path}/{PIP_COMPILED_REQUIREMENTS}")

        else:
            print("No requirements.txt found")

        # if there is a 'self.requirements.txt' file specifying how to
        # install the package that is being worked on install it
        if osp.exists(env_spec_path / SELF_REQUIREMENTS):
            cx.run(f"pip install -r {env_spec_path}/{SELF_REQUIREMENTS}")

        else:
            print("No self.requirements.txt found")

    print("----------------------------------------")
    print("to activate run:")
    print(f"source {env_path}/bin/activate")

    return env_path

@task
def make_env(cx,
             spec=None,
             path=None,
             venv=ENV_METHOD,
):

    assert spec is not None
    assert path is not None

    # choose your method:
    if venv == 'conda':
        conda_env(cx,
                  spec=spec,
                  path=path,
        )

    elif venv == 'venv':
        venv_env(cx,
                  spec=spec,
                  path=path,
        )


    elif venv == 'pyenv':
        pyenv_env(cx,
                  spec=spec,
                  path=path,
        )


@task(default=True)
def make(cx, name=DEFAULT_ENV):

    spec_path = Path(ENVS_DIR) / name

    # get the path to your envs based on the method
    if ENV_METHOD == 'conda':
        env_path = Path(CONDA_ENVS_DIR) / name

    elif ENV_METHOD == 'venv':
        env_path = Path(VENV_DIR) / name

    elif ENV_METHOD == 'pyenv':
        env_path = Path(PYENV_DIR) / name

    else:
        raise ValueError(f"Unrecognized venv type: {ENV_METHOD}")

    make_env(cx,
             spec=spec_path,
             path=env_path,
             venv = ENV_METHOD,
    )

@task
def ls_conda(cx):
    print('\n'.join(os.listdir(CONDA_ENVS_DIR)))

@task
def ls_venv(cx):

    print('\n'.join(os.listdir(VENV_DIR)))

@task
def ls_specs(cx):

    print('\n'.join(os.listdir(ENV_SPEC_DIR)))

@task
def ls(cx):

    # choose your method:
    if ENV_METHOD == 'conda':
        ls_conda(cx)

    elif ENV_METHOD == 'venv':
        ls_venv(cx)

@task
def clean(cx):
    cx.run(f"rm -rf {VENV_DIR}")


@task
def install_pythons(cx):
    """Install different python versions."""

    assert PYTHON_VERSION_SOURCE == 'pyenv', \
        "Only pyenv is supported for different python versions"

    with cx.prefix("unset PYENV_VERSION"):
        for version in PYTHON_VERSIONS:
            cx.run(f"pyenv install --skip-existing {version}",
                   warn=True)
