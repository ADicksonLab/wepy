from fabric import task
from patchwork.transfers import rsync

import os
import os.path as osp
from pathlib import Path
import json

## START EDIT: Edit these values to your profiles

# name of the bimhaw profile used in the bootstrapping process
PHASE1_PROFILE = "scooter"

# name of the bimhaw profile that is the end state target
FINAL_PROFILE = "bronco"

# name of the loadout (collection of installed programs) to install
FINAL_LOADOUT = "humvee"

## END EDIT

@task
def packages_bootstrap(cx):
    """Install the main distro provided packages"""

    packages = [
        'git',
        'rsync',
        'mg',
        'python3-venv',
    ]

    cx.run(f"sudo apt install -y {' '.join(packages)}")


@task
def bimhaw_bootstrap(cx):
    """Configure bimhaw and shells in system to PHASE 1.

    PHASE 1 is a loadout profile which is used to then install more
    advanced environments.

    """

    # make a python virtual env to install it at first and set the
    # shells
    cx.run("mkdir -p $HOME/tmp")
    cx.run("python3 -m venv ~/tmp/py_bootstrap_env")

    # run the bimhaw initialization stuff
    with cx.prefix(". tmp/py_bootstrap_env/bin/activate"):

        cx.run("rm -rf $HOME/.bimhaw")

        cx.run("pip install git+https://github.com/salotz/bimhaw.git")
        cx.run("python -m bimhaw.init "
               "--config ~/.salotz.d/bimhaw/config.py "
               "--lib ~/.salotz.d/bimhaw/lib")

        # cx.run("rm $HOME/.profile ~/.bashrc ~/.bash_logout")

        cx.run("bimhaw link-shells --force && "
               f"bimhaw profile -n {PHASE1_PROFILE}"
        )

@task
def pyenv_bootstrap(cx):
    with cx.cd("$HOME/.salotz.d"):
        cx.run("/bin/bash -l -c './lib/installers/python_sys.sh'")
        cx.run("/bin/bash -l -c './lib/installers/python_pyenv.bash'")
        cx.run("/bin/bash -l -c './lib/setups/python_pyenv.bash'")

        cx.run("/bin/bash -l -c 'pip3 install invoke'")
        cx.run("/bin/bash -l -c './lib/setups/python_virtualenv.sh'")

@task
def install_loadout(cx):

    with cx.cd(f"$HOME/.salotz.d"):
        cx.run(f"/bin/bash -l -c 'make -f lib/loadouts/{{FINAL_LOADOUT}}.mk all'")

        # load the bimhaw profile
        cx.run(f"/bin/bash -l -c './lib/setups/bimhaw.sh && bimhaw profile -n {FINAL_PROFILE}'")

@task()
def bootstrap(cx):
    """Take a container from blank to loaded out."""

    packages_bootstrap(cx)
    push_profile(cx)
    bimhaw_bootstrap(cx)
    pyenv_bootstrap(cx)
    install_loadout(cx)

@task
def push_profile(cx):
    """Push my configuration from local user. Assumes bootstrap install is
    done."""

    homedir = osp.expandvars('$HOME')
    rsync(cx,
          f"{homedir}/.salotz.d",
          f"{homedir}/",
          rsync_opts="-ahi --stats",
    )



@task
def push_project(cx):

    # get the directory to push to
    target_dir = Path(os.getcwd()).parent

    cx.run(f"mkdir -p {target_dir}")

    rsync(cx,
          os.getcwd(),
          target_dir,
          rsync_opts="-ahi --stats --filter=':- .gitignore'",
    )


@task
def pull_project(cx):

    # get the directory to push to
    target_dir = Path(os.getcwd()).parent

    rsync(cx,
          os.getcwd(),
          target_dir,
          rsync_opts="-ahi --stats --filter=':- .gitignore' --update",
    )
