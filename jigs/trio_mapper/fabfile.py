from fabric import task
from patchwork.transfers import rsync

import os
import os.path as osp
from pathlib import Path

@task
def sanity(cx):

    cx.run("echo hello")

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
