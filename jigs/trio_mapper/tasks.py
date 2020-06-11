from invoke import task

import os
import os.path as osp
from pathlib import Path

def tangle_orgfile(cx, file_path):
    """Tangle the target file using emacs in batch mode. Implicitly dumps
    things relative to the file."""

    cx.run(f"emacs -Q --batch -l org {file_path} -f org-babel-tangle")

@task
def init(cx):
    cx.run("mkdir -p _tangle_source")
    cx.run("mkdir -p _output")

@task
def clean(cx):
    cx.run("rm -rf _tangle_source")
    cx.run("rm -rf _output")

@task(pre=[init])
def tangle(cx):
    tangle_orgfile(cx, "README.org")
    cx.run(f"chmod ug+x ./_tangle_source/*.bash", warn=True)
    cx.run(f"chmod ug+x ./_tangle_source/*.sh", warn=True)
    cx.run(f"chmod ug+x ./_tangle_source/*.py", warn=True)

@task
def clean_env(cx):
    cx.run("rm -rf _env")

@task(pre=[init])
def env(cx):
    """Create the environment from the specs in 'env'. Must have the
    entire repository available as it uses the tooling from it.

    """

    jig_name = Path(os.getcwd()).stem

    with cx.cd("../../"):
        cx.run(f"inv py.env-jig -n {jig_name}")


## Custom

CONTAINER_TOOL = "podman"

@task
def compose_up(cx):
    cx.run(f"{CONTAINER_TOOL}-compose -f input/compose/compose.yaml up -d")

@task
def compose_down(cx):
    cx.run(f"{CONTAINER_TOOL}-compose -f input/compose/compose.yaml down")

