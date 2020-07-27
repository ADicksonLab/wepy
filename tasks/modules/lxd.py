from invoke import task

from ..config import (
    PROJECT_SLUG,
)

@task
def copy_ssh(cx, name='dev'):
    """Copy SSH keys to a container."""

    cx.run(f'ssh-keygen -f "$HOME/.ssh/known_hosts" -R "{PROJECT_SLUG}.dev.lxd"')
    cx.run(f"ssh-copy-id {PROJECT_SLUG}.{name}.lxd")

@task
def push_profile(cx, name='dev'):
    """Update your dotfiles in a container."""

    cx.run(f"fab -H {PROJECT_SLUG}.{name} push-profile")

@task
def bootstrap(cx, name='dev'):
    """Bootstrap the container from a bare image.

    Not necessary if you started from a premade dev env container.

    """

    cx.run(f"fab -H {PROJECT_SLUG}.{name} bootstrap")

@task
def push(cx, name='dev'):
    """Push the files for this project

    Ignores according to the gitignore file

    """

    cx.run(f"fab -H {PROJECT_SLUG}.{name} push-project")

@task
def pull(cx, name='dev'):
    """Pull the files for this project

    Ignores according to the gitignore file

    """

    cx.run(f"fab -H {PROJECT_SLUG}.{name} pull-project")
