from invoke import task

from ..config import (
    INITIAL_VERSION,
    GIT_LFS_TARGETS,
    VERSION,
)

## Constants

VCS_RELEASE_TAG_TEMPLATE = "v{}"

@task
def lfs_track(cx):
    """Update all the files that need tracking via git-lfs."""

    for lfs_target in GIT_LFS_TARGETS:
        cx.run("git lfs track {}".format(lfs_target))


@task
def init(cx):

    tag_string = VCS_RELEASE_TAG_TEMPLATE.format(INITIAL_VERSION)

    cx.run("git init && "
           "git add -A && "
           "git commit -m 'initial commit' && "
           f"git tag -a {tag_string} -m 'initialization release'")



@task
def publish(cx):

    tag_string = VCS_RELEASE_TAG_TEMPLATE.format(VERSION)

    cx.run(f"git push origin {tag_string}")


@task
def release(cx):

    tag_string = VCS_RELEASE_TAG_TEMPLATE.format(VERSION)

    print("Releasing: ", VERSION, "with tag: ", tag_string)

    cx.run(f"git tag -a {tag_string} -m 'See the changelog for details'")
