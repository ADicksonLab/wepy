"""Tasks for managing containers and clusters.

While a lot of this might be able to be done with special purpose
tools we try to cover as many things that we have tried.

Warning
-------

This does not cover best practices at this time.

"""
import os
from pathlib import Path

from invoke import task


from ..config import (
    PROJECT_SLUG,
    CONTAINER_TOOL,
)

## Container definitions

@task
def build(cx, root=None):
    """Build all containers in dir `containers` using Dockerfiles."""

    assert root is not None, \
        "Must provide a root directory with expected structure."

    jig_name = Path(root).stem

    containers_dir = Path(root) / "input/containers"

    cx.run(f"mkdir -p {root}/_output/containers")

    print(containers_dir)
    for container in os.listdir(containers_dir):

        container_dir = containers_dir / container

        image_name = f"{PROJECT_SLUG}-{jig_name}-{container}"

        print(f"making: {image_name}")

        # remove if already in there
        cx.run(f"{CONTAINER_TOOL} rmi {image_name}", warn=True)

        # rebuild
        cx.run(f"{CONTAINER_TOOL} build -t {image_name} {container_dir}")

        cx.run(f"{CONTAINER_TOOL} image save {image_name} > {root}/_output/containers/{image_name}.tar")

        # remove from the index
        cx.run(f"{CONTAINER_TOOL} rmi {image_name}")

@task
def list_built(cx, root=None):
    """List the built containers in dirs (not container tool memory)."""

    assert root is not None, \
        "Must provide a root directory with expected structure."

    images_dir = Path(root) / "_output/containers"

    image_names = []
    for image_fname in os.listdir(images_dir):
        print(image_fname)

        image_name = Path(image_fname).stem

        image_names.append(image_name)

    return image_names


@task
def load(cx):
    """Load the containers into container tool local memory."""

    assert root is not None, \
        "Must provide a root directory with expected structure."

    containers_list_built(cx)

    jig_name = Path(root).stem

    images_dir = Path(root) / "_output/containers"

    image_names = list_built(cx)

    for image_name in image_names:
        cx.run(f"{CONTAINER_TOOL} load < {images_dir}/{image_name}.tar {image_name}")

@task
def unload(cx):

    raise NotImplementedError

    assert root is not None, \
        "Must provide a root directory with expected structure."

    list_built(cx)

    jig_name = Path(root).stem

    images_dir = Path(root) / "_output/containers"

    image_names = list_built(cx)

    for image_name in image_names:
        cx.run(f"{CONTAINER_TOOL} rm {image_name}", warn=True)


