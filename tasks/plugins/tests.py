from invoke import task

from ..config import (
    REPORTS_DIR,
)

import sys
import os
import os.path as osp
from pathlib import Path

import pytest


# TODO: this should be done better
@task
def integration(cx, tag=None, node='node_minor'):
    """Run the integration tests.

    This is a large test suite and needs specific hardware resources
    in order to run all of them. For this reason there are different
    test objects which are tagged as different grades of nodes. The
    idea is that depending on the machine you are able to test on you
    will still be able to run some of the tests to test pure-python
    code paths or code paths that involve GPUs etc.

    The node types are:

    - minor :: no GPUs
    - dev :: at least 1 GPU
    - production :: more than 1 GPU

    You can use these as a 'mark' selection when running pytest or use
    it as the option for this command.

    """

    lines = [
        f"coverage run -m pytest ",
        f"-m 'not interactive' ",
        f"tests/test_integration",
    ]

    if node == 'minor':
        node = ''

    options = {
        "html" : (
            "--html=reports/pytest/{tag}/integration/report.html"
            if tag is not None
            else ""
        ),

        "node" : node,
    }

    if tag is None:
        cx.run('heerr',
               warn=True)
    else:
        cx.run(f"coverage run -m pytest  -m 'not interactive' tests/test_integration",
               warn=True)
