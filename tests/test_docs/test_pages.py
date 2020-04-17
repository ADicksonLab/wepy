"""Test the main documentation pages."""

import os
import os.path as osp
from pathlib import Path

import delegator

# the helper modules for testing
from myutils import (
    cd,
)


def test_dir_structure(datadir_factory):

    datadir = Path(datadir_factory.mkdatadir('_tangled_docs'))

    assert (datadir / "README").is_dir()
    assert (datadir / "info").is_dir()

    assert (datadir / 'README/README.org').is_file()
    assert (datadir / 'info/README/README.org').is_file()
