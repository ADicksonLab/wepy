import os
import os.path as osp
from pathlib import Path

import delegator

# the helper modules for testing
from myutils import (
    cd,
)


## write one test per example

# Lennard_Jones_Pair

def test_lennard_jones_pair(datadir_factory, printer):

    example = "Lennard_Jones_Pair"

    datadir = datadir_factory.mkdatadir(f'../_examples/{example}')

    assert (datadir / "README.org").is_file()
    assert (datadir / "input").is_dir()
    assert (datadir / "source").is_dir()


    ##

    assert (datadir / "source/trivial_run.py").is_file()

    with cd(datadir):
        assert Path(os.getcwd()) == datadir
        proc = delegator.run([
            "python",
            "source/trivial_run.py",
        ])

    if not proc.ok:
        printer("Error for hello.py:")
        printer(proc.err)

    assert proc.ok
