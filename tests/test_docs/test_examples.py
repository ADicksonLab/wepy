import os
import os.path as osp
from pathlib import Path

import delegator

# the helper modules for testing
from myutils import (
    cd,
)


## write one test per example

# library

def test_library(datadir_factory, printer):

    datadir = datadir_factory.mkdatadir('_examples/library')

    assert (datadir / "README.org").is_file()
    assert (datadir / "input").is_dir()
    assert (datadir / "source").is_dir()


    ## hello.py script
    assert (datadir / "source/hello.py").is_file()

    with cd(datadir):
        assert Path(os.getcwd()) == datadir
        proc = delegator.run("python source/hello.py")

    if not proc.ok:
        printer("Error for hello.py:")
        printer(proc.err)

    assert proc.ok


    ## failure.py script
    assert (datadir / "source/fail_script.py").is_file()

    with cd(datadir):
        assert Path(os.getcwd()) == datadir
        proc = delegator.run("python source/fail_script.py")

    if not proc.ok:
        printer("Error for fail_script.py:")
        printer(proc.err)

    assert not proc.ok
