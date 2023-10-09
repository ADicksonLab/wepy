# Standard Library
import os
import os.path as osp
from pathlib import Path

# Third Party Library
from pytest_shutil.cmdline import chdir
from pytest_shutil.run import run, run_as_main

### Tests

EXAMPLE = "RandomWalk"


def test_dir(datadir_factory, printer):
    datadir = datadir_factory.mkdatadir(f"../_examples/{EXAMPLE}")

    assert (datadir / "README.org").is_file()
    assert (datadir / "input").is_dir()
    assert (datadir / "source").is_dir()


def test_runs(datadir_factory, printer):
    datadir = datadir_factory.mkdatadir(f"../_examples/{EXAMPLE}")

    with chdir(datadir):
        run(
            [
                "bash",
                "_tangle_source/run0.bash",
            ],
        )


def test_scripts(datadir_factory, printer):
    datadir = datadir_factory.mkdatadir(f"../_examples/{EXAMPLE}")

    with chdir(datadir):
        run(
            ["python", "source/rw_conventional.py", "1", "10", "10", "3"],
        )

        run(
            ["python", "source/rw_revo.py", "1", "10", "10", "3"],
        )

        run(
            ["python", "source/rw_wexplore.py", "1", "10", "10", "3"],
        )
