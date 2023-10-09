# Standard Library
import os
import os.path as osp
from pathlib import Path

# Third Party Library
from pytest_shutil.cmdline import chdir
from pytest_shutil.run import run, run_as_main

### Tests

EXAMPLE = "Orchestrator"


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
            ["python", "source/make_orchestrator.py", "1", "10", "10", "3"],
        )
