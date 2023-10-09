# Standard Library
import os
import os.path as osp
from pathlib import Path

# Third Party Library
import pytest
from pytest_shutil.cmdline import chdir
from pytest_shutil.run import run, run_as_main

### Tests


def test_dir(datadir_factory, printer):
    example = "Lennard_Jones_Pair"

    datadir = datadir_factory.mkdatadir(f"../_examples/{example}")

    assert (datadir / "README.org").is_file()
    assert (datadir / "source").is_dir()


def test_trivial_run(datadir_factory, printer):
    example = "Lennard_Jones_Pair"

    datadir = datadir_factory.mkdatadir(f"../_examples/{example}")

    with chdir(datadir):
        run(
            [
                "python",
                "source/trivial_run.py",
            ],
        )


def test_sim_maker_run(datadir_factory, printer):
    example = "Lennard_Jones_Pair"

    datadir = datadir_factory.mkdatadir(f"../_examples/{example}")

    with chdir(datadir):
        run(
            [
                "python",
                "source/sim_maker_run.py",
                "2",
                "100",
                "10",
                "1",
                "Reference",
                "NoResampler",
            ],
        )

        run(
            [
                "python",
                "source/sim_maker_run.py",
                "2",
                "100",
                "10",
                "1",
                "Reference",
                "WExploreResampler",
            ],
        )
        run(
            [
                "python",
                "source/sim_maker_run.py",
                "2",
                "100",
                "10",
                "1",
                "Reference",
                "REVOResampler",
            ],
        )
        run(
            [
                "python",
                "source/sim_maker_run.py",
                "2",
                "100",
                "10",
                "1",
                "CPU",
                "NoResampler",
            ],
        )
        run(
            [
                "python",
                "source/sim_maker_run.py",
                "2",
                "100",
                "10",
                "1",
                "CPU",
                "WExploreResampler",
            ],
        )
        run(
            [
                "python",
                "source/sim_maker_run.py",
                "2",
                "100",
                "10",
                "1",
                "CPU",
                "REVOResampler",
            ],
        )


def test_we_analysis(datadir_factory, printer):
    example = "Lennard_Jones_Pair"

    datadir = datadir_factory.mkdatadir(f"../_examples/{example}")

    printer("Testing from inside test_we_analysis")

    with chdir(datadir):
        printer(f"Datadir: {datadir}")
        printer(f"Current dir: {os.getcwd()}")

        out = run(["python", "source/we.py", "10", "100", "10"])

        printer(out)

        assert (datadir / "_output/we/results.wepy.h5").is_file()
        assert (datadir / "_output/we/wepy.dash.org").is_file()

        out = run(
            [
                "python",
                "source/compute_distance_observable.py",
            ]
        )

        printer(out)

        out = run(
            [
                "python",
                "source/state_network.py",
            ]
        )

        printer(out)

        assert (datadir / "_output/state.dcd").is_file()
        assert (datadir / "_output/random_macrostates.csn.gexf").is_file()

        ### Tangled sources

        out = run(
            [
                "python",
                "_tangle_source/inspect_observable.py",
            ]
        )

        printer(out)

        out = run(
            [
                "bash",
                "./_tangle_source/run0.bash",
            ]
        )

        printer(out)

        out = run(
            [
                "bash",
                "./_tangle_source/run1.bash",
            ]
        )
