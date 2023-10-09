# Standard Library
import os
import os.path as osp
from pathlib import Path

# Third Party Library
from pytest_check import check
from pytest_shutil.cmdline import chdir
from pytest_shutil.run import run, run_as_main

### Tests


def test_tutorial(datadir_factory, printer):
    tutorial = "extended_test_drive"

    datadir = datadir_factory.mkdatadir(f"../_tutorials/{tutorial}")

    assert (datadir / "README.org").is_file()
    assert (datadir / "input").is_dir()


def test_run0(datadir_factory, printer):
    tutorial = "extended_test_drive"

    datadir = datadir_factory.mkdatadir(f"../_tutorials/{tutorial}")

    with chdir(datadir):
        # check that the help message runs

        run(
            [
                "bash",
                "_tangle_source/run-help.bash",
            ],
        )

        # default Run with WExplore

        run0_out = run(
            [
                "bash",
                "_tangle_source/run0.bash",
            ],
        )

        with check:
            assert (
                datadir / "_tangle_source/expected_run0_ls.txt"
            ).read_text() == run0_out

        assert (datadir / "_output/run0/root.wepy.h5").exists()
        assert (datadir / "_output/run0/root.dash.org").exists()
        assert (datadir / "_output/run0/root.init_top.pdb").exists()
        assert (datadir / "_output/run0/root.walkers.dcd").exists()

        # REVO run

        run(
            [
                "bash",
                "_tangle_source/revo_run.bash",
            ],
        )

        assert (datadir / "_output/revo_run/root.wepy.h5").exists()
        assert (datadir / "_output/revo_run/root.dash.org").exists()
        assert (datadir / "_output/revo_run/root.init_top.pdb").exists()
        assert (datadir / "_output/revo_run/root.walkers.dcd").exists()

        # No run

        run(
            [
                "bash",
                "_tangle_source/no_run.bash",
            ],
        )

        assert (datadir / "_output/no_run/root.wepy.h5").exists()
        assert (datadir / "_output/no_run/root.dash.org").exists()
        assert (datadir / "_output/no_run/root.init_top.pdb").exists()
        assert (datadir / "_output/no_run/root.walkers.dcd").exists()

        ## analysis

        # part 0
        analysis0_out = run(
            [
                "python",
                "_tangle_source/analysis0.py",
            ],
        )

        assert (datadir / "_output/run0/traj0.dcd").exists()
        assert (datadir / "_output/run0/last_cycle.dcd").exists()

        with check:
            assert (
                datadir / "_tangle_source/test_analysis_0.txt"
            ).read_text() == analysis0_out.strip()
