# Standard Library
import os
import os.path as osp
from pathlib import Path

# Third Party Library
from pytest_shutil.cmdline import chdir
from pytest_shutil.run import run, run_as_main

### Tests


def test_tutorial(datadir_factory, printer):
    tutorial = "data_analysis"

    datadir = datadir_factory.mkdatadir(f"../_tutorials/{tutorial}")

    assert (datadir / "README.ipynb").is_file()
    assert (datadir / "input").is_dir()

    with chdir(datadir):
        run(
            [
                "python",
                "_tangle_source/README.py",
            ],
        )

        assert (datadir / "_output/results_run1.wepy.h5").exists()
        assert (datadir / "_output/results_run2.wepy.h5").exists()
        assert (datadir / "_output/results_run3.wepy.h5").exists()

        assert (datadir / "_output/lj-pair.pdb").exists()
        assert (datadir / "_output/lj-pair_walker_lineage").exists()
