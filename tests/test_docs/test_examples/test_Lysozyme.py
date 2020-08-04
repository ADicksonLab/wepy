import os
import os.path as osp
from pathlib import Path

from pytest_shutil.cmdline import chdir
from pytest_shutil.run import run, run_as_main

### Tests

EXAMPLE = "Lysozyme"


def test_dir(datadir_factory, printer):

    datadir = datadir_factory.mkdatadir(f'../_examples/{EXAMPLE}')

    assert (datadir / "README.org").is_file()
    assert (datadir / "input").is_dir()
    assert (datadir / "source").is_dir()


def test_runs(datadir_factory, printer):

    datadir = datadir_factory.mkdatadir(f'../_examples/{EXAMPLE}')

    with chdir(datadir):
        run(['bash',
            '_tangle_source/run0.bash',
            ],
        )


def test_we(datadir_factory, printer):

    datadir = datadir_factory.mkdatadir(f'../_examples/{EXAMPLE}')

    with chdir(datadir):

        print("CPU-NoResampler")
        run(['python',
            'source/we.py',
             '2', '2', '10', '1', 'CPU', 'NoResampler'
            ],
        )

        print("CPU-REVOResampler")
        run(['python',
            'source/we.py',
             '2', '2', '10', '1', 'CPU', 'REVOResampler'
            ],
        )

        print("CPU_WExploreResampler")
        run(['python',
            'source/we.py',
             '2', '2', '10', '1', 'CPU', 'WExploreResampler'
            ],
        )

