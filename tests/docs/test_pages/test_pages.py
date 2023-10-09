"""Test the main documentation pages."""

# Standard Library
import os
import os.path as osp
from pathlib import Path

# Third Party Library
from pytest_shutil.cmdline import chdir
from pytest_shutil.run import run, run_as_main


def test_dir_structure(datadir_factory):
    datadir = Path(datadir_factory.mkdatadir("../_tangled_docs"))

    assert (datadir / "README").is_dir()
    assert (datadir / "info").is_dir()

    assert (datadir / "README/README.org").is_file()
    assert (datadir / "info/README/README.org").is_file()


def test_readme(datadir_factory):
    datadir = Path(datadir_factory.mkdatadir("../_tangled_docs/README"))

    with chdir(datadir):
        out = run(
            [
                "bash",
                "check_installation.bash",
            ],
        )


def test_installation(datadir_factory):
    datadir = Path(datadir_factory.mkdatadir("../_tangled_docs/info/installation"))

    with chdir(datadir):
        out = run(
            [
                "bash",
                "check_installation.bash",
            ],
        )


def test_quick_start(datadir_factory):
    datadir = Path(datadir_factory.mkdatadir("../_tangled_docs/info/quick_start"))

    with chdir(datadir):
        out = run(
            [
                "bash",
                "test_drive.bash",
            ],
        )

        out = run(
            [
                "python",
                "noresampler_example.py",
            ],
        )

        out = run(
            [
                "bash",
                "noresampler_example.bash",
            ],
        )


def test_introduction(datadir_factory):
    # STUB
    # datadir = Path(datadir_factory.mkdatadir('../_tangled_docs/info/introduction'))
    # with chdir(datadir):
    #     pass

    pass


def test_users_guide(datadir_factory):
    # STUB
    # datadir = Path(datadir_factory.mkdatadir('../_tangled_docs/info/users_guide'))
    # with chdir(datadir):
    #     pass

    pass


def test_howtos(datadir_factory):
    # STUB
    # datadir = Path(datadir_factory.mkdatadir('../_tangled_docs/info/howtos'))
    # with chdir(datadir):
    #     pass

    pass


def test_reference(datadir_factory):
    datadir = Path(datadir_factory.mkdatadir("../_tangled_docs/info/reference"))
    with chdir(datadir):
        out = run(
            [
                "python",
                "decision_fields_0.py",
            ],
        )

        out = run(
            [
                "python",
                "record_fields_0.py",
            ],
        )


def test_troubleshooting(datadir_factory):
    # STUB
    # datadir = Path(datadir_factory.mkdatadir('../_tangled_docs/info/troubleshooting'))
    # with chdir(datadir):
    #     pass

    pass
