# Standard Library
import os
import os.path as osp
import pdb

# Third Party Library
import pytest
from click.testing import CliRunner

# First Party Library
from wepy.orchestration.cli import cli as wepy_cli
from wepy.orchestration.orchestrator import Orchestrator

lj_fixtures = [
    "lj_orchestrator_defaults_file",
    "lj_orch_file_orchestrated_run",
    "lj_orchestrator_defaults_file_other",
    "lj_orch_file_other_orchestrated_run",
]


@pytest.mark.interactive
def test_orch_workdir(lj_orchestrator_defaults_file):
    pdb.set_trace()


@pytest.mark.usefixtures(*lj_fixtures)
class TestCLI:
    def test_ls_runs(self, lj_orch_file_orchestrated_run):
        orch_path = lj_orch_file_orchestrated_run.orch_path

        runner = CliRunner()

        result = runner.invoke(wepy_cli, ["ls", "runs", orch_path])

        assert result.exit_code == 0

    def test_ls_snapshots(self, lj_orch_file_orchestrated_run):
        orch_path = lj_orch_file_orchestrated_run.orch_path

        runner = CliRunner()

        result = runner.invoke(wepy_cli, ["ls", "snapshots", orch_path])

        assert result.exit_code == 0

    def test_ls_configs(self, lj_orch_file_orchestrated_run):
        orch_path = lj_orch_file_orchestrated_run.orch_path

        runner = CliRunner()

        result = runner.invoke(wepy_cli, ["ls", "configs", orch_path])

        assert result.exit_code == 0

    def test_run_orch(self, function_tmp_path_factory, lj_orchestrator_defaults_file):
        workdir = str(function_tmp_path_factory.mktemp("test_run"))

        n_steps = str(100)
        n_seconds = str(5)

        orch_path = lj_orchestrator_defaults_file.orch_path
        start_hash = lj_orchestrator_defaults_file.get_default_snapshot_hash()

        runner = CliRunner()

        result = runner.invoke(
            wepy_cli,
            [
                "run",
                "orch",
                "--job-dir",
                workdir,
                orch_path,
                start_hash,
                n_seconds,
                n_steps,
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0

    def test_run_snapshot(
        self, function_tmp_path_factory, lj_snapshot, lj_configuration
    ):
        workdir = str(function_tmp_path_factory.mktemp("test_run"))

        # write the snapshot to the file system
        serial_snap = Orchestrator.serialize(lj_snapshot)
        snap_path = osp.join(workdir, "snapshot.snap.dill.pkl")
        with open(snap_path, "wb") as wf:
            wf.write(serial_snap)

        # write the configuration to the file system
        serial_config = Orchestrator.serialize(lj_configuration)
        config_path = osp.join(workdir, "config.config.dill.pkl")
        with open(config_path, "wb") as wf:
            wf.write(serial_config)

        n_steps = str(100)
        n_seconds = str(5)

        runner = CliRunner()

        result = runner.invoke(
            wepy_cli,
            [
                "run",
                "snapshot",
                "--job-dir",
                workdir,
                snap_path,
                config_path,
                n_seconds,
                n_steps,
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0

    def test_get_snapshot(
        self, function_tmp_path_factory, lj_orchestrator_defaults_file
    ):
        workdir = str(function_tmp_path_factory.mktemp("test_run"))

        orch_path = lj_orchestrator_defaults_file.orch_path
        start_hash = lj_orchestrator_defaults_file.get_default_snapshot_hash()

        os.chdir(workdir)

        runner = CliRunner()

        result = runner.invoke(
            wepy_cli,
            ["get", "snapshot", orch_path, start_hash],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert osp.exists("{}.snap.dill.pkl".format(start_hash))

    def test_get_config(self, function_tmp_path_factory, lj_orchestrator_defaults_file):
        workdir = str(function_tmp_path_factory.mktemp("test_run"))

        orch_path = lj_orchestrator_defaults_file.orch_path
        config_hash = lj_orchestrator_defaults_file.get_default_configuration_hash()

        os.chdir(workdir)

        runner = CliRunner()

        result = runner.invoke(
            wepy_cli,
            ["get", "config", orch_path, config_hash],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert osp.exists("{}.config.dill.pkl".format(config_hash))

    def test_get_run(self, function_tmp_path_factory, lj_orch_file_orchestrated_run):
        workdir = str(function_tmp_path_factory.mktemp("test_run"))

        orch_path = lj_orch_file_orchestrated_run.orch_path
        start_hash, end_hash = lj_orch_file_orchestrated_run.run_hashes()[0]

        os.chdir(workdir)

        runner = CliRunner()

        result = runner.invoke(
            wepy_cli,
            ["get", "run", orch_path, start_hash, end_hash],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert osp.exists("{}-{}.orch.sqlite".format(start_hash, end_hash))

    def test_reconcile(
        self,
        function_tmp_path_factory,
        lj_orch_file_orchestrated_run,
        lj_orch_file_other_orchestrated_run,
    ):
        savedir = function_tmp_path_factory.mktemp("reconciliation")
        h5_target = str(savedir / "reconciled.wepy.h5")
        orch_target = str(savedir / "reconciled.orch.sqlite")

        orch_path = lj_orch_file_orchestrated_run.orch_path
        other_orch_path = lj_orch_file_other_orchestrated_run.orch_path

        runner = CliRunner()

        result = runner.invoke(
            wepy_cli,
            [
                "reconcile",
                "orch",
                "--hdf5",
                h5_target,
                orch_target,
                orch_path,
                other_orch_path,
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0

    def test_copy_h5(self, lj_orch_file_orchestrated_run):
        assert False
