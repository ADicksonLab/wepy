import pytest

from wepy.orchestration.orchestrator import Orchestrator
from wepy.orchestration.configuration import Configuration
from wepy.orchestration.snapshot import WepySimApparatus, SimSnapshot
from wepy.walker import Walker
from wepy.resampling.resamplers.wexplore import WExploreResampler
from wepy.runners.openmm import OpenMMRunner

def test_apparatus_configuration(datadir_factory, mocker):

    config = Configuration()

    assert config.apparatus_opts == {}

    reparam_config = config.reparametrize(
        apparatus_opts={
            'runner' : {
                'platform' : 'CPU',
            },
        })

    assert reparam_config.apparatus_opts == {
            'runner' : {
                'platform' : 'CPU',
            },
        }

    ## test that we can change the apparatus parameters in the sim_manager

    system_mock = mocker.Mock()
    topology_mock = mocker.Mock()
    integrator_mock = mocker.Mock()

    runner = OpenMMRunner(
        system_mock,
        topology_mock,
        integrator_mock,
    )

    resampler_mock = mocker.MagicMock() # mocker.patch('WExploreResampler')

    apparatus = WepySimApparatus(
        runner,
        resampler=resampler_mock,
        boundary_conditions=None,
    )
    apparatus._filters = (
        runner,
        None,
        resampler_mock,
    )

    state_mock = mocker.MagicMock() # mocker.patch('wepy.walker.WalkerState', autospec=True)

    walkers = [Walker(state_mock, 0.1) for i in range(1)]

    snapshot = SimSnapshot(
        walkers,
        apparatus,
    )
    snapshot._walkers = walkers
    snapshot._apparatus = apparatus


    datadir = datadir_factory.mkdatadir()

    orch = Orchestrator(
        orch_path=str(datadir / "test.orch.sqlite3")
    )


    sim_manager = orch.gen_sim_manager(
        snapshot,
        reparam_config,
    )

    sim_manager.init()

    # sim_mock = mocker.patch('wepy.runners.openmm.omma.Simulation')
    # platform_mock = mocker.patch('wepy.runners.openmm.omm.Platform')

    # _ = sim_manager.run_cycle(
    #     walkers,
    #     2,
    #     0,
    #     runner_opts={
    #         'platform' : 'CPU',
    #     }
    # )

    # platform_mock.getPlatformByName.assert_called_with('CPU')

    sim_mock = mocker.patch('wepy.runners.openmm.omma.Simulation')
    platform_mock = mocker.patch('wepy.runners.openmm.omm.Platform')
    platform_mock.getPlatformByName.\
        return_value.getPropertyNames.\
        return_value = ('Threads',)

    _ = sim_manager.run_cycle(
        walkers,
        2,
        0,
        runner_opts={
            'platform' : 'CPU',
            'platform_kwargs' : {'Threads' : '3'},
        }
    )

    platform_mock.getPlatformByName.assert_called_with('CPU')

    platform_mock.getPlatformByName.\
        return_value.getPropertyNames.\
        assert_called()

    platform_mock.getPlatformByName.\
        return_value.setPropertyDefaultValue.\
        assert_called_with(
        'Threads',
        '3'
    )
