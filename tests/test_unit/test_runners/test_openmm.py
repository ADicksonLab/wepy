import pytest

from wepy.walker import Walker
from wepy.runners.openmm import (
    OpenMMRunner,
    gen_walker_state,
)

import simtk.openmm as omm

from openmmtools.testsystems import LennardJonesPair

def test_runtime_platform():
    """Test that the platform can be changed at the time of the call to
    run_segment."""

    test_sys = LennardJonesPair()
    integrator = omm.LangevinIntegrator(
        300.,
        1,
        0.002,
    )

    positions = test_sys.positions.value_in_unit(test_sys.positions.unit)

    init_state = gen_walker_state(
        positions,
        test_sys.system,
        integrator)

    walker = Walker(
        init_state,
        1.0,
    )

    runner = OpenMMRunner(
        test_sys.system,
        test_sys.topology,
        integrator,
        platform="CPU",
        platform_kwargs={'Threads' : '2'},
    )

    plat_name, plat_kwargs = runner._resolve_platform(
        'CPU',
        None,
    )

    assert plat_name == 'CPU'
    assert plat_kwargs == None

    plat_name, plat_kwargs = runner._resolve_platform(
        'Reference',
        None,
    )

    assert plat_name == 'Reference'
    assert plat_kwargs == None

    plat_name, plat_kwargs = runner._resolve_platform(
        None,
        None,
    )

    assert plat_name == 'CPU'
    assert plat_kwargs == {'Threads' : '2'}

    plat_name, plat_kwargs = runner._resolve_platform(
        Ellipsis,
        None,
    )

    assert plat_name == None
    assert plat_kwargs == None

    # run checks
    _ = runner.run_segment(
        walker,
        2,
    )

    _ = runner.run_segment(
        walker,
        2,
        platform='Reference',
    )

    _ = runner.run_segment(
        walker,
        2,
        platform=Ellipsis,
    )
