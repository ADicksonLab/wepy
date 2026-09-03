"""Tests for Huber-Kim / WESTPA-style binned WE resampling."""

import numpy as np

from wepy.walker import Walker
from wepy.resampling.decisions.variable_population import (
    VariablePopulationCloneMergeDecision,
)
from wepy.resampling.resamplers.huber_kim import (
    HuberKimResampler,
    RectilinearBinMapper,
)


def _pcoord_1d(walker):
    return [walker.state]


def _assign_1d(mapper, walkers):
    return mapper.assign(np.asarray([[walker.state] for walker in walkers]))


def test_split_to_target_counts_like_westpa():
    mapper = RectilinearBinMapper([[0.0, 1.0, 2.0]])
    resampler = HuberKimResampler(
        progress_coordinate=_pcoord_1d,
        bin_mapper=mapper,
        bin_target_counts=[4, 4],
        rng=17,
    )

    walkers = [Walker(0.5, 0.25), Walker(1.5, 0.75)]
    resampled, _resampling_data, _resampler_data = resampler.resample(walkers)

    assignments = _assign_1d(mapper, resampled)
    assert len(resampled) == 8

    for bin_idx, bin_weight in ((0, 0.25), (1, 0.75)):
        weights = [
            walker.weight
            for walker, assignment in zip(resampled, assignments)
            if assignment == bin_idx
        ]
        assert len(weights) == 4
        assert np.allclose(weights, [bin_weight / 4.0] * 4)

    assert np.isclose(sum(walker.weight for walker in resampled), 1.0)


def test_adjust_counts_switch_reproduces_westpa_modes():
    mapper = RectilinearBinMapper([[0.0, 1.0, 2.0]])
    walkers = [Walker(0.5, 0.125), Walker(0.5, 0.125)]

    original_mode = HuberKimResampler(
        _pcoord_1d,
        mapper,
        [5, 5],
        adjust_counts=False,
        do_thresholds=False,
        rng=21,
    )
    adjusted_mode = HuberKimResampler(
        _pcoord_1d,
        mapper,
        [5, 5],
        adjust_counts=True,
        do_thresholds=False,
        rng=21,
    )

    original_walkers, _, _ = original_mode.resample(walkers)
    adjusted_walkers, _, _ = adjusted_mode.resample(walkers)

    assert len(original_walkers) == 6
    assert len(adjusted_walkers) == 5
    assert np.isclose(sum(w.weight for w in original_walkers), 0.25)
    assert np.isclose(sum(w.weight for w in adjusted_walkers), 0.25)


def test_merge_survivor_is_weighted():
    mapper = RectilinearBinMapper([[0.0, 1.0]])
    resampler = HuberKimResampler(
        _pcoord_1d,
        mapper,
        [1],
        do_thresholds=False,
        rng=1931,
    )

    n_rounds = 2000
    light_survivals = 0

    for _ in range(n_rounds):
        walkers = [Walker(0.25, 1.0 / 3.0), Walker(0.75, 2.0 / 3.0)]
        resampled, _, _ = resampler.resample(walkers)
        assert len(resampled) == 1
        assert np.isclose(resampled[0].weight, 1.0)
        light_survivals += int(resampled[0].state == 0.25)

    fraction = light_survivals / n_rounds
    assert 0.29 < fraction < 0.38


def test_passthrough_emits_identity_records_for_lineage():
    mapper = RectilinearBinMapper([[0.0, 1.0, 2.0]])
    resampler = HuberKimResampler(
        _pcoord_1d,
        mapper,
        [2, 2],
        do_thresholds=False,
        rng=2,
    )

    walkers = [
        Walker(0.25, 0.25),
        Walker(0.75, 0.25),
        Walker(1.25, 0.25),
        Walker(1.75, 0.25),
    ]
    resampled, resampling_data, _ = resampler.resample(walkers)

    assert len(resampled) == len(walkers)
    assert len(resampling_data) == len(walkers)
    assert all(
        int(record["decision_id"][0])
        == VariablePopulationCloneMergeDecision.ENUM.NOTHING.value
        for record in resampling_data
    )


def test_rectilinear_mapper_2d_c_order_and_boundaries():
    mapper = RectilinearBinMapper(
        [
            [-np.inf, 0.0, 1.0, np.inf],
            [-np.inf, 10.0, 20.0, np.inf],
        ]
    )

    coords = np.asarray(
        [
            [-1.0, 5.0],   # (0, 0) -> flat 0
            [-1.0, 10.0],  # (0, 1) -> flat 1; left-inclusive at 10
            [0.0, 5.0],    # (1, 0) -> flat 3; left-inclusive at 0
            [2.0, 25.0],   # (2, 2) -> flat 8
        ]
    )

    assert mapper.nbins_per_dim == (3, 3)
    assert mapper.nbins == 9
    assert mapper.assign(coords).tolist() == [0, 1, 3, 8]
    assert mapper.bin_tuple(8) == (2, 2)


def test_two_dimensional_progress_coordinate_is_binned_directly():
    mapper = RectilinearBinMapper(
        [
            [-np.inf, 0.0, np.inf],
            [-np.inf, 0.0, np.inf],
        ]
    )

    def pcoord(walker):
        return walker.state

    # One walker in each quadrant, unequal weights.
    walkers = [
        Walker(np.asarray([-1.0, -1.0]), 0.10),
        Walker(np.asarray([-1.0, +1.0]), 0.20),
        Walker(np.asarray([+1.0, -1.0]), 0.30),
        Walker(np.asarray([+1.0, +1.0]), 0.40),
    ]

    resampler = HuberKimResampler(
        progress_coordinate=pcoord,
        bin_mapper=mapper,
        bin_target_counts=2,
        adjust_counts=True,
        do_thresholds=False,
        rng=7,
    )

    out, _, data = resampler.resample(walkers)
    assignments = mapper.assign(np.stack([walker.state for walker in out]))

    assert len(out) == 8
    assert [np.sum(assignments == i) for i in range(4)] == [2, 2, 2, 2]
    assert np.isclose(sum(w.weight for w in out), 1.0)
    assert [int(record["bin_idx"][0]) for record in data] == [0, 1, 2, 3]


def test_progress_coordinate_cache_avoids_reprojection_after_cloning():
    mapper = RectilinearBinMapper([[0.0, 1.0]])
    calls = {"n": 0}

    def expensive_pcoord(walker):
        calls["n"] += 1
        return [walker.state]

    walkers = [Walker(0.5, 1.0)]
    resampler = HuberKimResampler(
        expensive_pcoord,
        mapper,
        [8],
        adjust_counts=True,
        do_thresholds=False,
        cache_progress_coordinates=True,
        rng=11,
    )

    out, _, _ = resampler.resample(walkers)

    assert len(out) == 8
    # All clones reuse the exact same state object, so the expensive pcoord
    # should be evaluated only once during the resampling cycle.
    assert calls["n"] == 1


def test_nonfinite_progress_coordinate_is_rejected():
    mapper = RectilinearBinMapper([[-np.inf, 0.0, np.inf]])
    resampler = HuberKimResampler(
        lambda walker: [np.nan],
        mapper,
        2,
        do_thresholds=False,
        rng=1,
    )

    try:
        resampler.resample([Walker(0.0, 1.0)])
    except Exception as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("Expected a non-finite progress-coordinate error")
