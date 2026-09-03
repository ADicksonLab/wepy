"""Tests for VariablePopulationCloneMergeDecision."""

import numpy as np

from wepy.walker import Walker
from wepy.resampling.decisions.variable_population import (
    VariablePopulationCloneMergeDecision,
)


def test_clone_changes_population_and_preserves_weight():
    decision = VariablePopulationCloneMergeDecision
    walkers = [Walker("a", 0.6), Walker("b", 0.4)]
    step = [
        decision.record(decision.ENUM.CLONE.value, target_idxs=(0, 1, 2)),
        decision.record(decision.ENUM.NOTHING.value, target_idxs=(3,)),
    ]

    out = decision.action(walkers, [step])

    assert len(out) == 4
    assert [walker.state for walker in out] == ["a", "a", "a", "b"]
    assert np.allclose([walker.weight for walker in out], [0.2, 0.2, 0.2, 0.4])
    assert np.isclose(sum(w.weight for w in out), 1.0)
    assert decision.parents(step) == [0, 0, 0, 1]


def test_merge_changes_population_and_preserves_weight():
    decision = VariablePopulationCloneMergeDecision
    walkers = [Walker("a", 0.2), Walker("b", 0.3), Walker("c", 0.5)]
    step = [
        decision.record(decision.ENUM.SQUASH.value, target_idxs=(0,)),
        decision.record(decision.ENUM.KEEP_MERGE.value, target_idxs=(0,)),
        decision.record(decision.ENUM.NOTHING.value, target_idxs=(1,)),
    ]

    out = decision.action(walkers, [step])

    assert len(out) == 2
    assert out[0].state == "b"
    assert np.isclose(out[0].weight, 0.5)
    assert out[1].state == "c"
    assert np.isclose(out[1].weight, 0.5)
    assert decision.parents(step) == [1, 2]


def test_multiple_steps_support_split_then_merge():
    decision = VariablePopulationCloneMergeDecision
    walkers = [Walker("a", 0.6), Walker("b", 0.4)]

    split_step = [
        decision.record(decision.ENUM.CLONE.value, target_idxs=(0, 1, 2)),
        decision.record(decision.ENUM.NOTHING.value, target_idxs=(3,)),
    ]
    merge_step = [
        decision.record(decision.ENUM.KEEP_MERGE.value, target_idxs=(0,)),
        decision.record(decision.ENUM.SQUASH.value, target_idxs=(0,)),
        decision.record(decision.ENUM.NOTHING.value, target_idxs=(1,)),
        decision.record(decision.ENUM.NOTHING.value, target_idxs=(2,)),
    ]

    out = decision.action(walkers, [split_step, merge_step])

    assert len(out) == 3
    assert np.isclose(sum(w.weight for w in out), 1.0)
