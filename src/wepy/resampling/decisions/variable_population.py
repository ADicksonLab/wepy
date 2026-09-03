"""Variable-population clone/merge decisions for wepy.

This extends MultiCloneMergeDecision without changing the decision vocabulary or
record schema. The difference is that output population size is inferred from
target indices, allowing binned WE resamplers to change the total walker count.
"""

from collections.abc import Mapping

import numpy as np

from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from wepy.walker import keep_merge, split


class VariablePopulationCloneMergeDecision(MultiCloneMergeDecision):
    """Clone/merge decision that supports a changing walker population.

    The decision IDs and fields are identical to ``MultiCloneMergeDecision``:
    NOTHING, CLONE, SQUASH, and KEEP_MERGE with ``target_idxs``.  The only
    behavioral difference is that the output size is inferred from the target
    indices rather than being fixed to ``len(walkers)``.
    """

    @staticmethod
    def _scalar(value):
        arr = np.asarray(value)
        if arr.size != 1:
            raise ValueError("decision_id must contain exactly one value")
        return int(arr.reshape(-1)[0])

    @staticmethod
    def _targets(value):
        arr = np.asarray(value, dtype=int).reshape(-1)
        return tuple(int(x) for x in arr)

    @classmethod
    def _unpack_record(cls, record):
        if isinstance(record, Mapping):
            decision_id = cls._scalar(record["decision_id"])
            targets = cls._targets(record["target_idxs"])
        else:
            decision_id = cls._scalar(record[0])
            targets = cls._targets(record[1])
        return decision_id, targets

    @classmethod
    def _output_size(cls, step_records):
        targets = []
        for record in step_records:
            _decision_id, rec_targets = cls._unpack_record(record)
            targets.extend(rec_targets)

        if not targets:
            raise ValueError("A resampling step contains no target indices")
        if min(targets) < 0:
            raise ValueError("Target indices must be non-negative")

        n_out = max(targets) + 1
        if set(targets) != set(range(n_out)):
            # Repeated targets are normal for merge groups, but every output
            # slot must be addressed by at least one record.
            missing = sorted(set(range(n_out)).difference(targets))
            if missing:
                raise ValueError(
                    "Resampling target indices do not cover all output slots; "
                    "missing {}".format(missing)
                )
        return n_out

    @classmethod
    def action(cls, walkers, decisions):
        current_walkers = list(walkers)

        for step_records in decisions:
            if len(step_records) != len(current_walkers):
                raise ValueError(
                    "Decision step has {} records for {} walkers".format(
                        len(step_records), len(current_walkers)
                    )
                )

            n_out = cls._output_size(step_records)
            output_walkers = [None] * n_out

            keep_by_target = {}
            squash_by_target = {}

            for walker_idx, record in enumerate(step_records):
                decision_id, targets = cls._unpack_record(record)

                if decision_id == cls.ENUM.NOTHING.value:
                    if len(targets) != 1:
                        raise ValueError("NOTHING requires exactly one target")
                    target = targets[0]
                    if output_walkers[target] is not None:
                        raise ValueError("Multiple walkers assigned to slot {}".format(target))
                    output_walkers[target] = current_walkers[walker_idx]

                elif decision_id == cls.ENUM.CLONE.value:
                    if len(targets) < 1:
                        raise ValueError("CLONE requires at least one target")
                    clones = split(current_walkers[walker_idx], number=len(targets))
                    for clone, target in zip(clones, targets):
                        if output_walkers[target] is not None:
                            raise ValueError("Multiple walkers assigned to slot {}".format(target))
                        output_walkers[target] = clone

                elif decision_id == cls.ENUM.SQUASH.value:
                    if len(targets) != 1:
                        raise ValueError("SQUASH requires exactly one merge target")
                    squash_by_target.setdefault(targets[0], []).append(walker_idx)

                elif decision_id == cls.ENUM.KEEP_MERGE.value:
                    if len(targets) != 1:
                        raise ValueError("KEEP_MERGE requires exactly one target")
                    target = targets[0]
                    if target in keep_by_target:
                        raise ValueError("Multiple KEEP_MERGE walkers target slot {}".format(target))
                    keep_by_target[target] = walker_idx

                else:
                    raise ValueError("Unknown clone/merge decision id {}".format(decision_id))

            for target, squash_idxs in squash_by_target.items():
                if target not in keep_by_target:
                    raise ValueError(
                        "SQUASH records target slot {} without a KEEP_MERGE record".format(target)
                    )

                keep_idx = keep_by_target[target]
                merge_group = [current_walkers[keep_idx]] + [
                    current_walkers[idx] for idx in squash_idxs
                ]
                merged_walker = keep_merge(merge_group, 0)

                if output_walkers[target] is not None:
                    raise ValueError("Multiple walkers assigned to slot {}".format(target))
                output_walkers[target] = merged_walker

            # A KEEP_MERGE without donors is harmless and is treated as a
            # state-preserving one-member merge.
            for target, keep_idx in keep_by_target.items():
                if target not in squash_by_target:
                    if output_walkers[target] is not None:
                        raise ValueError("Multiple walkers assigned to slot {}".format(target))
                    output_walkers[target] = current_walkers[keep_idx]

            if any(walker is None for walker in output_walkers):
                missing = [i for i, walker in enumerate(output_walkers) if walker is None]
                raise ValueError("Some walkers were not created; empty slots {}".format(missing))

            current_walkers = output_walkers

        return current_walkers

    @classmethod
    def parents(cls, step):
        """Return the realized parent index for every child in a step.

        For a merge, the stochastic KEEP_MERGE walker is the parent of the
        merged child; SQUASH lineages terminate at that step.
        """

        n_out = cls._output_size(step)
        parents = [None] * n_out

        for parent_idx, record in enumerate(step):
            decision_id, targets = cls._unpack_record(record)
            if decision_id in cls.ANCESTOR_DECISION_IDS:
                for child_idx in targets:
                    if parents[child_idx] is not None:
                        raise ValueError(
                            "Multiple ancestor walkers assigned to child {}".format(child_idx)
                        )
                    parents[child_idx] = parent_idx

        if any(parent is None for parent in parents):
            missing = [i for i, parent in enumerate(parents) if parent is None]
            raise ValueError("No ancestor assigned to child slots {}".format(missing))

        return parents

