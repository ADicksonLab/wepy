"""Huber-Kim / WESTPA-style binned weighted-ensemble resampling for wepy.

Implements the classic bin-based WE algorithm and the default single-subgroup
logic of WESTPA ``WEDriver``. Progress coordinates may be one- or
multidimensional; a 2-D tICA coordinate is therefore handled directly without
collapsing it to a radial distance.

The resampler supports WESTPA-style variable total walker population.
``adjust_counts=True`` enforces the requested target count in every occupied
bin, while ``adjust_counts=False`` leaves counts determined by the original
weight-threshold split/merge logic.

Resampling never propagates dynamics and never alters molecular coordinates.
For expensive progress coordinates (e.g. tICA transforms), values are cached by
walker-state identity for the duration of each resampling cycle because clones
and merge survivors reuse existing states.
"""

from collections.abc import Mapping
import math

import numpy as np

from wepy.resampling.decisions.variable_population import (
    VariablePopulationCloneMergeDecision,
)
from wepy.resampling.resamplers.resampler import Resampler, ResamplerError


class RectilinearBinMapper:
    """Small WESTPA-like rectilinear bin mapper for one or more coordinates.

    Parameters
    ----------
    boundaries : sequence of sequence of float
        One monotonically increasing boundary vector per progress-coordinate
        dimension.  ``[[0.0, 1.0, 2.0]]`` defines two one-dimensional bins.

    Notes
    -----
    For production WESTPA parity you can instead pass any mapper exposing an
    ``assign(coords)`` method, including a WESTPA-compatible mapper.  This
    helper exists so the resampler has no WESTPA runtime dependency.
    """

    def __init__(self, boundaries):
        """Validate and store rectilinear boundary vectors.

        Parameters
        ----------
        boundaries : sequence of sequence of float
            One strictly increasing edge vector per coordinate dimension.

        Notes
        -----
        A dimension with ``m`` edges contains ``m - 1`` intervals. The total
        number of bins is the product of the interval counts in all
        dimensions.
        """
        self.boundaries = tuple(np.asarray(edges, dtype=np.float32) for edges in boundaries)
        if len(self.boundaries) == 0:
            raise ValueError("At least one progress-coordinate dimension is required")

        for edges in self.boundaries:
            if edges.ndim != 1 or len(edges) < 2:
                raise ValueError("Each boundary vector must be one-dimensional with >=2 edges")
            if not np.all(np.diff(edges) > 0):
                raise ValueError("Bin boundaries must be strictly increasing")

        self.nbins_per_dim = tuple(len(edges) - 1 for edges in self.boundaries)
        self.nbins = int(np.prod(self.nbins_per_dim))

    @property
    def ndim(self):
        """int: Number of progress-coordinate dimensions."""
        return len(self.boundaries)

    def bin_tuple(self, bin_idx):
        """Convert a flattened bin ID to per-dimension integer indices.

        This is primarily a diagnostic inverse of the C-order flattening used
        by :meth:`assign`.
        """
        return tuple(int(x) for x in np.unravel_index(int(bin_idx), self.nbins_per_dim))

    def assign(self, coords):
        """Assign one or more progress coordinates to flattened bin IDs.

        Parameters
        ----------
        coords : array-like
            Coordinates shaped ``(n_walkers, ndim)``. A single coordinate or
            a one-dimensional coordinate vector is normalized when unambiguous.

        Returns
        -------
        numpy.ndarray
            One integer bin ID per input coordinate.

        Notes
        -----
        Intervals are left-inclusive and right-exclusive: ``[left, right)``.
        Non-finite coordinates are rejected even though boundary vectors may
        contain ``-inf`` and ``inf``.
        """
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim == 1:
            if len(self.boundaries) == 1:
                coords = coords.reshape(-1, 1)
            else:
                coords = coords.reshape(1, -1)

        if coords.ndim != 2 or coords.shape[1] != len(self.boundaries):
            raise ValueError(
                "coords must have shape (n_walkers, {}), got {}".format(
                    len(self.boundaries), coords.shape
                )
            )
        if np.any(~np.isfinite(coords)):
            raise ValueError(
                "Progress coordinates must be finite; bin boundaries may use +/-inf"
            )

        per_dim = []
        for dim, edges in enumerate(self.boundaries):
            values = coords[:, dim]
            # WESTPA semantics: [edge_i, edge_{i+1}); the final right
            # boundary is outside the bin space.
            idxs = np.searchsorted(edges, values, side="right") - 1
            outside = (values < edges[0]) | (values >= edges[-1])
            if np.any(outside):
                bad = values[outside]
                raise ValueError("Progress coordinate(s) outside bin boundaries: {}".format(bad))
            per_dim.append(idxs)

        return np.ravel_multi_index(tuple(per_dim), self.nbins_per_dim)


class HuberKimResampler(Resampler):
    """Binned Huber-Kim weighted-ensemble resampler with WESTPA semantics.

    Parameters
    ----------
    progress_coordinate : callable
        ``progress_coordinate(walker) -> scalar or 1-D array``.  It should be
        a function of the walker state, not of walker weight.
    bin_mapper : object or callable
        Preferred form is an object with ``assign(coords)`` returning integer
        bin IDs, as in WESTPA.  A callable mapping one progress coordinate to
        one integer bin ID is also accepted.
    bin_target_counts : int, sequence, mapping, or callable
        Desired walker count for each bin.  A scalar applies to every bin;
        a sequence is indexed by integer bin ID; a mapping is keyed by bin ID;
        a callable receives ``bin_idx``.
    adjust_counts : bool, default True
        If True, force each occupied bin to its exact target count after the
        weight-based split/merge pass.  This matches current WESTPA defaults.
        If False, populations may fluctuate as in the original Huber-Kim
        implementation described by WESTPA documentation.
    weight_split_threshold : float, default 2.0
        Split a walker if ``weight > threshold * ideal_weight``.
    weight_merge_cutoff : float, default 1.0
        Merge the lowest-weight prefix when its cumulative weight is at most
        ``cutoff * ideal_weight`` and at least two walkers are selected.
    do_thresholds : bool, default True
        Apply WESTPA's absolute post-resampling weight limits.
    largest_allowed_weight : float, default 1.0
        Absolute upper weight bound used when ``do_thresholds`` is True.
    smallest_allowed_weight : float, default 1e-310
        Absolute lower weight bound used when ``do_thresholds`` is True.
    rng : None, int, or numpy.random.Generator
        Random-number generator for stochastic merge survivor selection.  An
        integer creates ``Generator(MT19937(seed))``, matching WESTPA's RNG
        family.  ``None`` creates a fresh MT19937 generator.
    cache_progress_coordinates : bool, default True
        Cache progress coordinates by walker-state identity for the duration of
        one resampling cycle. This is safe because WE clone/merge operations
        preserve state coordinates and is useful for expensive tICA projectors.
    min_num_walkers, max_num_walkers : int, None, or Ellipsis
        Standard wepy resampler population constraints.  Defaults are None so
        WESTPA-style variable total populations are allowed.
    """

    DECISION = VariablePopulationCloneMergeDecision

    RESAMPLING_FIELDS = DECISION.FIELDS + Resampler.CYCLE_FIELDS
    RESAMPLING_SHAPES = DECISION.SHAPES + Resampler.CYCLE_SHAPES
    RESAMPLING_DTYPES = DECISION.DTYPES + Resampler.CYCLE_DTYPES
    RESAMPLING_RECORD_FIELDS = DECISION.RECORD_FIELDS + Resampler.CYCLE_RECORD_FIELDS

    RESAMPLER_FIELDS = (
        "bin_idx",
        "target_count",
        "pre_count",
        "post_count",
        "bin_weight",
        "ideal_weight",
    )
    RESAMPLER_SHAPES = ((1,),) * len(RESAMPLER_FIELDS)
    RESAMPLER_DTYPES = (int, int, int, int, float, float)
    RESAMPLER_RECORD_FIELDS = RESAMPLER_FIELDS

    def __init__(
        self,
        progress_coordinate,
        bin_mapper,
        bin_target_counts,
        adjust_counts=True,
        weight_split_threshold=2.0,
        weight_merge_cutoff=1.0,
        do_thresholds=True,
        largest_allowed_weight=1.0,
        smallest_allowed_weight=1e-310,
        rng=None,
        cache_progress_coordinates=True,
        min_num_walkers=None,
        max_num_walkers=None,
        debug_mode=False,
        **kwargs
    ):
        """Configure a WESTPA-like, bin-based Huber--Kim resampler.

        The constructor validates the coordinate/bin interfaces, relative and
        absolute weight thresholds, population constraints, and stochastic
        merge-survivor generator. It does not inspect walker states; that work
        begins in :meth:`resample`.

        See the class docstring for complete parameter descriptions.
        """
        super().__init__(
            min_num_walkers=min_num_walkers,
            max_num_walkers=max_num_walkers,
            debug_mode=debug_mode,
            **kwargs
        )

        if not callable(progress_coordinate):
            raise TypeError("progress_coordinate must be callable")
        if not (callable(bin_mapper) or hasattr(bin_mapper, "assign")):
            raise TypeError("bin_mapper must be callable or expose assign(coords)")
        if weight_split_threshold <= 0:
            raise ValueError("weight_split_threshold must be positive")
        if weight_merge_cutoff <= 0:
            raise ValueError("weight_merge_cutoff must be positive")
        if largest_allowed_weight <= 0:
            raise ValueError("largest_allowed_weight must be positive")
        if smallest_allowed_weight <= 0:
            raise ValueError("smallest_allowed_weight must be positive")
        if smallest_allowed_weight > largest_allowed_weight:
            raise ValueError("smallest_allowed_weight cannot exceed largest_allowed_weight")

        self.progress_coordinate = progress_coordinate
        self.bin_mapper = bin_mapper
        self.bin_target_counts = bin_target_counts
        self.adjust_counts = bool(adjust_counts)
        self.weight_split_threshold = float(weight_split_threshold)
        self.weight_merge_cutoff = float(weight_merge_cutoff)
        self.do_thresholds = bool(do_thresholds)
        self.largest_allowed_weight = float(largest_allowed_weight)
        self.smallest_allowed_weight = float(smallest_allowed_weight)
        self.cache_progress_coordinates = bool(cache_progress_coordinates)

        if isinstance(rng, np.random.Generator):
            self.rng = rng
        elif rng is None:
            self.rng = np.random.Generator(np.random.MT19937())
        elif isinstance(rng, (int, np.integer)):
            self.rng = np.random.Generator(np.random.MT19937(int(rng)))
        else:
            raise TypeError("rng must be None, an integer seed, or numpy.random.Generator")

        self._step_idx = 0
        self._resampling_data = None
        self._pcoord_cache = {}

    def _target_count(self, bin_idx):
        """Resolve the requested walker population for one bin.

        ``bin_target_counts`` may be a scalar, sequence, mapping, or callable.
        This adapter gives the decision algorithm one uniform integer lookup
        and rejects nonsensical negative targets. An occupied bin with target
        zero is rejected by :meth:`decide`, where occupancy is known.
        """
        target_spec = self.bin_target_counts

        if isinstance(target_spec, (int, np.integer)):
            target = int(target_spec)
        elif callable(target_spec):
            target = int(target_spec(bin_idx))
        elif isinstance(target_spec, Mapping):
            target = int(target_spec[bin_idx])
        else:
            target = int(target_spec[bin_idx])

        if target < 0:
            raise ResamplerError("Target count for bin {} is negative".format(bin_idx))
        return target

    def _progress_coordinate_for_walker(self, walker):
        """Return one walker's normalized, finite progress coordinate.

        Coordinates are cached by state-object identity for the duration of a
        resampling cycle. Cloning and merging alter weights and ancestry but do
        not alter molecular states, so cached coordinates remain valid while
        avoiding repeated expensive projections such as tICA transforms.

        Returns
        -------
        numpy.ndarray
            A one-dimensional float vector, including shape ``(1,)`` for a
            scalar progress coordinate.
        """
        state = walker.state
        cache_key = id(state)

        if self.cache_progress_coordinates:
            cached = self._pcoord_cache.get(cache_key)
            if cached is not None and cached[0] is state:
                return cached[1]

        coord = np.asarray(self.progress_coordinate(walker), dtype=float)
        if coord.ndim == 0:
            coord = coord.reshape(1)
        else:
            coord = coord.reshape(-1)

        if coord.size == 0:
            raise ResamplerError("progress_coordinate returned an empty coordinate")
        if np.any(~np.isfinite(coord)):
            raise ResamplerError(
                "progress_coordinate returned non-finite values: {}".format(coord)
            )

        if self.cache_progress_coordinates:
            # Keep a strong reference to the state so Python cannot reuse its
            # object id for another state during this resampling cycle.
            self._pcoord_cache[cache_key] = (state, coord)

        return coord

    def _assign_bins(self, walkers):
        """Project the current walkers and return one bin ID per walker.

        This method normalizes the output of either a vectorized mapper with an
        ``assign`` method or a callable mapper evaluated coordinate by
        coordinate. Shape, count, and non-negative-ID checks prevent walker
        records from silently becoming misaligned with bin assignments.
        """
        pcoords = [self._progress_coordinate_for_walker(walker) for walker in walkers]

        try:
            coord_array = np.stack(pcoords, axis=0)
        except ValueError as exc:
            raise ResamplerError(
                "progress_coordinate must return the same number of values for every walker"
            ) from exc

        if hasattr(self.bin_mapper, "assign"):
            assignments = self.bin_mapper.assign(coord_array)
        else:
            assignments = [self.bin_mapper(coord) for coord in coord_array]

        assignments = np.asarray(assignments, dtype=int).reshape(-1)
        if len(assignments) != len(walkers):
            raise ResamplerError(
                "bin_mapper returned {} assignments for {} walkers".format(
                    len(assignments), len(walkers)
                )
            )
        if np.any(assignments < 0):
            raise ResamplerError("bin_mapper returned a negative bin index")
        return assignments

    @staticmethod
    def _normalize_record(record, walker_idx, step_idx):
        """Convert a raw decision into wepy's array-valued record schema.

        ``walker_idx`` identifies the parent within this elementary step and
        ``step_idx`` orders sequential population-changing operations within
        the same WE cycle.
        """
        return {
            "decision_id": np.asarray([int(record["decision_id"])], dtype=int),
            "target_idxs": np.asarray(record["target_idxs"], dtype=int).reshape(-1),
            "step_idx": np.asarray([step_idx], dtype=int),
            "walker_idx": np.asarray([walker_idx], dtype=int),
        }

    def _apply_step(self, walkers, raw_records):
        """Apply and record one elementary variable-population decision step.

        Huber--Kim may require several sequential splits and merges. Applying
        each operation immediately is necessary because it changes the walker
        count and indices used by later operations. This method guarantees that
        every applied step is also normalized and appended to lineage data.
        """
        next_walkers = self.DECISION.action(walkers, [raw_records])
        for walker_idx, record in enumerate(raw_records):
            self._resampling_data.append(
                self._normalize_record(record, walker_idx, self._step_idx)
            )
        self._step_idx += 1
        return next_walkers

    def _split_at(self, walkers, walker_idx, n_children):
        """Replace one walker with ``n_children`` equal-weight clones.

        A complete decision record is constructed for the whole current
        population: walkers before the parent retain their slots, clones occupy
        consecutive slots beginning at ``walker_idx``, and later walkers shift
        right. The decision class performs the actual equal weight division.

        Requests for fewer than two children are a no-op.
        """
        if n_children < 2:
            return walkers

        n_old = len(walkers)
        shift = n_children - 1
        records = []

        for idx in range(n_old):
            if idx < walker_idx:
                records.append(
                    self.DECISION.record(self.DECISION.ENUM.NOTHING.value, target_idxs=(idx,))
                )
            elif idx == walker_idx:
                records.append(
                    self.DECISION.record(
                        self.DECISION.ENUM.CLONE.value,
                        target_idxs=tuple(range(idx, idx + n_children)),
                    )
                )
            else:
                records.append(
                    self.DECISION.record(
                        self.DECISION.ENUM.NOTHING.value,
                        target_idxs=(idx + shift,),
                    )
                )

        return self._apply_step(walkers, records)

    def _merge_at(self, walkers, merge_idxs, keep_idx):
        """Merge a group into one selected survivor and compact output slots.

        Parameters
        ----------
        walkers : sequence of Walker
            Current population.
        merge_idxs : iterable of int
            Indices donating their combined weight to one output walker.
        keep_idx : int
            Member of ``merge_idxs`` whose molecular state survives.

        Notes
        -----
        The keeper receives ``KEEP_MERGE``, donors receive ``SQUASH``, and all
        unaffected walkers receive ``NOTHING`` records. Survivor selection is
        intentionally separate and is normally performed by
        :meth:`_select_merge_survivor`.
        """
        merge_idxs = sorted(set(int(idx) for idx in merge_idxs))
        if len(merge_idxs) < 2:
            return walkers
        if keep_idx not in merge_idxs:
            raise ValueError("keep_idx must be a member of merge_idxs")

        donor_idxs = set(merge_idxs).difference((keep_idx,))
        survivor_old_idxs = [idx for idx in range(len(walkers)) if idx not in donor_idxs]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(survivor_old_idxs)}
        merge_target = old_to_new[keep_idx]

        records = []
        for idx in range(len(walkers)):
            if idx == keep_idx:
                records.append(
                    self.DECISION.record(
                        self.DECISION.ENUM.KEEP_MERGE.value,
                        target_idxs=(merge_target,),
                    )
                )
            elif idx in donor_idxs:
                records.append(
                    self.DECISION.record(
                        self.DECISION.ENUM.SQUASH.value,
                        target_idxs=(merge_target,),
                    )
                )
            else:
                records.append(
                    self.DECISION.record(
                        self.DECISION.ENUM.NOTHING.value,
                        target_idxs=(old_to_new[idx],),
                    )
                )

        return self._apply_step(walkers, records)

    def _indices_in_bin(self, walkers, bin_idx):
        """Return current walker indices assigned to ``bin_idx``.

        Indices are recomputed after population changes because every split or
        merge can shift later walkers. Progress-coordinate caching keeps these
        repeated assignments inexpensive.
        """
        assignments = self._assign_bins(walkers)
        return [int(idx) for idx in np.where(assignments == bin_idx)[0]]

    @staticmethod
    def _index_by_identity(walkers, target_walker):
        """Find a previously selected walker object's current index.

        Identity rather than equality is required because distinct walkers may
        legitimately contain identical states and weights. This lookup avoids
        stale indices after earlier candidates have been split.
        """
        for idx, walker in enumerate(walkers):
            if walker is target_walker:
                return idx
        raise RuntimeError("Walker selected for an operation is no longer present")

    def _select_merge_survivor(self, walkers, merge_idxs):
        """Select a merge survivor with probability proportional to weight.

        For merge group ``G``, walker ``i`` survives with probability
        ``weight[i] / sum(weight[j] for j in G)``. This stochastic rule makes
        the surviving state an unbiased representative of the merged
        probability. The returned value is an index into ``walkers``.
        """
        weights = np.asarray([walkers[idx].weight for idx in merge_idxs], dtype=float)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ResamplerError("Merge group has invalid total weight {}".format(total))
        cumulative = np.cumsum(weights)
        draw = self.rng.uniform(0.0, total)
        local_idx = int(np.searchsorted(cumulative, draw, side="right"))
        if local_idx >= len(merge_idxs):
            local_idx = len(merge_idxs) - 1
        return merge_idxs[local_idx]

    def _split_by_weight(self, walkers, bin_idx, ideal_weight):
        """Apply the normal bin-relative Huber--Kim split pass.

        Walkers heavier than ``weight_split_threshold * ideal_weight`` are
        split into ``ceil(weight / ideal_weight)`` children. Candidate objects
        are saved rather than candidate indices because earlier splits shift
        the current walker list.

        This pass balances statistical weight; it does not by itself guarantee
        the requested final count.
        """
        bin_idxs = self._indices_in_bin(walkers, bin_idx)
        candidates = [
            walkers[idx]
            for idx in sorted(bin_idxs, key=lambda i: (walkers[i].weight, i))
            if walkers[idx].weight > self.weight_split_threshold * ideal_weight
        ]

        for candidate in candidates:
            idx = self._index_by_identity(walkers, candidate)
            n_children = int(math.ceil(candidate.weight / ideal_weight))
            walkers = self._split_at(walkers, idx, n_children)

        return walkers

    def _merge_by_weight(self, walkers, bin_idx, ideal_weight):
        """Repeatedly merge an eligible light-weight prefix in one bin.

        Current bin members are sorted from lightest to heaviest. The largest
        prefix with cumulative weight no greater than
        ``weight_merge_cutoff * ideal_weight`` is merged when it contains at
        least two walkers. The procedure repeats because each merge changes
        both the population and the next eligible prefix.
        """
        while True:
            bin_idxs = self._indices_in_bin(walkers, bin_idx)
            sorted_idxs = sorted(bin_idxs, key=lambda i: (walkers[i].weight, i))
            if len(sorted_idxs) < 2:
                return walkers

            weights = np.asarray([walkers[idx].weight for idx in sorted_idxs], dtype=float)
            cumulative = np.cumsum(weights)
            mask = cumulative <= ideal_weight * self.weight_merge_cutoff
            merge_idxs = [idx for idx, selected in zip(sorted_idxs, mask) if selected]

            if len(merge_idxs) < 2:
                return walkers

            keep_idx = self._select_merge_survivor(walkers, merge_idxs)
            walkers = self._merge_at(walkers, merge_idxs, keep_idx)

    def _adjust_count(self, walkers, bin_idx, target_count):
        """Force an occupied bin to exactly ``target_count`` walkers.

        While below target, the heaviest walker is split in two. While above
        target, the two lightest walkers are merged with proportional-weight
        survivor selection. This implements WESTPA's count-adjusted mode after
        the normal relative-weight Huber--Kim pass.
        """
        while True:
            bin_idxs = self._indices_in_bin(walkers, bin_idx)
            if len(bin_idxs) >= target_count:
                break
            split_idx = max(bin_idxs, key=lambda i: (walkers[i].weight, -i))
            walkers = self._split_at(walkers, split_idx, 2)

        while True:
            bin_idxs = self._indices_in_bin(walkers, bin_idx)
            if len(bin_idxs) <= target_count:
                break
            sorted_idxs = sorted(bin_idxs, key=lambda i: (walkers[i].weight, i))
            merge_idxs = sorted_idxs[:2]
            keep_idx = self._select_merge_survivor(walkers, merge_idxs)
            walkers = self._merge_at(walkers, merge_idxs, keep_idx)

        return walkers

    def _split_by_threshold(self, walkers, bin_idx):
        """Enforce the absolute maximum walker-weight safeguard in one bin.

        Each walker above ``largest_allowed_weight`` is divided into
        ``ceil(weight / largest_allowed_weight)`` children. Unlike
        :meth:`_split_by_weight`, this limit is global and independent of the
        bin's ideal weight.
        """
        bin_idxs = self._indices_in_bin(walkers, bin_idx)
        candidates = [
            walkers[idx]
            for idx in sorted(bin_idxs, key=lambda i: (walkers[i].weight, i))
            if walkers[idx].weight > self.largest_allowed_weight
        ]

        for candidate in candidates:
            idx = self._index_by_identity(walkers, candidate)
            n_children = int(math.ceil(candidate.weight / self.largest_allowed_weight))
            walkers = self._split_at(walkers, idx, n_children)

        return walkers

    def _merge_by_threshold(self, walkers, bin_idx):
        """Merge groups of walkers below the absolute minimum weight.

        All current bin members lighter than ``smallest_allowed_weight`` form
        an eligible merge group when at least two exist. The pass repeats after
        each merge. This is a numerical/absolute-weight safeguard rather than
        the normal bin-relative merge rule.
        """
        while True:
            bin_idxs = self._indices_in_bin(walkers, bin_idx)
            sorted_idxs = sorted(bin_idxs, key=lambda i: (walkers[i].weight, i))
            merge_idxs = [
                idx for idx in sorted_idxs if walkers[idx].weight < self.smallest_allowed_weight
            ]
            if len(merge_idxs) < 2:
                return walkers

            keep_idx = self._select_merge_survivor(walkers, merge_idxs)
            walkers = self._merge_at(walkers, merge_idxs, keep_idx)

    def _validate_final_population(self, n_initial, n_final):
        """Check output count against configured wepy population bounds.

        ``None`` is unbounded, an integer is a hard bound, and ``Ellipsis`` is
        a dynamic bound equal to the input population. The method validates but
        does not alter a population to satisfy a bound.
        """
        min_setting = self.min_num_walkers_setting
        max_setting = self.max_num_walkers_setting

        if min_setting is Ellipsis and n_final < n_initial:
            raise ResamplerError(
                "Final walker count {} is below dynamic minimum {}".format(n_final, n_initial)
            )
        if max_setting is Ellipsis and n_final > n_initial:
            raise ResamplerError(
                "Final walker count {} exceeds dynamic maximum {}".format(n_final, n_initial)
            )
        if isinstance(min_setting, int) and n_final < min_setting:
            raise ResamplerError(
                "Final walker count {} is below minimum {}".format(n_final, min_setting)
            )
        if isinstance(max_setting, int) and n_final > max_setting:
            raise ResamplerError(
                "Final walker count {} exceeds maximum {}".format(n_final, max_setting)
            )

    def decide(self, walkers, occupied_bins=None):
        """Apply Huber--Kim decisions and return the resulting population.

        Parameters
        ----------
        walkers : sequence of Walker
            Valid current walkers for an active :meth:`resample` call.
        occupied_bins : iterable of int, optional
            Bins to process. When omitted they are inferred from ``walkers``.

        Returns
        -------
        decided_walkers : list of Walker
            Population after all bin-local split and merge decisions.
        resampler_data : list of dict
            One diagnostic summary for every processed occupied bin.

        Notes
        -----
        This method is the policy layer analogous to ``REVOResampler.decide``.
        There is one important structural difference. REVO preserves a fixed
        population, so it can plan a net clone/merge table against the original
        indices and apply it once. Huber--Kim can change population repeatedly;
        each split or merge changes the indices needed by the next decision.
        Consequently this method applies and records elementary decisions as it
        makes them instead of returning one static decision table.

        ``resample`` initializes the record/cache lifecycle and performs input
        and output validation. This method should normally not be called
        directly by user code.
        """

        current_walkers = list(walkers)
        if occupied_bins is None:
            assignments = self._assign_bins(current_walkers)
            occupied_bins = sorted(int(x) for x in np.unique(assignments))

        resampler_data = []
        for bin_idx in occupied_bins:
            pre_idxs = self._indices_in_bin(current_walkers, bin_idx)
            pre_count = len(pre_idxs)
            target_count = self._target_count(bin_idx)

            if target_count == 0:
                raise ResamplerError(
                    "Occupied bin {} has target count 0 ({} walkers present)".format(
                        bin_idx, pre_count
                    )
                )

            bin_weight = float(
                sum(current_walkers[idx].weight for idx in pre_idxs)
            )
            ideal_weight = bin_weight / target_count
            if not np.isfinite(ideal_weight) or ideal_weight <= 0:
                raise ResamplerError(
                    "Bin {} has invalid ideal weight {}".format(
                        bin_idx, ideal_weight
                    )
                )

            # WESTPA single-subgroup / Huber-Kim order.
            current_walkers = self._split_by_weight(
                current_walkers, bin_idx, ideal_weight
            )
            current_walkers = self._merge_by_weight(
                current_walkers, bin_idx, ideal_weight
            )

            if self.adjust_counts:
                current_walkers = self._adjust_count(
                    current_walkers, bin_idx, target_count
                )

            if self.do_thresholds:
                current_walkers = self._split_by_threshold(
                    current_walkers, bin_idx
                )
                current_walkers = self._merge_by_threshold(
                    current_walkers, bin_idx
                )

            post_count = len(self._indices_in_bin(current_walkers, bin_idx))
            resampler_data.append(
                {
                    "bin_idx": np.asarray([bin_idx], dtype=int),
                    "target_count": np.asarray([target_count], dtype=int),
                    "pre_count": np.asarray([pre_count], dtype=int),
                    "post_count": np.asarray([post_count], dtype=int),
                    "bin_weight": np.asarray([bin_weight], dtype=float),
                    "ideal_weight": np.asarray([ideal_weight], dtype=float),
                }
            )

        return current_walkers, resampler_data

    def resample(self, walkers, **kwargs):
        """Run one complete, validated Huber--Kim resampling cycle.

        This is the public wepy entry point. It owns the per-cycle cache and
        decision-record lifecycle, validates positive finite input weights,
        delegates scientific allocation policy to :meth:`decide`, emits an
        identity lineage step when nothing changes, checks probability
        conservation and population bounds, and guarantees cleanup in a
        ``finally`` block.

        Returns
        -------
        resampled_walkers : list of Walker
            Ensemble to propagate in the next cycle.
        resampling_data : list of dict
            Array-valued elementary clone/merge records used for lineage.
        resampler_data : list of dict
            Per-bin population and weight diagnostics.
        """

        self._resample_init(walkers=walkers, **kwargs)
        self._step_idx = 0
        self._resampling_data = []
        self._pcoord_cache = {}

        current_walkers = list(walkers)
        n_initial = len(current_walkers)

        initial_weights = np.asarray([walker.weight for walker in current_walkers], dtype=float)
        if np.any(~np.isfinite(initial_weights)) or np.any(initial_weights <= 0):
            self._resample_cleanup()
            raise ResamplerError("All walker weights must be finite and strictly positive")
        initial_total_weight = float(initial_weights.sum())

        resampler_data = []

        try:
            initial_assignments = self._assign_bins(current_walkers)
            occupied_bins = sorted(int(x) for x in np.unique(initial_assignments))
            current_walkers, resampler_data = self.decide(
                current_walkers, occupied_bins=occupied_bins
            )

            # Even a pass-through WE cycle needs a decision step so that
            # wepy's ancestry machinery can connect this generation to the
            # next one.
            if self._step_idx == 0:
                identity_records = [
                    self.DECISION.record(
                        self.DECISION.ENUM.NOTHING.value, target_idxs=(idx,)
                    )
                    for idx in range(len(current_walkers))
                ]
                current_walkers = self._apply_step(
                    current_walkers, identity_records
                )

            final_weights = np.asarray(
                [walker.weight for walker in current_walkers], dtype=float
            )
            if np.any(~np.isfinite(final_weights)) or np.any(final_weights <= 0):
                raise ResamplerError("Resampling produced a non-positive or non-finite weight")

            final_total_weight = float(final_weights.sum())
            if not np.isclose(
                initial_total_weight,
                final_total_weight,
                rtol=1e-12,
                atol=1e-15,
            ):
                raise ResamplerError(
                    "Walker weight was not conserved: before={} after={}".format(
                        initial_total_weight, final_total_weight
                    )
                )

            self._validate_final_population(n_initial, len(current_walkers))

            return current_walkers, self._resampling_data, resampler_data

        finally:
            self._pcoord_cache = {}
            self._resample_cleanup(
                resampling_data=self._resampling_data,
                resampler_data=resampler_data,
                walkers=current_walkers,
            )
