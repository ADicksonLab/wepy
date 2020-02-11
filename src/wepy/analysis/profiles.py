"""Analysis routines for generating free energy profiles over wepy
simulation data.

"""
import gc
import itertools as it

import numpy as np

from geomm.free_energy import free_energy

def _percentage_cumulative_partitions(ensemble_values,
                                     percentages=[0.25, 0.5, 0.75, 1.0]):

    n_cycles = ensemble_values.shape[0]
    n_trajs = ensemble_values.shape[1]

    start_idx = 0
    partitions = []
    for percentage in percentages:

        # get the percentage of the cycles based on this percentage
        end_idx = int(n_cycles // (1 / percentage))

        partitions.append((start_idx, end_idx))

    return partitions


def _tranche_cumulative_partitions(ensemble_values,
                                  time_tranche=None,
                                  num_partitions=4):
    n_cycles = ensemble_values.shape[0]
    n_trajs = ensemble_values.shape[1]

    start_idx = 0

    # if time_tranche was given determine the number of partitions
    if time_tranche is not None and time_tranche > 0:
        num_partitions = n_cycles // time_tranche

        # get the end indices for each tranche
        end_idxs = []
        for i in range(num_partitions):
            end_idx = time_tranche * (i + 1)

        # and create the array of partitions
        partitions = [(start_idx, end_idx) for end_idx in end_idxs]

        # if there is a remainder we will have another partition for
        # the remainder
        remainder = n_cycles % time_tranche
        if remainder > 0:
            num_partitions += 1
            end_idx = remainder + (tim_tranche * num_partitions)

            partitions.append((start_idx, end_idx))

    # check that one of them is defined
    elif num_partitions is None and num_partitions > 0:
        raise ValueError("Must specify either time_tranche size or partitions ratio")

    # if the partitions are given and the time_tranche is not we
    # determine the tranche size automatically
    else:
        time_tranche = n_cycles // num_partitions

        end_idxs = []
        for i in range(num_partitions):
            end_idx = time_tranche * (i + 1)

        # and create the array of partitions
        partitions = [(start_idx, end_idx) for end_idx in end_idxs]

        # if there is a remainder in this case we add it to the first
        # partition since the number of partitions is invariant
        # (unlike time_tranche option)
        remainder = n_cycles % time_tranche
        if remainder > 0:
            partitions[0][1] += remainder

    return partitions

# REFACTOR: this has an overly complex decision process encapsulated
# here, should be refactored. Was not properly done because of time
# constraints
def cumulative_partitions(ensemble_values,
                          time_tranche=None,
                          num_partitions=5,
                          percentages=None):
    """Calculate the cumulative free energies for an ensemble of
    trajectory weights.

    Parameters
    ----------

    ensemble_values : arraylikes of float of shape (n_cycles, n_trajs)
        Array of scalar values for all of the frames of an ensemble simulation.

    time_tranche : int
        Width of time-bins/windows/tranche to compute free energies
        for. This overrides the functionality of the 'partitions'
        argument. The last tranche will be truncated to the remainder
        after partitioning by tranches.

    num_partitions : int
        Will evenly partition the time coordinate given the
        dataset. The first partition will carry the
        remainder. Overriden by time_tranche.

    percentages: list of float, optional

    Yields
    ------

    cumulative_tranche : arraylike
        A slice along the cycles axis of the ensemble values starting
        at the beginning and including up to the end of the next
        tranche.

    """


    if percentages is None:
        partitions = _tranche_cumulative_partitions(ensemble_values,
                                                    time_tranche=time_tranche,
                                                    num_partitions=num_partitions)

    else:
        partitions = _percentage_cumulative_partitions(ensemble_values,
                                                       percentages=percentages)

    # yield the time-partitioned values when asked for
    for start_idx, end_idx in partitions:

        # compute the free energies
        yield ensemble_values[start_idx:end_idx]

def free_energy_profile(weights, observables, bins=30,
                  max_energy=100,
                  zero_point_energy=1e-12):
    """
    Parameters
    ----------

    weights : arraylikes of float of shape (n_trajs, n_cycles)
        The weights for all of the frames of an ensemble simulation.

    observables : arraylikes of shape (n_trajs, n_cycles)
        The scalar observables to compute the free energy over.

    bins : int or sequence of scalars or str
        The number of bins to bin along the observable axis, or the
        actual bin edges, or the spec for the binning method.

    Returns
    -------

    hist_fe : arraylike
        The free energies of the histogram bins

    bin_edges : arraylike
        The edges of the bins of the histogram.

    """

    assert weights.shape == observables.shape, "Weights and observables must correspond in shape"

    hist_weights, bin_edges = np.histogram(observables, weights=weights, bins=bins)

    hist_fe = free_energy(hist_weights,
                          max_energy=np.nan,
                          zero_point_energy=zero_point_energy)

    return hist_fe, bin_edges

def contigtrees_bin_edges(contigtrees, bins, field_key,
                          truncate_cycles=None):
    """Get the bin edges that best describes the aggregate data for
    multiple contigtrees.

    Parameters
    ----------

    contigtrees : list of ContigTree objects
        The contigtrees to draw the data from.

    bins : int or str or sequence of scalars or function
        The number of bins to make or the method to use for
        autobinning or the actual specification of bins. See
        `numpy.histogram_bin_edges` for exact meaning and methods.

    field_key : str
        The trajectory field key for the values to get bins for.

    truncate_cycles : None or int
        If None include all cycles of a run. Otherwise only consider
        cycles for spans up to (exclusive) the cycle index given.

    Returns
    -------
    bin_edges : array of dtype float
        The edges to pass into `numpy.histogram`


    See Also
    --------
    numpy.histogram_bin_edges : For the bin edge function

    """

    all_values = []

    for contigtree in contigtrees:
        with contigtree:

            contigtree_values = []
            for span_idx in contigtree.span_traces.keys():

                contig = contigtree.span_contig(span_idx)

                if truncate_cycles is not None:
                    contig_values = contig.contig_fields([field_key])\
                                    [field_key][0:truncate_cycles]

                else:
                    contig_values = contig.contig_fields([field_key])[field_key]

                contigtree_values.append(contig_values.reshape(-1))

            all_values.extend(contigtree_values)

    all_values = np.concatenate(all_values)
    gc.collect()

    if callable(bins):
        bin_edges = bins(all_values)

    else:
        bin_edges = np.histogram_bin_edges(all_values, bins=bins)

    # If the histogram_bin_edges actually used the weights we would
    # call it like this and actually have to retrieve the weights,
    # currently it doesn't so I won't implement that since it would
    # slow things down and not do anything

    # bin_edges = np.histogram_bin_edges(all_values, bins=bins,
    #                                    weights=all_weights)

    return bin_edges


class ContigTreeProfiler(object):
    """A wrapper class around a ContigTree that provides extra methods for
    generating free energy profiles for observables."""

    def __init__(self, contigtree,
                 truncate_cycles=None):
        """Create a wrapper around a contigtree for the profiler.

        Parameters
        ----------

        contigtree : ContigTree object
            The contigtree you want to generate profiles for.

        truncate_cycles : None or int
            If None include all cycles of a run. Otherwise only consider
            cycles for spans up to (exclusive) the cycle index given.

        """

        self._contigtree = contigtree

        self._truncate_cycles = truncate_cycles

        # get the ignored frames trace as a set
        if truncate_cycles is not None:
            self._ignore_trace = self._get_ignore_trace(self._contigtree, truncate_cycles)
        else:
            self._ignore_trace = set()

        # determine the global binning strategy. These are the bins
        # that best describe the entire pool of data in this contig
        # tree

    @classmethod
    def _get_ignore_trace(cls, contigtree, truncate_cycles):
        """Get a trace of the ignore frames if we are truncating on a
        particular cycle index. Is exclusive of the `truncate_cycles`
        value.

        Parameters
        ----------

        contigtree : ContigTree

        truncate_cycles : int
            The cycle index to truncate at (exclusive).

        Returns
        -------

        ignored_trace : set of (int, int)
            The frames to ignore given the truncation.

        """

        ignore_trace = set()
        for span_idx, span_trace in contigtree.span_traces.items():

            for run_idx, cycle_idx in span_trace:
                if cycle_idx >= truncate_cycles:
                    ignore_trace.add((run_idx, cycle_idx,))

        return ignore_trace

    @property
    def contigtree(self):
        """The underlying contigtree this wraps."""
        return self._contigtree

    def fe_profile_trace(self, trace, field_key,
                         bins=None,
                         ignore_truncate=False):
        """Calculate the free energy histogram over a trajectory field.

        Parameters
        ----------

        trace : list of tuple of ints (run_idx, traj_idx, cycle_idx)

        field_key : str
            The key for the trajectory field to calculate the profiles
            for. Must be a rank 0 (or equivalent (1,) rank) field.

        bins : int or sequence of scalars or str or None
            The number of bins to bin along the observable axis, or
            the actual bin edges, or the spec for the binning method,
            or None. If None will generate them with the 'auto'
            setting for this span.

        ignore_truncate : bool
            Ignore the truncate_cycles value.

        Returns
        -------

        fe_profile : arraylike of dtype float
            An array of free energies for each bin.

        """


        if not ignore_truncate:
            ignored_trace = []
            for key in trace:
                if key not in self._ignore_trace:
                    ignored_trace.append(key)

            trace = ignored_trace

        fields = self.contigtree.wepy_h5.get_trace_fields(trace, [field_key, 'weights'])

        weights = fields['weights'].flatten()
        values = fields[field_key]

        # the feature vector must be either shape 0 or (1,) i.e. rank
        # 0 equivalent, so we get one feature and check its shape
        test_feature = values[0]

        # if it doesn't satisfy this requirement we need to raise an error
        if not (test_feature.shape == () or test_feature.shape == (1,)):
            raise TypeError("The field key specified a field that is non-scalar."
                            "The shape of the feature vector must be () or (1,).")

        # determine the bin_edges

        # if the bins were not specified generate them from all the
        # values for this span
        if bins is None:
            bin_edges = np.histogram_bin_edges(values, bins='auto',
                                               weights=weights)

        # if the number or binning strategy was given generate them
        # according to that
        elif type(bins) in (str, int):
            bin_edges = np.histogram_bin_edges(values, bins=bins,
                                               weights=weights)

        # if it wasn't those things we assume it is already a correct
        # bin_edges
        else:
            bin_edges = bins


        fe_profile, _ = free_energy_profile(weights, values,
                                            bins=bin_edges)

        return fe_profile


    def fe_profile_all(self, field_key, bins=None,
                       ignore_truncate=False):
        """Calculate the free energy histogram over a trajectory field.

        Parameters
        ----------

        field_key : str
            The key for the trajectory field to calculate the profiles
            for. Must be a rank 0 (or equivalent (1,) rank) field.

        bins : int or sequence of scalars or str or None
            The number of bins to bin along the observable axis, or
            the actual bin edges, or the spec for the binning method,
            or None. If None will generate them with the 'auto'
            setting for this span.

        ignore_truncate : bool
            Ignore the truncate_cycles value.


        Returns
        -------

        fe_profile : arraylike of dtype float
            An array of free energies for each bin.

        """

        all_trace = list(it.chain(*[span_trace for span_trace in self.contigtree.span_traces.values()]))

        # this truncates based on the frame specs so doesn't need to
        # be ordered or anything
        fe_profile = self.fe_profile_trace(all_trace, field_key,
                                           bins=bins,
                                           ignore_truncate=ignore_truncate)

        return fe_profile


    def fe_profile(self, span, field_key,
                   bins=None,
                   ignore_truncate=False):
        """Calculate the free energy histogram over a trajectory field.

        Deprecation warning: This will likely be deprecated and
        renamed to fe_profile_span because it is misleading that a
        profile is span based.

        Parameters
        ----------

        span : int
            The index of the span to calculate profiles for.

        field_key : str
            The key for the trajectory field to calculate the profiles
            for. Must be a rank 0 (or equivalent (1,) rank) field.

        bins : int or sequence of scalars or str or None
            The number of bins to bin along the observable axis, or
            the actual bin edges, or the spec for the binning method,
            or None. If None will generate them with the 'auto'
            setting for this span.

        ignore_truncate : bool
            Ignore the truncate_cycles value.


        Returns
        -------

        fe_profile : arraylike of dtype float
            An array of free energies for each bin.

        """

        # make the contig for this span
        contig = self.contigtree.span_contig(span)

        if not ignore_truncate:
            # get the weights
            weights = contig.contig_fields(['weights'])['weights'][0:self._truncate_cycles]

            # then get the values for the field key
            values = contig.contig_fields([field_key])[field_key][0:self._truncate_cycles]

        else:
            # get the weights
            weights = contig.contig_fields(['weights'])['weights'][0:self._truncate_cycles]

            # then get the values for the field key
            values = contig.contig_fields([field_key])[field_key][0:self._truncate_cycles]

        # reshape to match
        weights = weights.reshape((weights.shape[0], weights.shape[1]))

        # the feature vector must be either shape 0 or (1,) i.e. rank
        # 0 equivalent

        # so we get one feature and check its shape
        test_feature = values[0,0]

        # if it doesn't satisfy this requirement we need to raise an error
        if not (test_feature.shape == () or test_feature.shape == (1,)):
            raise TypeError("The field key specified a field that is non-scalar."
                            "The shape of the feature vector must be () or (1,).")

        # determine the bin_edges

        # if the bins were not specified generate them from all the
        # values for this span
        if bins is None:
            bin_edges = np.histogram_bin_edges(values, bins='auto',
                                               weights=weights)

        # if the number or binning strategy was given generate them
        # according to that
        elif type(bins) in (str, int):
            bin_edges = np.histogram_bin_edges(values, bins=bins,
                                               weights=weights)

        # if it wasn't those things we assume it is already a correct
        # bin_edges
        else:
            bin_edges = bins


        fe_profile, _ = free_energy_profile(weights, values,
                                            bins=bin_edges)

        return fe_profile


    def fe_cumulative_profiles(self, span, field_key,
                               bins=None,
                               time_tranche=None,
                               num_partitions=5,
                               ignore_truncate=False):
        """Calculate the cumulative free energy histograms for a trajectory
        field.

        Parameters
        ----------

        span : int
            The index of the span to calculate profiles for.

        field_key : str
            The key for the trajectory field to calculate the profiles
            for. Must be a rank 0 (or equivalent (1,) rank) field.

        bins : int or sequence of scalars or str or None
            The number of bins to bin along the observable axis, or
            the actual bin edges, or the spec for the binning method,
            or None. If None will generate them with the 'auto'
            setting for this span.

        time_tranche : int
            Width of time-bins/windows/tranche to compute free
            energies for. This overrides the functionality of the
            'num_partitions' argument. The last tranche will be truncated
            to the remainder after partitioning by tranches. Overrides
            partitions.

        num_partitions : int
            Will evenly partition the time coordinate given the
            dataset. The first partition will carry the remainder.
            Overriden by time_tranche.

        ignore_truncate : bool
            Ignore the truncate_cycles value.

        Returns
        -------

        cumulative_fe_profiles : list of arraylike of dtype float
            A list of each cumulative free energy profile. Each profile is
            an array of free energies for each bin.

        profile_num_cycles : list of int
            The number of cycles in each cumulative fe profile.

        """

        # make the contig for this span
        contig = self.contigtree.span_contig(span)

        if not ignore_truncate:
            # get the weights
            weights = contig.contig_fields(['weights'])['weights'][0:self._truncate_cycles]
            # then get the values for the field key
            values = contig.contig_fields([field_key])[field_key][0:self._truncate_cycles]

        else:
            # get the weights
            weights = contig.contig_fields(['weights'])['weights']
            # then get the values for the field key
            values = contig.contig_fields([field_key])[field_key]

        # reshape to match
        weights = weights.reshape((weights.shape[0], weights.shape[1]))

        # determine the bin_edges

        # if the bins were not specified generate them from all the
        # values for this span
        if bins is None:
            bin_edges = np.histogram_bin_edges(values, bins='auto',
                                               weights=weights)

        # if the number or binning strategy was given generate them
        # according to that
        elif type(bins) in (str, int):
            bin_edges = np.histogram_bin_edges(values, bins=bins,
                                               weights=weights)

        # if it wasn't those things we assume it is already a correct
        # bin_edges
        else:
            bin_edges = bins


        # make the cumulative generators for each
        weights_partition_gen = cumulative_partitions(weights, time_tranche=time_tranche,
                                                      num_partitions=num_partitions)
        values_partition_gen = cumulative_partitions(values, time_tranche=time_tranche,
                                                     num_partitions=num_partitions)

        # then generate each profile for each cumulative tranche

        fe_profiles = []
        profile_num_cycles = []
        for weights_part in weights_partition_gen:

            values_part = next(values_partition_gen)

            profile_num_cycles.append(len(values_part))

            fe_profile, _ = free_energy_profile(weights_part, values_part,
                                                bins=bin_edges)

            fe_profiles.append(fe_profile)

        return fe_profiles, profile_num_cycles


    def fe_all_cumulative_profiles(self, field_key,
                                   bins=None,
                                   time_tranche=None,
                                   num_partitions=5,
                                   percentages=None,
                                   ignore_truncate=False):
        """Calculate the cumulative free energy histograms for a trajectory
        field for all spans in a contigtree.

        Parameters
        ----------

        field_key : str
            The key for the trajectory field to calculate the profiles
            for. Must be a rank 0 (or equivalent (1,) rank) field.

        bins : int or sequence of scalars or str or None
            The number of bins to bin along the observable axis, or
            the actual bin edges, or the spec for the binning method,
            or None. If None will generate them with the 'auto'
            setting for this span.

        time_tranche : int
            Width of time-bins/windows/tranche to compute free
            energies for. This overrides the functionality of the
            'num_partitions' argument. The last tranche will be truncated
            to the remainder after partitioning by tranches. Overrides
            partitions.

        num_partitions : int
            Will evenly partition the time coordinate given the
            dataset. The first partition will carry the remainder.
            Overriden by time_tranche.

        percentages : list of float

        ignore_truncate : bool
            Ignore the truncate_cycles value.

        Returns
        -------

        cumulative_fe_profiles : list of arraylike of dtype float
            A list of each cumulative free energy profile. Each profile is
            an array of free energies for each bin.

        profile_num_cycles : list of int
            The number of cycles in each cumulative fe profile.

        """


        # There are two big use cases here.
        #
        # 1. We are passing in the bin edges since we want to plot
        # this profile with other data on the same scale
        #
        # 2. We are just wanting to show this one on its own.
        #
        # In the former we don't do anything and fail if it doesn't
        # work. In the latter we want to infer bins like normal. The
        # first thing we do is distinguish between which case we have.

        # if the bins were not specified generate them from all the
        # values for this span
        if bins is None or type(bins) in (str, int):

            # if we are doing the former we need to get the bins for
            # all of the data first. Once we have that we worry about
            # doing the tranches etc. So get all the data, get the
            # bins, and then delete it and start fresh with the
            # tranches

            # we use a trace orientation here for simplicity in truncation

            # trace of all of the frames in the contigtree
            all_trace = list(it.chain(*[span_trace for span_trace in
                                        self.contigtree.span_traces.values()]))

            # filter it for the ignored fields if applicable
            if not ignore_truncate:
                ignored_trace = []
                for key in all_trace:
                    if key not in self._ignore_trace:
                        ignored_trace.append(key)

                trace = ignored_trace

            else:
                trace = all_trace

            # get the fields for the weights and the field
            fields = self.contigtree.wepy_h5.get_trace_fields(trace, [field_key, 'weights'])


            all_weights = fields['weights']
            all_values = fields[field_key]

            all_weights = all_weights.flatten()
            gc.collect()

            # the feature vector must be either shape 0 or (1,) i.e. rank
            # 0 equivalent

            # so we get one feature and check its shape
            test_feature = all_values[0]

            # if it doesn't satisfy this requirement we need to raise an error
            if not (test_feature.shape == () or test_feature.shape == (1,)):
                raise TypeError("The field key specified a field that is non-scalar."
                                "The shape of the feature vector must be () or (1,).")

            # then get the bin edges according to the specific option
            if bins is None:
                bin_edges = np.histogram_bin_edges(all_values, bins='auto',
                                                   weights=all_weights)

            # if the number or binning strategy was given generate them
            # according to that
            elif type(bins) in (str, int):
                bin_edges = np.histogram_bin_edges(all_values, bins=bins,
                                                   weights=all_weights)

            # clean up these values since we will be loading in new
            # values that are structured in a different way.
            del all_values
            del all_weights
            gc.collect()

        # if it wasn't those things we assume it is already a correct
        # bin_edges
        else:
            bin_edges = bins

        # now we need to decide on the length of the interval that we
        # will be using so that we can partition it and then use that
        # to grab the right subset of data

        spans_weights_part_gens = []
        spans_values_part_gens = []
        for span_idx in self.contigtree.span_traces.keys():

            contig = self.contigtree.span_contig(span_idx)

            if not ignore_truncate:
                # get the weights
                contig_weights = contig.contig_fields(['weights'])['weights'][0:self._truncate_cycles]
                # then get the values for the field key
                contig_values = contig.contig_fields([field_key])[field_key][0:self._truncate_cycles]

            else:
                # get the weights
                contig_weights = contig.contig_fields(['weights'])['weights']
                # then get the values for the field key
                contig_values = contig.contig_fields([field_key])[field_key]

            # reshape to match
            contig_weights = contig_weights.reshape((contig_weights.shape[0],
                                                     contig_weights.shape[1]))

            # make the cumulative generators for each
            contig_weights_partition_gen = cumulative_partitions(contig_weights,
                                                          time_tranche=time_tranche,
                                                                 num_partitions=num_partitions,
                                                                 percentages=percentages)
            contig_values_partition_gen = cumulative_partitions(contig_values,
                                                         time_tranche=time_tranche,
                                                                num_partitions=num_partitions,
                                                                percentages=percentages)

            # we want to include all spans at once so save these for the next part
            spans_weights_part_gens.append(contig_weights_partition_gen)
            spans_values_part_gens.append(contig_values_partition_gen)


        part_profiles = []
        # use the first one for the main loop iteration
        for first_weights_part in spans_weights_part_gens[0]:

            gc.collect()

            # then add the other ones
            weights_parts = [first_weights_part.reshape(-1)]
            for other_weights_part_gen in spans_weights_part_gens[1:]:

                weights_part = next(other_weights_part_gen)
                weights_parts.append(weights_part.reshape(-1))

            values_parts = []
            for values_part_gen in spans_values_part_gens:

                values_part = next(values_part_gen)
                values_parts.append(values_part.reshape(-1))

            # concatenate them to feed them to the main function
            agg_weights_part = np.concatenate(weights_parts)
            agg_values_part = np.concatenate(values_parts)

            # then compute the FE curve from the aggregate of spans
            part_profile, _ = free_energy_profile(agg_weights_part,
                                                  agg_values_part,
                                                  bins=bin_edges)

            part_profiles.append(part_profile)

        return part_profiles

    def bin_edges(self, bins, field_key):
        """Compute the bin edges that best describes all of the contigtree
        data for the number of bins.

        Parameters
        ----------

        bins : int or str
            The number of bins to make or the method to use for
            autobinning.

        field_key : str
            The trajectory field key for the values to get bins for.

        Returns
        -------
        bin_edges : array of dtype float
            The edges to pass into `numpy.histogram`


        See Also
        --------
        numpy.histogram_bin_edges : For the bin edge function

        """

        all_values = np.concatenate([fields[field_key]
                                     for fields
                                     in self.contigtree.wepy_h5.iter_trajs_fields([field_key])])

        bin_edges = np.histogram_bin_edges(all_values, bins=bins)

        # If the histogram_bin_edges actually used the weights we would
        # call it like this and actually have to retrieve the weights,
        # currently it doesn't so I won't implement that since it would
        # slow things down and not do anything

        # bin_edges = np.histogram_bin_edges(all_values, bins=bins,
        #                                    weights=all_weights)


        return bin_edges

