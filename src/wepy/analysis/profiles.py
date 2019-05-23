"""Analysis routines for generating free energy profiles over wepy
simulation data.

"""
import numpy as np

from geomm.free_energy import free_energy

def cumulative_partitions(ensemble_values,
                          time_tranche=None,
                          num_partitions=5):
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

    Yields
    ------

    cumulative_tranche : arraylike
        A slice along the cycles axis of the ensemble values starting
        at the beginning and including up to the end of the next
        tranche.

    """


    n_cycles = ensemble_values.shape[0]
    n_trajs = ensemble_values.shape[1]

    # if time_tranche was given determine the number of partitions
    if time_tranche is not None and time_tranche > 0:
        num_partitions = n_cycles // time_tranche
        partitions = [time_tranche for _ in range(num_partitions)]

        # if there is a remainder we will have another partition for
        # the remainder
        remainder = n_cycles % time_tranche
        if remainder > 0:
            num_partitions += 1
            partitions.append(remainder)

    # check that one of them is defined
    elif num_partitions is None and num_partitions > 0:
        raise ValueError("Must specify either time_tranche size or partitions ratio")

    # if the partitions are given and the time_tranche is not we
    # determine the tranche size automatically
    else:
        time_tranche = n_cycles // num_partitions
        partitions = [time_tranche for _ in range(num_partitions)]
        remainder = n_cycles % time_tranche
        if remainder > 0:
            partitions[0] += remainder

    # yield the time-partitioned values when asked for
    for p_idx in range(len(partitions)):

        # get the index for the end of the slice for this partition
        end_idx = sum(partitions[0:p_idx+1]) #- 1

        # compute the free energies
        yield ensemble_values[0:end_idx]

def free_energy_profile(weights, observables, bins=30,
                  max_energy=100,
                  zero_point_energy=1e-12):
    """
    Parameters
    ----------

    free_energies : arraylikes of float of shape (n_trajs, n_cycles)
        The free energies for all of the frames of an ensemble simulation.

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

def contigtrees_bin_edges(contigtrees, bins, field_key):
    """Get the bin edges that best describes the aggregate data for
    multiple contigtrees.

    Parameters
    ----------

    num_bins : int or str
        The number of bins to make or the method to use for autobinning.

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

    all_values = []
    for contigtree in contigtrees:
        all_values.append(np.concatenate([fields[field_key]
                                          for fields
                                          in contigtree.wepy_h5.iter_trajs_fields([field_key])]))

    all_values = np.concatenate(all_values)

    bin_edges = np.histogram_bin_edges(all_values, bins=bins)

    return bin_edges


class ContigTreeProfiler(object):
    """A wrapper class around a ContigTree that provides extra methods for
    generating free energy profiles for observables."""

    def __init__(self, contigtree):
        """Create a wrapper around a contigtree for the profiler.

        Parameters
        ----------

        contigtree : ContigTree object
            The contigtree you want to generate profiles for.

        """

        self._contigtree = contigtree

        # determine the global binning strategy. These are the bins
        # that best describe the entire pool of data in this contig
        # tree


    @property
    def contigtree(self):
        """The underlying contigtree this wraps."""
        return self._contigtree

    def fe_profile(self, span, field_key, bins=None):
        """Calculate the free energy histogram over a trajectory field.

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
            'partitions' argument. The last tranche will be truncated
            to the remainder after partitioning by tranches. Overrides
            partitions.

        num_partitions : int
            Will evenly partition the time coordinate given the
            dataset. The first partition will carry the remainder.
            Overriden by time_tranche.

        Returns
        -------

        fe_profile : arraylike of dtype float
            An array of free energies for each bin.

        """

        # make the contig for this span
        contig = self.contigtree.span_contig(span)

        # get the weights
        weights = contig.contig_fields(['weights'])['weights']
        # reshape to match
        weights = weights.reshape((weights.shape[0], weights.shape[1]))
        # then get the values for the field key
        values = contig.contig_fields([field_key])[field_key]

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
            bin_edges = np.histogram_bin_edges(values, bins='auto')

        # if the number or binning strategy was given generate them
        # according to that
        elif type(bins) in (str, int):
            bin_edges = np.histogram_bin_edges(values, bins=bins)

        # if it wasn't those things we assume it is already a correct
        # bin_edges
        else:
            bin_edges = bins


        fe_profile, _ = free_energy_profile(weights, values,
                                            bins=bin_edges)

        return fe_profile


    def fe_cumulative_profiles(self, span, field_key, bins=None,
                               time_tranche=None, num_partitions=5):
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
            'partitions' argument. The last tranche will be truncated
            to the remainder after partitioning by tranches. Overrides
            partitions.

        num_partitions : int
            Will evenly partition the time coordinate given the
            dataset. The first partition will carry the remainder.
            Overriden by time_tranche.

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

        # get the weights
        weights = contig.contig_fields(['weights'])['weights']
        # reshape to match
        weights = weights.reshape((weights.shape[0], weights.shape[1]))
        # then get the values for the field key
        values = contig.contig_fields([field_key])[field_key]

        # determine the bin_edges

        # if the bins were not specified generate them from all the
        # values for this span
        if bins is None:
            bin_edges = np.histogram_bin_edges(values, bins='auto')

        # if the number or binning strategy was given generate them
        # according to that
        elif type(bins) in (str, int):
            bin_edges = np.histogram_bin_edges(values, bins=bins)

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

        return bin_edges

