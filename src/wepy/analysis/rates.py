"""Compute rates for warping events."""

from collections import defaultdict

def calc_warp_rate(warping_records, total_sampling_time):
    """Calculate the warping rates for each target in these records.

    Takes a set of warping records which are named tuples that have
    the fields set in the Abstract Base Class for boundary conditions
    (wepy.boundary_conditions.boundary.Boundary).

    Namely these are: weight, target_idx, cycle_idx

    Parameters
    ----------

    warping_records : list of namedtuples implementing the warping interface
        The list of warping records for which events will be used to
        calculate the rates.

    total_sampling_time : float
        The total amount of sampling time that was run to generate
        these warping records.

    Returns
    -------

    target_weights_rates : dict of int : tuple of (float, float, float)
        A dictionary where each key is for a target present in the
        warping records and each value is a tuple giving the total
        accumulated weight in that target boundary, the total flux of
        probability across the boundary (rate), and the total sampling
        time it was calculated with,respectively.

    See Also
    --------

    wepy.boundary_conditions.boundary.Boundary : for specs on fields
                                                 necessary for warping_records

    """

    # we wish we could return a tuple of (0., np.inf) for no records
    # but we don't know the target indices so there isn't really a way
    # to do this without defaults which will only wreak havoc I am
    # sure. So this must be dealt with outside of this function.

    # get the weights from the records, the fields for the records are
    # defined in the base class for boundary conditions

    # we separate them based on the target
    target_warp_weights = defaultdict(list)
    for rec in warping_records:

        # get the weight
        target_warp_weights[rec.target_idx].append(rec.weight)

    # sum each target weights up, and compute the weights
    target_results = {}
    for target_key, weights in target_warp_weights.items():

        total_weight = sum(weights)
        rate = total_weight / total_sampling_time

        target_results[target_key] = (total_weight, rate, total_sampling_time)

    return target_results


def contig_warp_rates(contig, cycle_time, time_points=None):
    """Calculate the warping rates for a contig.

    Automatically figures out the total sampling time for two options
    which is set by the 'time_points' kwarg.

    If 'time_points' is None then a rate will only be computed for the
    end of the simulation.

    If 'time_points' is Ellipsis then a rate will be computed for each
    cycle in the entire run.

    Will work with runs that used boundary conditions deriving from
    and preserving the warping record fields from the abstract base
    class (wepy.boundary_conditions.boundary.Boundary).

    Parameters
    ----------

    contig : analysis.contig_tree.Contig
        Underlying WepyHDF5 must be open for reading.

    cycle_time : float
        The sampling time for each cycle.

    time_points : None or Ellipsis
        Determines how many and which time points are computed for the
        rates of each run. None, only the end; Ellipsis, all cycles.

    Returns
    -------

    run_target_weights_rates : list of dict of int : tuple of (float, float, float)

        List where each value of a run is a list of outputs from
        calc_warping_rate. The number and meaning of the elements of
        the list depend on the value of the 'time_points' kwarg.

    See Also
    --------

    wepy.analysis.rates.calc_warp_rate

    wepy.boundary_conditions.boundary.Boundary : for specs on fields
                                                 necessary for warping_records

    """

    # if time_points is None that means just get the rate for the
    # whole run or Ellipsis to evaluate it for all cycles.
    if time_points is None:
        n_cycle_points = [contig.num_cycles]
        cycle_idxs = [contig.num_cycles - 1]

    elif time_points is Ellipsis:
        n_cycle_points = [i + 1 for i in range(contig.num_cycles)]
        cycle_idxs = list(range(contig.num_cycles))

    # get a vector for the number of walkers for each cycle_idx
    n_walkers = [contig.num_walkers(cycle_idx) for cycle_idx in cycle_idxs]

    # calculate the total sampling time for each point which is the
    # cycle_index times the real sampling time for each walker in the
    # ensemble times the number of walkers
    total_sampling_times = [n_cycle * cycle_time * n_walkers[i]
                            for i, n_cycle in enumerate(n_cycle_points)]

    all_recs = contig.warping_records()

    contig_rates = []
    for cycle_idx, total_sampling_time in zip(cycle_idxs, total_sampling_times):
        recs = [rec for rec in all_recs if rec.cycle_idx <= cycle_idx]
        rate = calc_warp_rate(recs, total_sampling_time)
        contig_rates.append(rate)

    return contig_rates
