"""Collection of routines for counting transitions and calculating
transition probability matrices.

There are two basic data structures used in this module for edge
transitions: dictionary based and matrix based.

The dictionary based approach is used to interface with graph
representations, like in the
~wepy.analysis.network.MacroStateNetwork~.

Arrays are used as an interface to numerical libraries which use this
to calculate other properties such as steady-state probabilities.

Wepy is primarily meant to support expedient data structuring and will
not support the latter numerical calculations and is left up to
external libraries and domain specific code.

Routines
--------

transition_counts : Given a structured array of assignments of
    microstate to macrostate labels and a listing of all kinetic
    transitions over the microstates returns the counts of transitions
    between each macrostate as a dictionary mapping macrostate
    transition edges to the total counts each microstate transition.

counts_d_to_matrix : Converts a dictionary mapping macrostate
    transition edges to a asymmetric transition matrix.

normalize_counts : Normalizes outgoing transition counts from each
    macrostate to 1 for a transition counts matrix.

transition_counts_matrix : Given a structured array of assignments of
    microstate to macrostate labels and a listing of all kinetic
    transitions over the microstates returns the counts of transitions
    between each macrostate as an assymetric transition counts matrix.

transition_probability_matrix : Given a structured array of assignments of
    microstate to macrostate labels and a listing of all kinetic
    transitions over the microstates returns the probability of transitions
    between each macrostate as an assymetric transition probability matrix.

run_transition_counts_matrix : Generates an asymmetric transition
    counts matrix directly from a single WepyHDF5 run.

run_transition_probability_matrix : Generates an asymmetric transition
    probability matrix directly from a single WepyHDF5 run.
"""

import itertools as it
from collections import defaultdict

import numpy as np

def transition_counts(
        assignments,
        transitions,
        weights=None):
    """Make a dictionary of the count of microstate transitions between macrostates.

    If weights are given counts are the weight instead of 1.

    Parameters
    ----------

    assignments: mixed array_like of dim (n_run, n_traj, n_cycle) type int
        Assignment of microstates to macrostate labels, where N_runs
        is the number of runs, N_traj is the number of trajectories,
        and N_cycle is the number of cycles.

    transitions: list of list of tuples of ints (run_idx, traj_idx, cycle_idx)
        List of run traces corresponding to transitions. Only the
        first and last trace elements are used.

    weights: mixed array_like of dim (n_run, n_traj, n_cycle) type float or None
        Same shape as the assignments, but gives the weight of the walker at this time.
          (Optional)

    Returns
    -------
    transition_counts : dict of (int, int): int
        Dictionary mapping transitions between macrostate labels to
        the count of microstate transitions between them. If weights
        are given this is the weighted counts.

    """

    # for each transition (start, end) (don't permute) and count them
    # up in a dictionary
    countsmat_d = defaultdict(int)
    for transition in transitions:

        start = transition[0]
        end = transition[-1]

        # get the assignments for the transition
        start_assignment = assignments[start[0]][start[1]][start[2]]
        end_assignment = assignments[end[0]][end[1]][end[2]]


        if weights is not None:
            # get the weight for this walker
            weight = weights[start[0]][start[1]][start[2]]

        else:
            weight = 1.0

        countsmat_d[(start_assignment, end_assignment)] += weight

    return countsmat_d

def counts_d_to_matrix(counts_d):
    """Convert a dictionary of counts for macrostate transitions to an
    assymetric transitions counts matrix.

    Parameters
    ----------
    counts_d : dict of (int, int): int
        Dictionary mapping transitions between macrostate labels to
        the count of microstate transitions between them.

    Returns
    -------
    counts_matrix : numpy.ndarray of int
        Assymetric counts matrix of dim (n_macrostates, n_macrostates).

    """
    # get the number of unique nodes in the counts_d
    max_assignment = max(it.chain(*counts_d.keys()))
    countsmat = np.zeros((max_assignment+1, max_assignment+1))
    for transition, n_trans in counts_d.items():
        countsmat[transition] = n_trans

    return countsmat

def normalize_counts(transition_counts_matrix):
    """Normalize the macrostate outgoing transition counts to 1.0 for each macrostate.

    Parameters
    ----------
    transition_counts_matrix : numpy.ndarray of int
        Assymetric counts matrix of dim (n_macrostates, n_macrostates).

    Returns
    -------
    transition_probability_matrix : numpy.ndarray of float
        Assymetric transition probability matrix of dim (n_macrostates, n_macrostates).

    """

    # if there are any columns that sum to zero we need to set the
    # diagonal value to 1.
    col_sums = transition_counts_matrix.sum(axis=0)

    zero_cols = np.where(col_sums == 0.0)[0]

    if len(zero_cols) > 0:
        transition_counts_matrix = transition_counts_matrix.copy()

        for zero_col_idx in zero_cols:
            transition_counts_matrix[zero_col_idx, zero_col_idx] = 1.0

    return np.divide(transition_counts_matrix, transition_counts_matrix.sum(axis=0))

def transition_counts_matrix(assignments, transitions):
    """Make an asymmetric array of the count of microstate transitions between macrostates.

    Parameters
    ----------
    assignments: mixed array_like of dim (n_run, n_traj, n_cycle) type int
        Assignment of microstates to macrostate labels, where N_runs
        is the number of runs, N_traj is the number of trajectories,
        and N_cycle is the number of cycles.
    transitions: list of list of tuples of ints (run_idx, traj_idx, cycle_idx)
        List of run traces corresponding to transitions. Only the
        first and last trace elements are used.

    Returns
    -------
    transition_counts_matrix : numpy.ndarray of int
        Assymetric counts matrix of dim (n_macrostates, n_macrostates).
    """

    # count them and return as a dictionary
    countsmat_d = transition_counts(assignments, transitions)

    # convert to matrix
    countsmat = counts_d_to_matrix(countsmat_d)

    return countsmat

def transition_probability_matrix(assignments, transitions):
    """Make an asymmetric array of macrostates transition probabilities from microstate assignments.

    Parameters
    ----------
    assignments: mixed array_like of dim (n_run, n_traj, n_cycle) type int
        Assignment of microstates to macrostate labels, where N_runs
        is the number of runs, N_traj is the number of trajectories,
        and N_cycle is the number of cycles.
    transitions: list of list of tuples of ints (run_idx, traj_idx, cycle_idx)
        List of run traces corresponding to transitions. Only the
        first and last trace elements are used.

    Returns
    -------
    transition_probability_matrix : numpy.ndarray of float
        Assymetric transition probability matrix of dim (n_macrostates, n_macrostates).
    """

    # get the counts
    trans_counts_mat = transition_counts_matrix(assignments, transitions)

    # normalize to get the transition probabilities
    trans_prob_mat = normalize_counts(trans_counts_mat)

    return trans_prob_mat

def run_transition_counts_matrix(wepy_hdf5, run_idx, assignment_key, transitions):
    """Generates an asymmetric transition counts matrix directly from a single WepyHDF5 run.

    Parameters
    ----------
    wepy_hdf5 : WepyHDF5 object

    run_idx : int

    assignment_key : str
        Field from trajectory data to use as macrostate label.
    transitions : list of list of tuples of ints (run_idx, traj_idx, cycle_idx)
        List of run traces corresponding to transitions. Only the
        first and last trace elements are used.

    Returns
    -------
    transition_counts_matrix : numpy.ndarray of int
        Assymetric counts matrix of dim (n_macrostates, n_macrostates).

    """


    total_counts_d = defaultdict(int)

    max_assignment = 0
    for transition in transitions:

        start = transition[0]
        end = transition[-1]

        # Gets cluster pair from the hdf5 file
        assignments = wepy_hdf5.get_run_trace_fields(run_idx, [start, end],
                                                  [assignment_key])[assignment_key]

        # Add a count to the cluster pair in the dictionary
        total_counts_d[(assignments[0], assignments[1])] += 1

        # If the assignment index is higher than previously seen
        # assignments, update the max_assignment
        max_assg = max(assignments)
        if max_assg > max_assignment:
            max_assignment = max_assg

    # make a matrix of the counts
    counts_matrix = np.zeros((max_assignment+1, max_assignment+1))
    for transition, n_trans in total_counts_d.items():
            counts_matrix[transition] = n_trans

    return counts_matrix

def run_transition_probability_matrix(wepy_hdf5, run_idx, assignment_key, transitions):
    """Generates an asymmetric transition counts matrix directly from a single WepyHDF5 run.

    Parameters
    ----------
    wepy_hdf5 : WepyHDF5 object

    run_idx : int

    assignment_key : str
        Field from trajectory data to use as macrostate label.
    transitions : list of list of tuples of ints (run_idx, traj_idx, cycle_idx)
        List of run traces corresponding to transitions. Only the
        first and last trace elements are used.

    Returns
    -------
    transition_probability_matrix : numpy.ndarray of float
        Assymetric transition probability matrix of dim (n_macrostates, n_macrostates).

    """

    # get the counts for the run
    counts_mat = run_transition_counts_matrix(wepy_hdf5, run_idx, assignment_key, transitions)

    # normalize to get the probabilities
    trans_prob_matrix = normalize_counts(counts_mat)

    return trans_prob_matrix
