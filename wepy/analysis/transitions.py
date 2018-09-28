import itertools as it
from collections import defaultdict

import numpy as np

def transition_counts(assignments, transitions):
    """Make a dictionary of transition counts.

    assignments: a list of [N_run, [N_traj x N_cycle]] arrays of ints
    where N_runs is the number of runs, N_traj is the number of
    trajectories, and N_cycle is the number of cycles

    transitions: list of traces (a trace is a list of tuples
    specifying the run, trajectory, and frame).

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

        countsmat_d[(start_assignment, end_assignment)] += 1

    return countsmat_d

def counts_d_to_matrix(counts_d):
    # get the number of unique nodes in the counts_d
    max_assignment = max(it.chain(*counts_d.keys()))
    countsmat = np.zeros((max_assignment+1, max_assignment+1))
    for transition, n_trans in counts_d.items():
        countsmat[transition] = n_trans

    return countsmat

def normalize_counts(transition_counts_matrix):

    return np.divide(transition_counts_matrix, transition_counts_matrix.sum(axis=0))

def transition_counts_matrix(assignments, transitions):
    """Make a transition count matrix for a single run.


    assignments: a list of N_run, [N_traj x N_cycle] arrays of ints
    where N_runs is the number of runs, N_traj is the number of
    trajectories, and N_cycle is the number of cycles

    transitions: list of traces (a trace is a list of tuples
    specifying the run, trajectory, and frame).

    """

    # count them and return as a dictionary
    countsmat_d = transition_counts(assignments, transitions)

    # convert to matrix
    countsmat = counts_d_to_matrix(countsmat_d)

    return countsmat

def transition_probability_matrix(assignments, transitions):

    """
    This determines a transition matrix for a variable lag time.

    Inputs:

    assignments : (numpy array [n_traj x n_timestep]):
    This is an array that indicates the cluster number for each traj at each timestep.

    sliding_window(iterable) : list of transitions. Transitions are a
    tuple of the start and end frame for a transition. Start and end
    frames are given by (traj_idx, frame_idx).

    Outputs: trans_prob_mat (numpy array [n_cluster x n_cluster]):

    A transition probability matrix.

    """

    # get the counts
    trans_counts_mat = transition_counts_matrix(assignments, transitions)

    # normalize to get the transition probabilities
    trans_prob_mat = normalize_counts(trans_counts_mat)

    return trans_prob_mat

def run_transition_counts_matrix(wepy_hdf5, run_idx, assignment_key, transitions):
    """Make a transition counts matrix from a WepyHDF5 run for a
    particular assignment given a set of transitions.

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
    """Make a transition probability matrix from a WepyHDF5 run for a
    particular assignment given a set of transitions.

    """

    # get the counts for the run
    counts_mat = run_transition_counts_matrix(wepy_hdf5, run_idx, assignment_key, transitions)

    # normalize to get the probabilities
    trans_prob_matrix = normalize_counts(counts_mat)

    return trans_prob_matrix
