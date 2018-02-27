import numpy as np


def transmat(cluster_array, sliding_window):

	"""
	This determines a transition matrix for a variable lag time.

	Inputs:

	cluster_array (numpy array [n_timestep x n_walker numpy array]): 
	This is an array that indicates the cluster number for each walker at each timestep.

	sliding_window(iterable): Contains several lists of length of the lag time.
	Each list contains (2 x 1) tuples indicating the cycle and walker parents for a single
	cycle of WExplore.  

	Outputs: transition_matrix (numpy array [n_cluster x n_cluster]):

	A transition matrix based upon the lag time.

	"""

	# Obtains the number of clusters.
	n_cluster = np.max(cluster_array)

	# Allocates memory for the transition matrix
	transition_matrix = np.zeros([n_cluster + 1, n_cluster + 1])

	for step in windows:

	    # Determines the starting and ending walker
	    start_time = item[0][0]
	    original_walker = item[0][1]
	    end_time = item[-1][0]
	    final_walker = item[-1][1]

	    # Determines the cluster number of the begining walker
	    # and end walker.

	    start_cluster = cluster_array[start_time][original_walker]
	    end_cluster = cluster_array[end_time][final_walker]

	    # Adds a counter to the transition matrix
	    transition_matrix[star_cluster][end_cluster] += 1

	# Transforms transition matrix to show probabilities

	for cluster in range(n_cluster):
	    total = np.sum(transition_matrix[cluster]
	    transition_matrix[cluster] /= total
	
	return(transition_matrix)

