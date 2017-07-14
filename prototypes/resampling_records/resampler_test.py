import numpy as np

import networkx as nx

from wepy.walker import Walker
from wepy.resampling.clone_merge import RandomCloneMergeResampler
from wepy.resampling.clone_merge import clone_parent_table, clone_parent_panel

from resampling_tree.tree import monte_carlo_minimization, make_graph

n_walkers = 50
init_weight = 1.0 / n_walkers

init_walkers = [Walker(i, init_weight) for i in range(n_walkers)]


resampler = RandomCloneMergeResampler(12312535346)

# make a template string for pretty printing results as we go
result_template_str = "|".join(["{:^10}" for i in range(n_walkers + 1)])

# print the initial walkers
print("The initial walkers:")
# slots
slot_str = result_template_str.format("slot", *[i for i in range(n_walkers)])
print(slot_str)
# states
walker_state_str = result_template_str.format("state",
    *[str(walker.state) for walker in init_walkers])
print(walker_state_str)
# weights
walker_weight_str = result_template_str.format("weight",
    *[str(walker.weight) for walker in init_walkers])
print(walker_weight_str)

# do resampling of the initial walkers
num_resamplings = 100
resampled_walkers = []
resampling_records = []
walkers = init_walkers
for i in range(num_resamplings):
    print("---------------------------------------------------------------------------------------")
    print("cycle: {}".format(i))
    # do resampling
    cycle_walkers, cycle_records = resampler.resample(walkers, debug_prints=True)

    # reset the walkers to these cycle walkers
    walkers = cycle_walkers

    # save the walkers
    resampled_walkers.append(cycle_walkers)
    # save the resampling records
    resampling_records.append(cycle_records)

    # print results for this cycle
    print("Net state of walkers after resampling:")
    print("--------------------------------------")
    # slots
    slot_str = result_template_str.format("slot", *[i for i in range(n_walkers)])
    print(slot_str)
    # states
    walker_state_str = result_template_str.format("state",
        *[str(walker.state) for walker in cycle_walkers])
    print(walker_state_str)
    # weights
    walker_weight_str = result_template_str.format("weight",
        *[str(walker.weight) for walker in cycle_walkers])
    print(walker_weight_str)


# write the output to a parent panel of all merges and clones within cycles
parent_panel = clone_parent_panel(resampling_records)

# make a table of the net parents for each cycle to a table
parent_table = np.array(clone_parent_table(resampling_records))

# write out the table to a csv
np.savetxt("parents.dat", parent_table)

# make a weights table for the walkers
weights = []
for cycle in resampled_walkers:
    cycle_weights = [walker.weight for walker in cycle]
    weights.append(cycle_weights)
weights = np.array(weights)

# create a tree visualization
# make a distance array of equal distances from scratch
distances = []
for cycle in resampled_walkers:
    d_matrix = np.ones((len(cycle), len(cycle)))
    distances.append(d_matrix)
distances = np.array(distances)

node_positions = monte_carlo_minimization(parent_table, distances, weights, 50, debug=True)
nx_graph = make_graph(parent_table, node_positions,
                          weight=weights)

nx.write_gexf(nx_graph, "random_resampler_tree.gexf")
