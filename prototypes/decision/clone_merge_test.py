import numpy as np

from wepy.walker import Walker
from wepy.resampling.decision import CloneMergeDecision
from wepy.resampling.decider.decider import RandomCloneMergeDecider

n_walkers = 50
init_weight = 1.0 / n_walkers

init_walkers = [Walker(i, init_weight) for i in range(n_walkers)]


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


