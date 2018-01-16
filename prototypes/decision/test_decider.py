import numpy as np

from wepy.walker import Walker
from wepy.resampling.decider.decider import RandomCloneMergeDecider


DECISION = RandomCloneMergeDecider.DECISION
VAL_NAME_D = DECISION.enum_dict_by_value()

def print_walkers(walkers):

    n_walkers = len(walkers)
    # make a template string for pretty printing results as we go
    int_result_template_str = "|".join(["{:^10}" for i in range(n_walkers + 1)])
    float_result_template_str = "|".join(["{:^10.8}" for i in range(n_walkers + 1)])

    # print a dividing line
    print("-" * 13 * n_walkers)

    # slots
    # slot_str = int_result_template_str.format("slot", *[i for i in range(n_walkers)])
    # print(slot_str)
    # states
    walker_state_str = int_result_template_str.format("state",
        *[str(walker.state) for walker in walkers])
    print(walker_state_str)
    # weights
    walker_weight_str = float_result_template_str.format("weight",
        *[str(walker.weight) for walker in walkers])
    print(walker_weight_str)

    # print a dividing line
    print("-" * 13 * n_walkers)

def print_decisions(decisions):
    n_decisions = len(decisions)
    # make a template string for pretty printing results as we go
    int_result_template_str = "|".join(["{:^10}" for i in range(n_decisions + 1)])
    float_result_template_str = "|".join(["{:^10.8}" for i in range(n_decisions + 1)])

    # print a dividing line
    print("-" * 13 * n_decisions)

    # decisions
    decision_name_str = int_result_template_str.format("enum",
        *[str(VAL_NAME_D[decision].name) for decision, instruct in decisions])
    print(decision_name_str)
    # intstructions
    instruct_str = float_result_template_str.format("instruct",
        *[str(instruct[:]) for decision, instruct in decisions])
    print(instruct_str)

    # print a dividing line
    print("-" * 13 * n_decisions)

def test_probs(walkers):

    probs = [walker.weight for walker in walkers]

    # make sure they sum to 1.0
    assert np.isclose(np.sum(probs), 1.0), "weights do not sum to 1.0"

if __name__ == "__main__":
    n_walkers = 7
    init_weight = 1.0 / n_walkers
    init_walkers = [Walker(i, init_weight) for i in range(n_walkers)]


    novelties = list(range(n_walkers))
    no_error = True
    seed = 0
    while no_error:

        print("Seed: {}".format(seed))
        decider = RandomCloneMergeDecider(seed=seed)

        try:
            decisions, aux_data = decider.decide(novelties)
        except ValueError:
            no_error = False
            print("ERROR")
        else:
            print_decisions(decisions)
            seed += 1

    import ipdb; ipdb.set_trace()
    decisions, aux_data = decider.decide(novelties)
