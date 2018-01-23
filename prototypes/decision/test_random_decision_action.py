import numpy as np

from wepy.walker import Walker
from wepy.resampling.decision import CloneMergeDecision

DECISION = CloneMergeDecision
VAL_NAME_D = CloneMergeDecision.enum_dict_by_value()

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
    n_walkers = 6
    init_weight = 1.0 / n_walkers
    init_walkers = [Walker(i, init_weight) for i in range(n_walkers)]



    enum = CloneMergeDecision.ENUM
    inst_recs = dict(CloneMergeDecision.INSTRUCTION_RECORDS)

    # TEST 1
    print("test {}".format(1))
    print_walkers(init_walkers)
    # make some mock decisions of clones and merges
    decisions = [(enum.CLONE.value, inst_recs[enum.CLONE](0, 1)),
                      (enum.SQUASH.value, inst_recs[enum.SQUASH](2)),
                      (enum.KEEP_MERGE.value, inst_recs[enum.KEEP_MERGE](2)),
                      (enum.NOTHING.value, inst_recs[enum.NOTHING](3)),
                      (enum.NOTHING.value, inst_recs[enum.NOTHING](4)),
                      (enum.NOTHING.value, inst_recs[enum.NOTHING](5)),
    ]

    print_decisions(decisions)


    new_walkers = CloneMergeDecision.action(init_walkers, decisions)

    print_walkers(new_walkers)
    test_probs(new_walkers)

    # TEST 2
    print("test {}".format(2))
    print_walkers(init_walkers)
    decisions = [(enum.CLONE.value, inst_recs[enum.CLONE](0, 1)),
                      (enum.SQUASH.value, inst_recs[enum.SQUASH](2)),
                      (enum.KEEP_MERGE.value, inst_recs[enum.KEEP_MERGE](2)),
                      (enum.CLONE.value, inst_recs[enum.CLONE](3, 4)),
                      (enum.SQUASH.value, inst_recs[enum.SQUASH](5)),
                      (enum.KEEP_MERGE.value, inst_recs[enum.KEEP_MERGE](5)),

    ]

    print_decisions(decisions)


    new_walkers = CloneMergeDecision.action(init_walkers, decisions)

    print_walkers(new_walkers)
    test_probs(new_walkers)


    # TEST 3
    print("test {}".format(3))
    print_walkers(init_walkers)
    decisions = [(enum.CLONE.value, inst_recs[enum.CLONE](5, 1)),
                      (enum.SQUASH.value, inst_recs[enum.SQUASH](3)),
                      (enum.KEEP_MERGE.value, inst_recs[enum.KEEP_MERGE](3)),
                      (enum.NOTHING.value, inst_recs[enum.NOTHING](2)),
                      (enum.NOTHING.value, inst_recs[enum.NOTHING](0)),
                      (enum.NOTHING.value, inst_recs[enum.NOTHING](4)),

    ]

    print_decisions(decisions)


    new_walkers = CloneMergeDecision.action(init_walkers, decisions)

    print_walkers(new_walkers)
    test_probs(new_walkers)

    # TEST 3
    print("test {}".format(3))
    print_walkers(init_walkers)
    decisions = [(enum.CLONE.value, inst_recs[enum.CLONE](0, 1)),
                      (enum.NOTHING.value, inst_recs[enum.NOTHING](1)),
                      (enum.CLONE.value, inst_recs[enum.CLONE](2, 3)),
                      (enum.KEEP_MERGE.value, inst_recs[enum.KEEP_MERGE](3)),
                      (enum.SQUASH.value, inst_recs[enum.SQUASH](3)),
                      (enum.NOTHING.value, inst_recs[enum.NOTHING](5)),

    ]

    print_decisions(decisions)


    new_walkers = CloneMergeDecision.action(init_walkers, decisions)

    print_walkers(new_walkers)
    test_probs(new_walkers)
