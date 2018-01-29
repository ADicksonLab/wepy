from wepy.resampling.decisions.clone_merge import CloneMergeDecisionEnum

from wepy.hdf5 import RESAMPLING_RECORDS_FIELDS

ANCESTOR_DECISION_IDS = (CloneMergeDecisionEnum.NOTHING.value,
                         CloneMergeDecisionEnum.KEEP_MERGE.value,
                         CloneMergeDecisionEnum.CLONE.value,)

def clone_parent_panel(resampling_records):

    parent_panel = []

    # initialize the first cycle as the initial walkers, in a single
    # stage
    init_parent_table = [ [i for i in range(len(cycle_stages[0]))] ]
    parent_panel.append(parent_panel)

    # each cycle after the first cycle
    for record_idx, resampling_record in enumerate(resampling_records):

        # each stage in the resampling for that cycle
        # make a stage parent table
        cycle_parent_table = []

        # now iterate through the rest of the stages
        for stage_idx, stage in enumerate(cycle_stages):

            # initialize a list for the parents of this stages walkers
            stage_parents = [None for i in range(len(stage))]

            # the rest of the stages parents are based on the previous stage
            for parent_idx, resampling_record in enumerate(stage):

                # get the field values by their field name
                rec_map = {name : resampling_record[i] for i, name in
                           enumerate(RESAMPLING_RECORDS_FIELDS)}

                # the stage parents for the next stage, initialize to
                # the same idx
                # stage_parents = [i for i in range(len(stage))]

                # if the decision for this resampling record is an
                # ancestor decision type we want to write the parents
                # for the next stage
                
                # if the parent is NOTHING or KEEP_MERGE it will have an index
                if any([rec_map['decision_id'] ==  in [CloneMergeDecisionEnum.NOTHING.value,
                                                  CloneMergeDecisionEnum.KEEP_MERGE.value]]):
                    # single child
                    child_idx = resampling_record.instruction
                    # set the parent in the next row to this parent_idx
                    stage_parents[child_idx] = parent_idx
                # if it is CLONE it will have 2 indices
                elif resampling_record.decision == CloneMergeDecisionEnum.CLONE.value:
                    children = resampling_record.instruction[:]
                    for child_idx in children:
                        stage_parents[child_idx] = parent_idx
                elif resampling_record.decision == CloneMergeDecisionEnum.SQUASH.value:
                    # do nothing this one has no children
                    pass
                else:
                    raise TypeError("Decision type not recognized")

                # for the full stage table save all the intermediate parents
                stage_parent_table.append(stage_parents)

        # for the full parent panel
        full_parent_panel.append(stage_parent_table)

    return full_parent_panel

def clone_parent_table(clone_merge_resampling_record):
    # all of the parent tables including within cycle stage parents,
    # this is 3D so I call it a panel
    net_parent_table = [[i for i in range(len(clone_merge_resampling_record[0][0]))]]

    # each cycle
    for cycle_idx, cycle_stages in enumerate(clone_merge_resampling_record):
        # each stage in the resampling for that cycle
        # make a stage parent table
        stage_parent_table = []
        for stage_idx, stage in enumerate(cycle_stages):

            # the initial parents are just their own indices,
            # this accounts for when there is no resampling done
            stage_parents = [i for i in range(len(stage))]
            # add it to the stage parents matrix
            stage_parent_table.append(stage_parents)

            # the rest of the stages parents are based on the previous stage
            for parent_idx, resampling_record in enumerate(stage):
                # the stage parents for the next stage, initialize to
                # the same idx
                stage_parents = [i for i in range(len(stage))]
                # if the parent is NOTHING or KEEP_MERGE it will have an index
                if resampling_record.decision in [CloneMergeDecisionEnum.NOTHING,
                                                  CloneMergeDecisionEnum.KEEP_MERGE]:
                    # single child
                    child_idx = resampling_record.instruction
                    # set the parent in the next row to this parent_idx
                    stage_parents[child_idx] = parent_idx
                # if it is CLONE it will have 2 indices
                elif resampling_record.decision is CloneMergeDecisionEnum.CLONE:
                    children = resampling_record.instruction[:]
                    for child_idx in children:
                        stage_parents[child_idx] = parent_idx
                elif resampling_record.decision  is CloneMergeDecisionEnum.SQUASH:
                    # do nothing this one has no children
                    pass
                else:
                    raise TypeError("Decision type not recognized")

                # for the full stage table save all the intermediate parents
                stage_parent_table.append(stage_parents)

            # for the net table we only want the end results,
            # we start at the last cycle and look at its parent
            stage_net_parents = []
            n_stages = len(stage_parent_table)
            for walker_idx, parent_idx in enumerate(stage_parent_table[-1]):
                # initialize the root_parent_idx which will be updated
                root_parent_idx = parent_idx

                # if no resampling skip the loop and just return the idx
                if n_stages > 0:
                    # go back through the stages getting the parent at each stage
                    for prev_stage_idx in range(n_stages):
                        prev_stage_parents = stage_parent_table[-(prev_stage_idx+1)]
                        root_parent_idx = prev_stage_parents[root_parent_idx]

                # when this is done we should have the index of the root parent,
                # save this as the net parent index
                stage_net_parents.append(root_parent_idx)

        # for this stage save the net parents
        net_parent_table.append(stage_net_parents)

    return net_parent_table
