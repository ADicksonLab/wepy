class Resampler(object):
    """Superclass for resamplers that use the the Novelty->Decider
    framework."""
    def __init__(self, novelty, decider):
        self.novelty = novelty
        self.decider = decider

    def resample(self, walkers):

        resampled_walkers = walkers
        resampling_actions = []
        aux_data = {}
        finished = False
        step_count = 0
        while not finished:
            novelties, novelty_aux = self.novelty.novelties(walkers)
            finished, decisions, decider_aux = self.decider.decide(novelties)
            resampled_walkers = self.decider.decision.action(walkers, decisions)
            resampling_actions.append(decisions)
            aux_data.update([novelty_aux, decider_aux])

            step_count += 1

        return resampled_walkers, resampling_actions, aux_data

    def actions_from_list(self, walkers_squashed,num_clones):
        n_walkers = len(num_clones)
        
        # determine resampling actions
        walker_actions = [self.DECISION.record(self.DECISION.ENUM.NOTHING.value, (i,))
                    for i in range(n_walkers)]

        free_walkers = []
        for w in range(n_walkers):
            if num_clones[w] > 0 and len(walkers_squashed[w]) > 0:
                raise ValueError("Error! cloning and merging occuring with the same walker")

            # add walkers squashed by w onto the free_walkers list
            if len(walkers_squashed[w]) > 0:
                free_walkers += walkers_squashed[w]
                for squash_idx in walkers_squashed[w]:
                    walker_actions[squash_idx] = self.DECISION.record(self.DECISION.ENUM.SQUASH.value,(w,))
                walker_actions[w] = self.DECISION.record(self.DECISION.ENUM.KEEP_MERGE.value,(w,))
            if num_clones[w] > 0:
                slots = []
                for i in range(num_clones[w]):
                    slots.append(free_walkers.pop())
                walker_actions[w] = self.DECISION.record(self.DECISION.ENUM.CLONE.value,tuple(slots))
        return walker_actions
