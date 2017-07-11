from wepy.walker import merge

class Resampler(object):

    def resample(self, walkers, decisions):
        raise NotImplementedError

class StubResampler(Resampler):

    def resample(self, walkers, decisions):
        return walkers

# TODO
class SimpleCloneMerge(Resampler):
    """Just clones and merges, don't rely on any specific method."""

    def resample(self, walkers, decisions):
        to_clone = []
        to_merge = []
        for walker_idx, decision in enumerate(decisions):
            if decision is Decision.CLONE:
                to_clones.append(walker_idx)
            elif decision is Decision.MERGE:
                to_merge.append(walker_idx)
            else:
                pass

        # make sure that the number of clones is equal to twice the
        # number of merges so we can keep the number of walkers the same
        assert len(to_clone) == 2*len(to_merge), \
            "Number of clones and merges does not match"

        resampling_records = []
        new_walkers = []
        walker_idx = 0
        while True:
            if walker_idx in to_clone:
                parent_walker_idx = walker_idx
                # get the parent walker
                parent_walker = next(walkers)
                # clone it
                child_walkers = parent_walker.clone()
                # put one in the place that this parent is in
                new_walkers.append(child_walkers.pop())
                # then search for the next merged one


            # get two walkers to merge to make room for the clone and
            # merge them
            merge_idxs = [to_merge.pop(), to_merge.pop()]
            walkers = [walkers[i] for i in merge_idxs]
            merged_walker, keep_walker_idx = merge(walker_b)
            new_walkers.append(merged_walker)

            # create a (cloned, squashed_0,...,squashed_k, merged) tuple
            resampling_record = (parent_walker_idx, *squashed_walkers,
                                 keep_walker_idx)
            resampling_records.append(resampling_record)

        return new_walkers, resampling_records


class WExplore(Resampler):
    pass

class WExplore2(Resampler):
    # I lied Nazanin put your code here!!
    pass
