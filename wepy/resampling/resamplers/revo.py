import multiprocessing as mulproc
import random as rand
import itertools as it
import logging

import numpy as np

from wepy.resampling.resamplers.resampler import Resampler
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision

class REVOResampler(Resampler):
    """ """

    DECISION = MultiCloneMergeDecision

    # state change data for the resampler
    RESAMPLER_FIELDS = ('n_walkers', 'distance_matrix', 'spread', 'image_shape', 'images')
    RESAMPLER_SHAPES = ((1,), Ellipsis, (1,), Ellipsis, Ellipsis)
    RESAMPLER_DTYPES = (np.int, np.float, np.float, np.int, None)

    # fields that can be used for a table like representation
    RESAMPLER_RECORD_FIELDS = ('spread',)

    # fields for resampling data
    RESAMPLING_FIELDS = DECISION.FIELDS + ('step_idx', 'walker_idx',)
    RESAMPLING_SHAPES = DECISION.SHAPES + ((1,), (1,),)
    RESAMPLING_DTYPES = DECISION.DTYPES + (np.int, np.int,)

    # fields that can be used for a table like representation
    RESAMPLING_RECORD_FIELDS = DECISION.RECORD_FIELDS + ('step_idx', 'walker_idx',)


    def __init__(self, seed=None, pmin=1e-12, pmax=0.1, dpower=4, merge_dist=2.5,
                 d0=None, distance=None, init_state=None, weights=True):

        self.decision = self.DECISION

        # the minimum probability for a walker
        self.pmin=pmin
        # ln(probability_min)
        self.lpmin = np.log(pmin/100)

        # maximum probability for a walker
        self.pmax=pmax

        #
        self.dpower = dpower

        #
        self.merge_dist = merge_dist

        # the distance metric
        assert distance is not None, "Must give a distance metric class"
        self.distance = distance

        # the characteristic distance, d0
        assert d0 is not None, "Must give a d0 value (characteristic distance)"
        self.d0 = d0

        # setting the random seed
        self.seed = seed
        if seed is not None:
            rand.seed(seed)

        # setting the weights parameter
        self.weights = weights

        # we do not know the shape and dtype of the images until
        # runtime so we determine them here
        assert init_state is not None, "must give an initial state to infer data about the image"
        image = self.distance.image(init_state)
        self.image_dtype = image.dtype

    # we need this to on the fly find out what the datatype of the
    # image is
    def resampler_field_dtypes(self):
        """ """

        # index of the image idx
        image_idx = self.resampler_field_names().index('images')

        # dtypes adding the image dtype
        dtypes = list(super().resampler_field_dtypes())
        dtypes[image_idx] = self.image_dtype

        return tuple(dtypes)

    def _calcspread(self, walkerwt, amp, distance_matrix):
        """

        Parameters
        ----------
        walkerwt :
            
        amp :
            
        distance_matrix :
            

        Returns
        -------

        """

        n_walkers = len(walkerwt)
        # the value to be optimized
        spread = 0

        #
        wsum = np.zeros(n_walkers)

        # weight factors for the walkers
        wtfac = np.zeros(n_walkers)

        # set the weight factors
        for i in range(n_walkers):

            if walkerwt[i] > 0 and amp[i] > 0:
                if self.weights:
                    wtfac[i] = np.log(walkerwt[i]/amp[i]) - self.lpmin
                else:
                    wtfac[i] = 1

            else:
                wtfac[i] = 0

            if wtfac[i] < 0:
                wtfac[i] = 0

        #
        for i in range(n_walkers - 1):
            if amp[i] > 0:
                for j in range(i+1, n_walkers):
                    if amp[j] > 0:
                        d = ((distance_matrix[i][j]/self.d0)**self.dpower) * wtfac[i] * wtfac[j]
                        spread += d * amp[i] * amp[j]
                        wsum[i] += d * amp[j]
                        wsum[j] += d * amp[i]

        # another implementation for personal clarity
        # for i, j in it.combinations(range(len(n_walkers)), 2):
        #     if amp[i] > 0 and amp[j] > 0:
        #         d = ((distance_matrix[i][j])**self.dpower) * wtfac[i] * wtfac[j]
        #         spread += d * amp[i] * amp[j]
        #         wsum[i] = += d * amp[j]
        #         wsum[j] += d * amp[i]

        return spread, wsum

    def decide_clone_merge(self, walkerwt, amp, distance_matrix):
        """

        Parameters
        ----------
        walkerwt :
            
        amp :
            
        distance_matrix :
            

        Returns
        -------

        """

        n_walkers = len(walkerwt)

        spreads = []
        merge_groups = [[] for i in range(n_walkers)]
        walker_clone_nums = [0 for i in range(n_walkers)]

        new_wt = walkerwt.copy()
        new_amp = amp.copy()
        # initialize the actions to nothing, will be overwritten

        # calculate the initial spread which will be optimized
        spread, wsum = self._calcspread(walkerwt, new_amp, distance_matrix)
        spreads.append(spread)

        # maximize the variance through cloning and merging
        logging.info("Starting variance optimization:", spread)

        productive = True
        while productive:
            productive = False
            # find min and max wsums, alter new_amp

            # initialize to None, we may not find one of each
            minwind = None
            maxwind = None

            # selects a walker with minimum wsum and a walker with
            # maximum wsum walker (distance to other walkers) will be
            # tagged for cloning (stored in maxwind), except if it is
            # already a keep merge target
            max_tups = []
            for i, value in enumerate(wsum):
                # 1. must have an amp >=1 which gives the number of clones to be made of it
                # 2. clones for the given amplitude must not be smaller than the minimum probability
                # 3. must not already be a keep merge target
                if (new_amp[i] >= 1) and \
                   (new_wt[i]/(new_amp[i] + 1) > self.pmin) and \
                   (len(merge_groups[i]) == 0):
                    max_tups.append((value, i))


            if len(max_tups) > 0:
                maxvalue, maxwind = max(max_tups)

            # walker with the lowest wsum (distance to other walkers)
            # will be tagged for merging (stored in minwind)
            min_tups = [(value, i) for i,value in enumerate(wsum)
                        if new_amp[i] == 1 and (new_wt[i]  < self.pmax)]

            if len(min_tups) > 0:
                minvalue, minwind = min(min_tups)

            # does minwind have an eligible merging partner?
            # closedist = self.merge_dist
            closewalk = None
            condition_list = np.array([i is not None for i in [minwind, maxwind]])
            if condition_list.all() and minwind != maxwind:

                # get the walkers that aren't the minimum and the max
                # wsum walkers, as candidates for merging
                closewalks = set(range(n_walkers)).difference([minwind, maxwind])

                # remove those walkers that if they were merged with
                # the min wsum walker would violate the pmax
                closewalks = [idx for idx in closewalks
                                      if (new_amp[idx]==1) and
                                       (new_wt[idx] + new_wt[minwind] < self.pmax)
                                      ]

                # if there are any walkers left, get the distances of
                # the close walkers to the min wsum walker if that
                # distance is less than the maximum merge distance
                if len(closewalks) > 0:
                    closewalks_dists = [(distance_matrix[minwind][i], i) for i in closewalks
                                            if distance_matrix[minwind][i] < (self.merge_dist)]

                    # if any were found set this as the closewalk
                    if len(closewalks_dists) > 0:
                        closedist, closewalk = min(closewalks_dists)


            # did we find a closewalk?
            condition_list = np.array([i is not None for i in [minwind, maxwind, closewalk]])
            if condition_list.all() :

                # change new_amp
                tempsum = new_wt[minwind] + new_wt[closewalk]
                new_amp[minwind] = new_wt[minwind]/tempsum
                new_amp[closewalk] = new_wt[closewalk]/tempsum
                new_amp[maxwind] += 1

                # re-determine spread function, and wsum values
                newspread, wsum = self._calcspread(new_wt, new_amp, distance_matrix)

                if newspread > spread:
                    spreads.append(newspread)

                    logging.info("Variance move to", newspread, "accepted")

                    productive = True
                    spread = newspread

                    # make a decision on which walker to keep
                    # (minwind, or closewalk), equivalent to:
                    # `random.choices([closewalk, minwind],
                    #                 weights=[new_wt[closewalk], new_wt[minwind])`
                    r = rand.uniform(0.0, new_wt[closewalk] + new_wt[minwind])

                     # keeps closewalk and gets rid of minwind
                    if r < new_wt[closewalk]:
                        keep_idx = closewalk
                        squash_idx = minwind

                    # keep minwind, get rid of closewalk
                    else:
                        keep_idx = minwind
                        squash_idx = closewalk

                    # if keep_idx == maxwind:
                    #     import ipdb; ipdb.set_trace()

                    # if len(merge_groups[maxwind]) > 0:
                    #     import ipdb; ipdb.set_trace()
                    #     print("Attempting to clone a walker which is a keep idx of a merge group")

                    # if walker_clone_nums[keep_idx] > 0:
                    #     import ipdb; ipdb.set_trace()
                    #     print("Attempting to merge a walker which is to be cloned")

                    # update weight
                    new_wt[keep_idx] += new_wt[squash_idx]
                    new_wt[squash_idx] = 0.0

                    # update new_amps
                    new_amp[squash_idx] = 0
                    new_amp[keep_idx] = 1

                    # add the squash index to the merge group
                    merge_groups[keep_idx].append(squash_idx)

                    # add the indices of the walkers that were already
                    # in the merge group that was just squashed
                    merge_groups[keep_idx].extend(merge_groups[squash_idx])

                    # reset the merge group that was just squashed to empty
                    merge_groups[squash_idx] = []

                    # increase the number of clones that the cloned
                    # walker has
                    walker_clone_nums[maxwind] += 1

                    # new spread for starting new stage
                    newspread, wsum = self._calcspread(new_wt, new_amp, distance_matrix)
                    spreads.append(newspread)

                    logging.info("variance after selection:", newspread)

                # if not productive
                else:
                    new_amp[minwind] = 1
                    new_amp[closewalk] = 1
                    new_amp[maxwind] -= 1

        # given we know what we want to clone to specific slots
        # (squashing other walkers) we need to determine where these
        # squashed walkers will be merged
        walker_actions = self.assign_clones(merge_groups, walker_clone_nums)

        # because there is only one step in resampling here we just
        # add another field for the step as 0 and add the walker index
        # to its record as well
        for walker_idx, walker_record in enumerate(walker_actions):
            walker_record['step_idx'] = np.array([0])
            walker_record['walker_idx'] = np.array([walker_idx])

        return walker_actions, spreads[-1]

    def _all_to_all_distance(self, walkers):
        """

        Parameters
        ----------
        walkers :
            

        Returns
        -------

        """
        # initialize an all-to-all matrix, with 0.0 for self distances
        dist_mat = np.zeros((len(walkers), len(walkers)))

        # make images for all the walker states for us to compute distances on
        images = []
        for walker in walkers:
            image = self.distance.image(walker.state)
            images.append(image)

        # get the combinations of indices for all walker pairs
        for i, j in it.combinations(range(len(images)), 2):

            # calculate the distance between the two walkers
            dist = self.distance.image_distance(images[i], images[j])

            # save this in the matrix in both spots
            dist_mat[i][j] = dist
            dist_mat[j][i] = dist

        return [walker_dists for walker_dists in dist_mat], images

    def resample(self, walkers):
        """

        Parameters
        ----------
        walkers :
            

        Returns
        -------

        """

        n_walkers = len(walkers)
        walkerwt = [walker.weight for walker in walkers]
        amp = [1 for i in range(n_walkers)]

        # calculate distance matrix
        distance_matrix, images = self._all_to_all_distance(walkers)

        logging.info("distance_matrix")
        logging.info(np.array(distance_matrix))

        # determine cloning and merging actions to be performed, by
        # maximizing the spread, i.e. the Decider
        resampling_data, spread = self.decide_clone_merge(walkerwt, amp, distance_matrix)

        # convert the target idxs and decision_id to feature vector arrays
        for record in resampling_data:
            record['target_idxs'] = np.array(record['target_idxs'])
            record['decision_id'] = np.array([record['decision_id']])

        # actually do the cloning and merging of the walkers
        resampled_walkers = self.decision.action(walkers, [resampling_data])
        # flatten the distance matrix and give the number of walkers
        # as well for the resampler data, there is just one per cycle
        resampler_data = [{'distance_matrix' : np.ravel(np.array(distance_matrix)),
                           'n_walkers' : np.array([len(walkers)]),
                           'spread' : np.array([spread]),
                           'images' : np.ravel(np.array(images)),
                           'image_shape' : np.array(images[0].shape)}]

        return resampled_walkers, resampling_data, resampler_data
