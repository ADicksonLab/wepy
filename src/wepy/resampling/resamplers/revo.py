import multiprocessing as mulproc
import random as rand
import itertools as it

import logging
from eliot import start_action, log_call

import numpy as np

from wepy.resampling.resamplers.resampler import Resampler
from wepy.resampling.resamplers.clone_merge  import CloneMergeResampler
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision

class REVOResampler(CloneMergeResampler):
    r"""Resampler implementing the REVO algorithm.

    You can find more detailed information in the paper "REVO:
    Resampling of ensembles by variation optimization" but
    briefly:

    REVO is a Weighted Ensemble based enhanced sampling algorithm
    which uses cloning and merging to create ensembles of diverse
    trajectories without defining any regions. It instead optimizes a
    measure of “variation” that depends on the pairwise distances
    between the walkers and their weights.


    REVO solves this optimization problem using a greedy algorithm
    which at each step selects best walkers for resampling operations
    (cloning and merging) in order to maximize the "trajectory
    variation".

    The trajectory variation is defined as


    .. math::
        V = \sum_{i} V_i = \sum_i \sum_{j}(\frac{d_{ij}}{d_0})
        ^{\alpha}\phi_i\phi_j

    where

    :math:`V_i` : the trajectory variation value of walker `i`

    :math:`d_{ij}` : the distance between walker i and j
    according the distance metric

    :math:`\alpha` : modulates the influence of the distances in the
    variation calculation

    :math:`d_0` : the characteristic distance and is used to make the
    equation unitless.

    :math:`\phi` : is a non-negative function which is a measure of
    the relative importance of the walker and is referred to as a
    "novelty function".  Here it is a function of a walker's weight.

    Furthermore REVO needs the following parameters:

       pmin: the minimum statistical weight. REVO does not clone
       walkers with a weight less than pmin.

       pmax: The maximum statistical weight. It prevents the
       accumulation of too much weight in one walker.

       merge_dist: This is the merge-distance threshold. The distance
        between merged walkers should be less than this value.

    The resample function, called during every cycle, takes the
    ensemble of walkers and performs the follow steps:

       - Calculate the pairwise all-to-all distance matrix using the distance metric
       - Decides which walkers should be merged or cloned
       - Applies the cloning and merging decisions to get the resampled walkers
       - Creates the resampling data that includes
       - distance_matrix : the calculated all-to-all distance matrix

           - n_walkers : the number of walkers. number of walkers is
             kept constant thought the resampling.

           - variation : the final value of trajectory variation

           - images : the images of walkers that is defined by the distance object

           - image_shape : the shape of the image

    The algorithm saves the records of cloning and merging
    information in resampling data.

    Only the net clones and merges are recorded in the resampling records.

    """

    # fields for resampler data
    RESAMPLING_FIELDS = CloneMergeResampler.RESAMPLING_FIELDS
    RESAMPLING_SHAPES = CloneMergeResampler.RESAMPLING_SHAPES #+ (Ellipsis,)
    RESAMPLING_DTYPES = CloneMergeResampler.RESAMPLING_DTYPES #+ (np.int,)


    # fields that can be used for a table like representation
    RESAMPLING_RECORD_FIELDS = CloneMergeResampler.RESAMPLING_RECORD_FIELDS

    # fields for resampling data
    RESAMPLER_FIELDS = CloneMergeResampler.RESAMPLER_FIELDS + \
                       ('num_walkers', 'distance_matrix', 'variation', 'image_shape', 'images')
    RESAMPLER_SHAPES = CloneMergeResampler.RESAMPLER_SHAPES + \
                       ((1,), Ellipsis, (1,), Ellipsis, Ellipsis)
    RESAMPLER_DTYPES = CloneMergeResampler.RESAMPLER_DTYPES + \
                       (np.int, np.float, np.float, np.int, None)

    # fields that can be used for a table like representation
    RESAMPLER_RECORD_FIELDS = CloneMergeResampler.RESAMPLER_RECORD_FIELDS + \
                              ('variation',)


    def __init__(self,
                 merge_dist=None,
                 char_dist=None,
                 distance=None,
                 init_state=None,
                 weights=True,
                 pmin=1e-12,
                 pmax=0.1,
                 dist_exponent=4,
                 seed=None,
                 **kwargs):

        """Constructor for the REVO Resampler.

        Parameters
        ----------

        dist_exponent : int
          The distance exponent that modifies distance and weight novelty
          relative to each other in the variation equation.

        merge_dist : float
            The merge distance threshold. Units should be the same as
            the distance metric.

        char_dist : float
            The characteristic distance value. It is calculated by
            running a single dynamic cycle and then calculating the
            average distance between all walkers. Units should be the
            same as the distance metric.

        distance : object implementing Distance
            The distance metric to compare walkers.

        weights : bool
            Turns off or on the weight novelty in
            calculating the variation equation. When weight is
            False, the value of the novelty function is set to 1 for all
            walkers.

        init_state : WalkerState object
            Used for automatically determining the state image shape.

        seed : None or int, optional
            The random seed. If None, the system (random) one will be used.

        """

        # call the init methods in the CloneMergeResampler
        # superclass. We set the min and max number of walkers to be
        # constant
        super().__init__(pmin=pmin, pmax=pmax,
                         min_num_walkers=Ellipsis,
                         max_num_walkers=Ellipsis,
                         **kwargs)

        assert merge_dist is not None, "Merge distance must be given."
        assert distance is not None,  "Distance object must be given."
        assert char_dist is not None, "Characteristic distance value (d0) must be given"
        assert init_state is not None,  "An initial state must be given."

        # ln(probability_min)
        self.lpmin = np.log(self.pmin/100)
        self.dist_exponent = dist_exponent

        # the distance metric

        self.merge_dist = merge_dist

        # the distance metric

        self.distance = distance

        # the characteristic distance, char_dist

        self.char_dist = char_dist

        # setting the random seed
        self.seed = seed
        if seed is not None:
            rand.seed(seed)

        # setting the weights parameter
        self.weights = weights

        # we do not know the shape and dtype of the images until
        # runtime so we determine them here

        image = self.distance.image(init_state)
        self.image_dtype = image.dtype

    def resampler_field_dtypes(self):
        """ Finds out the datatype of the image.

        Returns
        -------
        datatypes : tuple of datatype
        The type of reasampler image.

        """

        # index of the image idx
        image_idx = self.resampler_field_names().index('images')

        # dtypes adding the image dtype
        dtypes = list(super().resampler_field_dtypes())
        dtypes[image_idx] = self.image_dtype

        return tuple(dtypes)

    def _novelty(self, walker_weight, num_walker_copy):
        """Calculates the novelty fuction value.

        Parameters
        ----------

        walker_weight : float
            The weight of the walker.

        num_walker_copy : int
          The number of copies of the walker.

        Returns
        -------
        novelty : float
        The calcualted value of novelty for the given walker.

        """

        novelty = 0

        if walker_weight > 0 and num_walker_copy > 0:

            if self.weights:

                novelty = np.log(walker_weight/num_walker_copy) - self.lpmin

            else:

                novelty = 1

        if novelty < 0:

            novelty = 0

        return novelty

    def _calcvariation(self, walker_weights, num_walker_copies, distance_matrix):
        """Calculates the variation value.

        Parameters
        ----------

        walker_weights : list of float
            The weights of all walkers. The sum of all weights should be 1.0.

        num_walker_copies : list of int
            The number of copies of each walker.
            0 means the walker is not exists anymore.
            1 means there is one of the this walker.
            >1 means it should be cloned to this number of walkers.

        distance_matrix : list of arraylike of shape (num_walkers)

        Returns
        -------
        variation : float
           The calculated variation value.

        walker_variations : arraylike of shape (num_walkers)
           The Vi value of each walker.

        """

        num_walkers = len(walker_weights)



        # set the novelty values
        walker_novelties = np.array([self._novelty(walker_weights[i], num_walker_copies[i])
                               for i in range(num_walkers)])


        # the value to be optimized
        variation = 0

        # the walker variation values (Vi values)
        walker_variations = np.zeros(num_walkers)


        # calculate the variation and walker variation values
        for i in range(num_walkers - 1):

            if num_walker_copies[i] > 0:
                for j in range(i+1, num_walkers):

                    if num_walker_copies[j] > 0:

                        partial_variation = ((distance_matrix[i][j] / self.char_dist) ** self.dist_exponent) \
                        * walker_novelties[i] * walker_novelties[j]

                        variation += partial_variation * num_walker_copies[i] * num_walker_copies[j]
                        walker_variations[i] += partial_variation * num_walker_copies[j]
                        walker_variations[j] += partial_variation * num_walker_copies[i]

        return variation, walker_variations

    def decide(self, walker_weights, num_walker_copies, distance_matrix):
        """Optimize the trajectory variation by making decisions for resampling.

        Parameters
        ----------

        walker_weights : list of flaot
            The weights of all walkers. The sum of all weights should be 1.0.

        num_walker_copies : list of int
            The number of copies of each walker.
            0 means the walker is not exists anymore.
            1 means there is one of the this walker.
            >1 means it should be cloned to this number of walkers.

        distance_matrix : list of arraylike of shape (num_walkers)

        Returns
        -------
        variation : float
            The optimized value of the trajectory variation.

        resampling_data : list of dict of str: value
            The resampling records resulting from the decisions.

        """
        num_walkers = len(walker_weights)

        variations = []
        merge_groups = [[] for i in range(num_walkers)]
        walker_clone_nums = [0 for i in range(num_walkers)]

        # make copy of walkers properties
        new_walker_weights = walker_weights.copy()
        new_num_walker_copies = num_walker_copies.copy()


        # calculate the initial variation which will be optimized
        variation, walker_variations = self._calcvariation(walker_weights,
                                                           new_num_walker_copies,
                                                           distance_matrix)
        variations.append(variation)

        # maximize the variance through cloning and merging
        logging.info("Starting variance optimization: {}".format(variation))

        productive = True
        while productive:
            productive = False
            # find min and max walker_variationss, alter new_amp

            # initialize to None, we may not find one of each
            min_idx = None
            max_idx = None

            # selects a walker with minimum walker_variations and a walker with
            # maximum walker_variations walker (distance to other walkers) will be
            # tagged for cloning (stored in maxwind), except if it is
            # already a keep merge target
            max_tups = []
            for i, value in enumerate(walker_variations):
                # 1. must have an amp >=1 which gives the number of clones to be made of it
                # 2. clones for the given amplitude must not be smaller than the minimum probability
                # 3. must not already be a keep merge target
                if (new_num_walker_copies[i] >= 1) and \
                   (new_walker_weights[i]/(new_num_walker_copies[i] + 1) > self.pmin) and \
                   (len(merge_groups[i]) == 0):
                    max_tups.append((value, i))


            if len(max_tups) > 0:
                max_value, max_idx = max(max_tups)

            # walker with the lowest walker_variations (distance to other walkers)
            # will be tagged for merging (stored in min_idx)
            min_tups = [(value, i) for i,value in enumerate(walker_variations)
                        if new_num_walker_copies[i] == 1 and (new_walker_weights[i]  < self.pmax)]

            if len(min_tups) > 0:
                min_value, min_idx = min(min_tups)

            # does min_idx have an eligible merging partner?
            # closedist = self.merge_dist
            closewalk = None
            condition_list = np.array([i is not None for i in [min_idx, max_idx]])
            if condition_list.all() and min_idx != max_idx:

                # get the walkers that aren't the minimum and the max
                # walker_variations walkers, as candidates for merging
                closewalks = set(range(num_walkers)).difference([min_idx, max_idx])

                # remove those walkers that if they were merged with
                # the min walker_variations walker would violate the pmax
                closewalks = [idx for idx in closewalks
                                      if (new_num_walker_copies[idx]==1) and
                                       (new_walker_weights[idx] + new_walker_weights[min_idx] < self.pmax)
                                      ]

                # if there are any walkers left, get the distances of
                # the close walkers to the min walker_variations walker if that
                # distance is less than the maximum merge distance
                if len(closewalks) > 0:
                    closewalks_dists = [(distance_matrix[min_idx][i], i) for i in closewalks
                                            if distance_matrix[min_idx][i] < (self.merge_dist)]

                    # if any were found set this as the closewalk
                    if len(closewalks_dists) > 0:
                        closedist, closewalk = min(closewalks_dists)


            # did we find a closewalk?
            condition_list = np.array([i is not None for i in [min_idx, max_idx, closewalk]])
            #if we find a walker for cloning, a walker and its close neighbor for merging
            if condition_list.all() :

                # change new_amp
                tempsum = new_walker_weights[min_idx] + new_walker_weights[closewalk]
                new_num_walker_copies[min_idx] = new_walker_weights[min_idx]/tempsum
                new_num_walker_copies[closewalk] = new_walker_weights[closewalk]/tempsum
                new_num_walker_copies[max_idx] += 1

                # re-determine variation function, and walker_variations values
                new_variation, walker_variations = self._calcvariation(new_walker_weights, new_num_walker_copies, distance_matrix)

                if new_variation > variation:
                    variations.append(new_variation)

                    logging.info("Variance move to {} accepted".format(new_variation))

                    productive = True
                    variation = new_variation

                    # make a decision on which walker to keep
                    # (min_idx, or closewalk), equivalent to:
                    # `random.choices([closewalk, min_idx],
                    #                 weights=[new_walker_weights[closewalk], new_walker_weights[min_idx])`
                    r = rand.uniform(0.0, new_walker_weights[closewalk] + new_walker_weights[min_idx])

                     # keeps closewalk and gets rid of min_idx
                    if r < new_walker_weights[closewalk]:
                        keep_idx = closewalk
                        squash_idx = min_idx

                    # keep min_idx, get rid of closewalk
                    else:
                        keep_idx = min_idx
                        squash_idx = closewalk

                    # update weight
                    new_walker_weights[keep_idx] += new_walker_weights[squash_idx]
                    new_walker_weights[squash_idx] = 0.0

                    # update new_num_walker_copies
                    new_num_walker_copies[squash_idx] = 0
                    new_num_walker_copies[keep_idx] = 1

                    # add the squash index to the merge group
                    merge_groups[keep_idx].append(squash_idx)

                    # add the indices of the walkers that were already
                    # in the merge group that was just squashed
                    merge_groups[keep_idx].extend(merge_groups[squash_idx])

                    # reset the merge group that was just squashed to empty
                    merge_groups[squash_idx] = []

                    # increase the number of clones that the cloned
                    # walker has
                    walker_clone_nums[max_idx] += 1

                    # new variation for starting new stage
                    new_variation, walker_variations = self._calcvariation(new_walker_weights,
                                                                          new_num_walker_copies,
                                                                          distance_matrix)
                    variations.append(new_variation)

                    logging.info("variance after selection: {}".format(new_variation))

                # if not productive
                else:
                    new_num_walker_copies[min_idx] = 1
                    new_num_walker_copies[closewalk] = 1
                    new_num_walker_copies[max_idx] -= 1

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

        return walker_actions, variations[-1]

    def _all_to_all_distance(self, walkers):
        """ Calculate the pairwise all-to-all distances between walkers.

        Parameters
        ----------
        walkers : list of walkers


        Returns
        -------
        distance_matrix : list of arraylike of shape (num_walkers)

        images : list of image obeject

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

    @log_call(include_args=[],
              include_result=False)
    def resample(self, walkers):
        """Resamples walkers based on REVO algorithm

        Parameters
        ----------
        walkers : list of walkers


        Returns
        -------
        resampled_walkers : list of resampled_walkers

        resampling_data : list of dict of str: value
            The resampling records resulting from the decisions.

        resampler_data :list of dict of str: value
            The resampler records resulting from the resampler actions.

        """

        #initialize the parameters
        num_walkers = len(walkers)
        walker_weights = [walker.weight for walker in walkers]
        num_walker_copies = [1 for i in range(num_walkers)]

        # calculate distance matrix
        distance_matrix, images = self._all_to_all_distance(walkers)

        logging.info("distance_matrix")
        logging.info("\n{}".format(str(np.array(distance_matrix))))

        # determine cloning and merging actions to be performed, by
        # maximizing the variation, i.e. the Decider
        resampling_data, variation = self.decide(walker_weights, num_walker_copies, distance_matrix)

        # convert the target idxs and decision_id to feature vector arrays
        for record in resampling_data:
            record['target_idxs'] = np.array(record['target_idxs'])
            record['decision_id'] = np.array([record['decision_id']])

        # actually do the cloning and merging of the walkers
        resampled_walkers = self.DECISION.action(walkers, [resampling_data])

       # flatten the distance matrix and give the number of walkers
        # as well for the resampler data, there is just one per cycle
        resampler_data = [{'distance_matrix' : np.ravel(np.array(distance_matrix)),
                           'num_walkers' : np.array([len(walkers)]),
                           'variation' : np.array([variation]),
                           'images' : np.ravel(np.array(images)),
                           'image_shape' : np.array(images[0].shape)}]

        return resampled_walkers, resampling_data, resampler_data
