import numpy as np

from wepy.rsampling.novelty.novelty import Novelty

class Weight_Factor_Novelty(Novelty):

    def __init__(self, n_walkers, pmin=1e-12):
        
        # The minimum weight for a walker
        self.pmin = pmin
        
        # log(probability_min)
        self.lpmin = np.log(pmin / 100)

        # The total number of walkers
        assert n_walkers is not None, "Must give the total number of walkers"
        self.n_walkers = n_walkers

        self.clone_merge_iter = 0

        # setting the weights parameter

    def calculate_novelty(self, walkerwt, amp, images):
        
        # Weight factor for the walkers
        novelty = np.zeros(self.n_walkers)

        # Set the weight factor
        for walker in range(self.n_walkers):
            
            if walkerwt[walker] > 0 and amp[walker] > 0:
                novelty[walker] = np.log(walkerwt[walker] / amp[walker]) - self.lpmin

            else:
                novelty[walker] = 0
                
            if novelty[walker] < 0:
                novelty[walker] = 0

        return(novelty)
        
