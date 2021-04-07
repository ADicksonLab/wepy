import numpy as np

class Novelty():

    cycle_counter = 0
    clone_merge_iter = 0
    sample_counter = 0

    def __init__(self):
        pass

    def calculate_novelty(self, image = None, amp = None, weight = None):
        """
        Calculate the novelty factor used in determining the spread
        between walkers that govern merging and cloning.
        """

        raise NotImplementedError

    def update_cycle_counter(self):
        self.cycle_counter += 1

    def update_clone_merge_iter(self):
        self.n_clones += 1

    def reset_clone_merge_iter(self):
        self.n_clone_merge_iter = 0

    def update_sample_counter(self):
        self.sample_counter += 1
