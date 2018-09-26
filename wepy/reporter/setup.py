from copy import deepcopy
import pickle
import logging

from wepy.reporter.reporter import FileReporter

class SetupReporter(FileReporter):

    def __init__(self, file_path, mode='x'):

        super().__init__(file_path, mode=mode)

    def init(self, *args,
             resampler=None, boundary_conditions=None,
             init_walkers=None,
             **kwargs):

        self.init_walkers = init_walkers
        self.resampler = resampler
        self.boundary_conditions = boundary_conditions

        # copy this reporter with as well as all of the objects it contains
        self_copy = deepcopy(self)

        # save this object as a pickle, open it in the mode
        with open(self.file_path, mode=self.mode+"b") as wf:
            pickle.dump(self_copy, wf)

