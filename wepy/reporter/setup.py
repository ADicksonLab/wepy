from copy import deepcopy
import pickle

from wepy.reporter.reporter import FileReporter

class SetupReporter(FileReporter):

    def __init__(self, file_path, mode='x',
                 resampler=None,
                 boundary_conditions=None):

        super().__init__(file_path, mode=mode)

        self.resampler = resampler
        self.boundary_conditions = boundary_conditions

    def init(self, *args, **kwargs):

        # copy this reporter with as well as all of the objects it contains
        self_copy = deepcopy(self)

        # save this object as a pickle, open it in the mode
        with open(self.file_path, mode=self.mode+"+b") as wf:
            pickle.dump(self_copy, wf)

    def cleanup(self, *args, **kwargs):
        pass

    def report(self, *args, **kwargs):
        pass
