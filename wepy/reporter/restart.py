from copy import deepcopy
import pickle

from wepy.reporter.reporter import FileReporter

from wepy.sim_manager import Manager

class RestartReporter(FileReporter):

    def __init__(self, file_path, mode='x'):

        super().__init__(file_path, mode=mode)

    def init(self, *args,
             runner=None,
             resampler=None,
             boundary_conditions=None,
             work_mapper=None,
             reporters=None,
             **kwargs):

        # get a reference for each the parts
        self.runner = runner
        self.resampler = resampler
        self.boundary_conditions = boundary_conditions
        self.reporters = reporters

    def cleanup(self, *args, work_mapper=None, **kwargs):

        # before calling this be sure that the other reporters have
        # been cleaned up or at least be sure they don't contain
        # anything that will cause errors on deepcopying or pickle

        self.work_mapper = work_mapper

        # copy the sim_manager
        self_copy = deepcopy(self)

        # save this object as a pickle, open it in the mode
        with open(self.file_path, mode=self.mode+"+b") as wf:
            pickle.dump(self_copy, wf)


    def report(self, cycle_idx, new_walkers, *args,
               resampled_walkers=None,
               **kwargs):

        self.cycle_idx = cycle_idx

        # walkers after the sampling segment
        self.new_walkers = new_walkers

        # resampled walkers which would be used if the run was
        # continued (what we are preparing for here)
        self.restart_walkers = resampled_walkers



    def new_sim_manager(self):

        sim_manager = Manager(self.restart_walkers,
                              runner=self.runner,
                              resampler=self.resampler,
                              boundary_conditions=self.boundary_conditions,
                              work_mapper=self.work_mapper,
                              reporters=self.reporters)
        return sim_manager
