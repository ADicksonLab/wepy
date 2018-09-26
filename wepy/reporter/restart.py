import os.path as osp
from copy import deepcopy
import pickle
import logging

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

        # copy this reporter
        self_copy = deepcopy(self)

        # save this object as a pickle, open it in the mode
        with open(self.file_path, mode=self.mode+"b") as wf:
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



    def new_sim_manager(self, file_report_suffix=None, reporter_base_path=None):
        """Generate a simulation manager from the objects in this restarter.

        All objects are deepcopied so that this restarter object can
        be used multiple times, without reading from disk again.

        If a `file_report_suffix` string is given all reporters
        inheriting from FileReporter will have their `file_path`
        attribute modified. `file_report_suffix` will be appended to
        the first clause (clauses here are taken to be portions of the
        file name separated by '.'s) so that for 'file.txt' the
        substring 'file' will be replaced by 'file{}'. Where the
        suffix is formatted into that string.

        """

        # check the arguments for correctness
        if (file_report_suffix is not None) or \
           (reporter_base_path is not None):

            # if just the base path is given
            if file_report_suffix is None:
                assert type(reporter_base_path) is str, \
                    "'reporter_base_path' must be a string, given {}".format(type(reporter_base_path))

            if reporter_base_path is None:
                assert type(file_report_suffix) is str, \
                    "'file_report_suffix' must be a string, given {}".format(type(file_report_suffix))

        # copy the reporters from this objects list of reporters, we
        # also need to replace, the restart reporter in that list with
        # this one.
        reporters = []
        for reporter in self.reporters:

            # if the reporter is a restart reporter we replace it with
            # a copy of this reporter, it will be mutated later
            # potentially to change the path to save the pickle
            if isinstance(reporter, RestartReporter):
                reporters.append(deepcopy(self))
            else:
                # otherwise make a copy of the reporter
                reporters.append(deepcopy(reporter))

        # copy all of the other objects before construction
        restart_walkers = deepcopy(self.restart_walkers)
        runner = deepcopy(self.runner)
        resampler = deepcopy(self.resampler)
        boundary_conditions = deepcopy(self.boundary_conditions)
        work_mapper = deepcopy(self.work_mapper)

        # modify them if this was specified

        # update the FileReporter paths
        # iterate through the reporters and add the suffix to the
        # FileReporter subclasses
        for reporter in reporters:

            # check if the reporter class is a FileReporter subclass
            if issubclass(type(reporter), FileReporter):

                # because we are making a new file with any change, we
                # need to modify the access mode to a conservative
                # creation mode
                reporter.mode = 'x'

                if file_report_suffix is not None:
                    filename = osp.basename(reporter.file_path)
                    # get the clauses
                    clauses = filename.split('.')
                    # make a template out of the first clause
                    template_clause = clauses[0] + "{}"

                    # fill in the suffix to the template clause
                    mod_clause = template_clause.format(file_report_suffix)

                    # combine them back into a filename
                    new_filename = '.'.join([mod_clause, *clauses[1:]])
                else:
                    # if we don't have a suffix just return the original name
                    filename = osp.basename(reporter.file_path)
                    new_filename = filename

                # if a new base path was given make that the path
                # to the filename
                if reporter_base_path is not None:
                    new_path = osp.join(reporter_base_path, new_filename)

                # if it wasn't given, add the rest of the original path back
                else:
                    new_path = osp.join(osp.dirname(reporter.file_path), new_filename)

                # make a copy of the reporter and pass this to the
                # sim manager instead of the one in the object
                new_reporter = reporter
                new_reporter.file_path = new_path

        # construct the sim manager
        sim_manager = Manager(restart_walkers,
                              runner=runner,
                              resampler=resampler,
                              boundary_conditions=boundary_conditions,
                              work_mapper=work_mapper,
                              reporters=reporters)
        return sim_manager
