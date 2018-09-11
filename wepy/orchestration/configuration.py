
# these are the supported reporter types for reparametrizing
from wepy.reporter.reporter import FileReporter

# supported work mapper types
from wepy.work_mapper import WorkerMapper

class Configuration():

    SUPPORTED_REPORTERS = (FileReporter,)
    SUPPORTED_WORK_MAPPERS = (WorkerMapper,)

    ROOT_NAME_DEFAULT = 'root'

    def __init__(self, root_name=None, work_mapper=None, reporters=None):

        if work_mapper is not None:
            self._work_mapper = work_mapper
        else:
            # set as a default
            self._work_mapper = None

        if reporters is not None:
            self._reporters = reporters
        else:
            self._reporters = []

        if root_name is not None:
            self._root_name = root_name
        else:
            self._root_name = self.ROOT_NAME_DEFAULT

    @property
    def reporters(self):
        return deepcopy(self._reporters)

    @property
    def work_mapper(self):
        return deepcopy(self._work_mapper)

    # file reporter types

    def reparametrize_file_reporter(self, workdir_path, config_suffix):

        path_template = osp.join(osp.realpath(workdir_path), "{}_{}".format)

    def reparametrize_reporters(self, **kwargs):
        """Accepts variadic number of parameters corresponding to the
        reporters which are ordered. Each argument will be unpacked as
        the arguments to a reporter.reparametrize(*params[i]) method
        supporting reparametrization. Reporters that do not support
        reparametrization should still have a an empty tuple passed.

        """

        new_reporters = []
        # go through the reporters, we access directly because we will
        # make copies ourselves
        for reporter_idx, reporter in enumerate(self._reporters):

            reporter_class = type(reporter)

            # check for the different possible cases and reparametrize
            # them accordingly
            if issubclass(reporter_class, FileReporter):
                new_reporter = self.reparametrize_file_reporter(reporter,
                                                                kwargs['file_reporter'])

            # if it is not any of the supported reporters we just copy
            # it
            else:
                new_reporter = deepcopy(reporter)

            new_reporters.append(new_reporter)

        return new_reporters

    # work mapper types
    def reparametrize_worker_mapper(self, n_workers, worker_type):

        new_work_mapper = self.work_mapper

        if n_workers is not None:
            new_work_mapper.num_workers = n_workers

        if worker_type is not None:
            new_work_mapper.worker_type = worker_type

        return new_work_mapper

    def reparametrize_work_mapper(self, *params):

        work_mapper_class = type(self._work_mapper)

        if issubclass(work_mapper_class, WorkerMapper):
            new_work_mapper = self.reparametrize_worker_mapper(*params)
        else:
            # get a copy of the existing work mapper to use
            new_work_mapper = self.work_mapper

        return new_work_mapper

    def reparametrize(self, work_mapper_args, reporter_args):

        if work_mapper_args is not None:
            new_work_mapper = self.reparametrize_work_mapper(work_mapper_args)
        else:
            new_work_mapper = self.work_mapper

        if reporter_args is not None:
            new_reporters = self.reparametrize_reporters(*reporter_args)
        else:
            new_reporters = self.reporters

        new_configuration = type(self)(work_mapper=new_work_mapper,
                                       reporters=new_reporters)

        return new_configuration
