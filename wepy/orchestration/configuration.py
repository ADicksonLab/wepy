import os.path as osp
from copy import deepcopy

# these are the supported reporter types for reparametrizing
from wepy.reporter.reporter import FileReporter

# supported work mapper types
from wepy.work_mapper.mapper import WorkerMapper

class Configuration():

    DEFAULT_WORKDIR = osp.realpath(osp.curdir)
    DEFAULT_CONFIG_NAME = "root"
    DEFAULT_NARRATION = ""
    DEFAULT_MODE = 'x'

    def __init__(self, config_name=None, work_dir=None, narration=None,
                 mode=None,
                 work_mapper=None,
                 reporter_classes=None, reporter_partial_kwargs=None):

        # reporters and partial kwargs
        if reporter_classes is not None:
            self.reporter_classes = reporter_classes
        else:
            self.reporter_classes = []

        if reporter_partial_kwargs is not None:
            self.reporter_partial_kwargs = reporter_partial_kwargs
        else:
            self.reporter_partial_kwargs = []

        # file path localization variables

        # config string
        if config_name is not None:
            self.config_name = config_name
        else:
            self.config_name = self.DEFAULT_CONFIG_NAME

        if work_dir is not None:
            self.work_dir = work_dir
        else:
            self.work_dir = self.DEFAULT_WORKDIR

        # narration


        if narration is not None:
            narration = "_{}".format(narration) if len(narration) > 0 else ""
            self.narration = narration
        else:
            self.narration = self.DEFAULT_NARRATION


        # file modes, if none are given we set to the default, this
        # needs to be done before generating the reporters
        if mode is not None:
            self.mode = mode
        else:
            self.mode = self.DEFAULT_MODE

        # generate the reporters for this configuration
        self._reporters = self._gen_reporters()

        # work mapper
        if work_mapper is not None:
            self._work_mapper = work_mapper
        else:
            # set as a default
            self._work_mapper = None

    def _gen_reporters(self):

        reporters = []
        for idx, reporter_class in enumerate(self.reporter_classes):

            filename = reporter_class.SUGGESTED_FILENAME_TEMPLATE.format(
                               narration=self.narration,
                               config=self.config_name,
                               ext=reporter_class.SUGGESTED_EXTENSION)

            file_path = osp.join(self.work_dir, filename)

            file_paths = [file_path]

            modes = [self.mode for i in range(len(file_paths))]

            reporter = reporter_class(file_paths=file_paths, modes=modes,
                                      **self.reporter_partial_kwargs[idx])

            reporters.append(reporter)

        return reporters

    @property
    def reporters(self):
        return deepcopy(self._reporters)

    @property
    def work_mapper(self):
        return deepcopy(self._work_mapper)


    def reparametrize(self, **kwargs):

        # dictionary of the possible reparametrizations from the
        # current configuration
        params = {'mode' : self.mode,
                  'config_name' : self.config_name,
                  'work_dir' : self.work_dir,
                  'narration' : self.narration,
                  'work_mapper' : self.work_mapper,
                  'reporter_classes' : self.reporter_classes,
                  'reporter_partial_kwargs' : self.reporter_partial_kwargs}

        for key, value in kwargs.items():
            # if the value is given we replace the old one with it
            if value is not None:
                params[key] = value

        new_configuration = type(self)(**params)

        return new_configuration


    ### TODO remove or not...
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
