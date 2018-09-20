import os.path as osp
from copy import deepcopy

from wepy.work_mapper.mapper import Mapper, WorkerMapper

class Configuration():

    DEFAULT_WORKDIR = osp.realpath(osp.curdir)
    DEFAULT_CONFIG_NAME = "root"
    DEFAULT_NARRATION = ""
    DEFAULT_MODE = 'x'

    def __init__(self,
                 # reporters
                 config_name=None, work_dir=None,
                 mode=None, narration=None,
                 reporter_classes=None,
                 reporter_partial_kwargs=None,
                 # work mappers
                 n_workers=None,
                 work_mapper_class=None,
                 work_mapper_partial_kwargs=None):

        ## reporter stuff

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

        ## work mapper

        self._work_mapper_partial_kwargs = work_mapper_partial_kwargs

        # if the number of workers was sepcified and no work_mapper
        # class was specified default to the WorkerMapper
        if (n_workers is not None) and (work_mapper_class is None):
            self._n_workers = n_workers
            self._work_mapper_class = WorkerMapper
        # if no number of workers was specified and no work_mapper
        # class was specified we default to the serial mapper
        elif (n_workers is None) and (work_mapper_class is None):
            self._n_workers = None
            self._work_mapper_class = Mapper
        # otherwise if the work_mapper class was given we use it and
        # whatever the numbr of workers was
        else:
            self._n_workers = n_workers
            self._work_mapper_class = work_mapper_class

        # work mapper
        if work_mapper is not None:
            self._work_mapper = work_mapper
        else:
            # set as a default
            self._work_mapper = None

        # then generate a work mapper
        self._work_mapper = self._work_mapper_class(n_workers=self._n_workers,
                                                    worker_type=self._worker_type,
                                                    **self._work_mapper_partial_kwargs)


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

    def _gen_work_mapper(self):

        work_mapper = self._work_mapper_class(n_workers=self._n_workers)

        return work_mapper

    @property
    def reporters(self):
        return deepcopy(self._reporters)

    @property
    def work_mapper(self):
        return deepcopy(self._work_mapper)


    def reparametrize(self, **kwargs):

        # dictionary of the possible reparametrizations from the
        # current configuration
        params = {# related to the work mapper
                  'n_workers' : self.n_workers,
                  'work_mapper' : self.work_mapper,
                  # those related to the reporters
                  'mode' : self.mode,
                  'config_name' : self.config_name,
                  'work_dir' : self.work_dir,
                  'narration' : self.narration,
                  'reporter_classes' : self.reporter_classes,
                  'reporter_partial_kwargs' : self.reporter_partial_kwargs}

        for key, value in kwargs.items():
            # if the value is given we replace the old one with it
            if value is not None:
                params[key] = value

        new_configuration = type(self)(**params)

        return new_configuration
