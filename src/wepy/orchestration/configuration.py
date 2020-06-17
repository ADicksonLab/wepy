import os.path as osp
from copy import deepcopy
import logging
import itertools as it

from wepy.work_mapper.mapper import Mapper, WorkerMapper
from wepy.work_mapper.worker import Worker

class Configuration():
    """ """

    DEFAULT_WORKDIR = osp.realpath(osp.curdir)
    DEFAULT_CONFIG_NAME = "root"
    DEFAULT_NARRATION = ""
    DEFAULT_REPORTER_CLASS = ""

    # if there is to be reporter class in filenames use this template
    # to put it into the filename
    REPORTER_CLASS_SEG_TEMPLATE = ".{}"
    DEFAULT_MODE = 'x'

    def __init__(self,
                 # reporters
                 config_name=None,
                 work_dir=None,
                 mode=None,
                 narration=None,
                 reporter_classes=None,
                 reporter_partial_kwargs=None,
                 # work mappers
                 work_mapper_class=None,
                 work_mapper_partial_kwargs=None,
                 # monitors
                 monitor_class=None,
                 monitor_partial_kwargs=None,
                 # apparatus configuration options
                 apparatus_opts=None,
    ):

        ## reporter stuff

        # reporters and partial kwargs
        if reporter_classes is not None:
            self._reporter_classes = reporter_classes
        else:
            self._reporter_classes = []

        if reporter_partial_kwargs is not None:
            self._reporter_partial_kwargs = reporter_partial_kwargs
        else:
            self._reporter_partial_kwargs = []

        # file path localization variables

        # config string
        if config_name is not None:
            self._config_name = config_name
        else:
            self._config_name = self.DEFAULT_CONFIG_NAME

        if work_dir is not None:
            self._work_dir = work_dir
        else:
            self._work_dir = self.DEFAULT_WORKDIR

        # narration
        if narration is not None:
            narration = "_{}".format(narration) if len(narration) > 0 else ""
            self._narration = narration
        else:
            self._narration = self.DEFAULT_NARRATION


        # file modes, if none are given we set to the default, this
        # needs to be done before generating the reporters
        if mode is not None:
            self._mode = mode
        else:
            self._mode = self.DEFAULT_MODE

        # generate the reporters for this configuration
        self._reporters = self._gen_reporters()

        ## work mapper

        # the partial kwargs that will be passed for reparametrization
        if work_mapper_partial_kwargs is None:
            self._work_mapper_partial_kwargs = {}
        else:
            self._work_mapper_partial_kwargs = work_mapper_partial_kwargs

        # if the number of workers is not given set it to None
        if 'num_workers' not in self._work_mapper_partial_kwargs:
            self._work_mapper_partial_kwargs['num_workers'] = None

        # same for the worker type
        if 'worker_type' not in self._work_mapper_partial_kwargs:
            self._work_mapper_partial_kwargs['worker_type'] = None

        # if the number of workers was sepcified and no work_mapper
        # class was specified default to the WorkerMapper
        if (self._work_mapper_partial_kwargs['num_workers'] is not None) and \
           (work_mapper_class is None):
            self._work_mapper_class = WorkerMapper

        # if no number of workers was specified and no work_mapper
        # class was specified we default to the serial mapper
        elif (self._work_mapper_partial_kwargs['num_workers'] is None) and \
             (work_mapper_class is None):
            self._work_mapper_class = Mapper

        # otherwise if the work_mapper class was given we use it and
        # whatever the number of workers was
        else:
            self._work_mapper_class = work_mapper_class

        # then generate a work mapper
        self._work_mapper = self._work_mapper_class(**self._work_mapper_partial_kwargs)


        ### Monitor options

        # get the names of the reporters in the order they are
        reporter_order = tuple([str(reporter_class.__name__)
                                for reporter_class in self._reporter_classes])

        # init the kwargs for the monitor
        if monitor_partial_kwargs is None:
            self._monitor_partial_kwargs = {}
        else:
            self._monitor_partial_kwargs = monitor_partial_kwargs

        # choose the monitor class (None is okay)
        self._monitor_class = monitor_class

        # generate the object
        if self._monitor_class is not None:

            self._monitor = self._monitor_class(
                reporter_order=reporter_order,
                **self._monitor_partial_kwargs,
            )

        else:
            self._monitor = None

        ### Apparatus options

        # the runtime configuration of the apparatus can be configured
        # via these options.
        self._apparatus_opts = apparatus_opts if apparatus_opts is not None else {}

    @property
    def reporter_classes(self):
        """ """
        return self._reporter_classes

    @property
    def reporter_partial_kwargs(self):
        """ """
        return self._reporter_partial_kwargs

    @property
    def config_name(self):
        """ """
        return self._config_name

    @property
    def work_dir(self):
        """ """
        return self._work_dir

    @property
    def narration(self):
        """ """
        return self._narration

    @property
    def mode(self):
        """ """
        return self._mode

    @property
    def reporters(self):
        """ """
        return self._reporters

    @property
    def work_mapper_class(self):
        """ """
        return self._work_mapper_class

    @property
    def work_mapper_partial_kwargs(self):
        """ """
        return self._work_mapper_partial_kwargs

    @property
    def work_mapper(self):
        """ """
        return self._work_mapper

    @property
    def monitor_class(self):
        """ """
        return self._monitor_class

    @property
    def monitor_partial_kwargs(self):
        """ """
        return self._monitor_partial_kwargs

    @property
    def monitor(self):
        """ """
        return self._monitor


    @property
    def apparatus_opts(self):
        """ """
        return self._apparatus_opts


    def _gen_reporters(self):
        """ """

        # check the extensions of all the reporters. If any of them
        # are the same raise a flag to add the reporter names to the
        # filenames

        # the number of filenames
        all_exts = list(it.chain(*[[ext for ext in rep.SUGGESTED_EXTENSIONS]
                                 for rep in self.reporter_classes]))
        n_exts = len(all_exts)

        # the number of unique ones
        n_unique_exts = len(set(all_exts))

        duplicates = False
        if n_unique_exts < n_exts:
            duplicates = True

        # then go through and make the inputs for each reporter
        reporters = []
        for idx, reporter_class in enumerate(self.reporter_classes):

            # first we have to generate the filenames for all the
            # files this reporter needs. The number of file names the
            # reporter needs is given by the number of suggested
            # extensions it has
            file_paths = []
            for extension in reporter_class.SUGGESTED_EXTENSIONS:


                # if previously found that there are duplicates in the
                # extensions we need to name with the reporter class string
                if duplicates:

                    # use the __name__ attribute of the class and put
                    # it into the template to make a segment out of it
                    reporter_class_seg_str = self.REPORTER_CLASS_SEG_TEMPLATE.format(
                        reporter_class.__name__)

                    # then make the filename with this
                    filename = reporter_class.SUGGESTED_FILENAME_TEMPLATE.format(
                                       narration=self.narration,
                                       config=self.config_name,
                                       reporter_class=reporter_class_seg_str,
                                       ext=extension)

                # otherwise don't use the reporter class names to keep it clean
                else:
                    filename = reporter_class.SUGGESTED_FILENAME_TEMPLATE.format(
                                       narration=self.narration,
                                       config=self.config_name,
                                       reporter_class=self.DEFAULT_REPORTER_CLASS,
                                       ext=extension)


                file_path = osp.join(self.work_dir, filename)

                file_paths.append(file_path)

            modes = [self.mode for i in range(len(file_paths))]

            reporter = reporter_class(file_paths=file_paths, modes=modes,
                                  **self.reporter_partial_kwargs[idx])

            reporters.append(reporter)

        return reporters

    # TODO: remove, not used
    def _gen_work_mapper(self):
        """ """

        work_mapper = self._work_mapper_class(n_workers=self._default_n_workers)

        return work_mapper

    @property
    def reporters(self):
        """ """
        return deepcopy(self._reporters)

    @property
    def work_mapper(self):
        """ """
        return deepcopy(self._work_mapper)

    def reparametrize(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs :
            

        Returns
        -------

        """

        # dictionary of the possible reparametrizations from the
        # current configuration
        params = {

            # related to the work mapper
            'work_mapper_class' : self.work_mapper_class,
            'work_mapper_partial_kwargs' : self.work_mapper_partial_kwargs,

            # monitor
            'monitor_class' : self.monitor_class,
            'monitor_partial_kwargs' : self.monitor_partial_kwargs,

            # those related to the reporters
            'mode' : self.mode,
            'config_name' : self.config_name,
            'work_dir' : self.work_dir,
            'narration' : self.narration,
            'reporter_classes' : self.reporter_classes,
            'reporter_partial_kwargs' : self.reporter_partial_kwargs,

            # apparatus
            'apparatus_opts' : self.apparatus_opts,
        }

        for key, value in kwargs.items():

            # for the partial kwargs we need to update them not
            # completely overwrite
            if key in [
                    'work_mapper_partial_kwargs',
                    'reporter_partial_kwargs',
                    'monitor_partial_kwargs',
            ]:
                if value is not None:
                    params[key].update(value)

            # if the value is given we replace the old one with it
            elif value is not None:
                params[key] = value

        new_configuration = type(self)(**params)

        return new_configuration
