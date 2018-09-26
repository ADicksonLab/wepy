import os
import os.path as osp
import pickle
import logging

class ReporterError(Exception):
    pass

class Reporter(object):

    def __init__(self, **kwargs):
        pass

    def init(self, **kwargs):
        method_name = 'init'
        assert not hasattr(super(), method_name), \
            "Superclass with method {} is masked".format(method_name)

    def report(self, **kwargs):
        method_name = 'report'
        assert not hasattr(super(), method_name), \
            "Superclass with method {} is masked".format(method_name)

    def cleanup(self, **kwargs):
        method_name = 'cleanup'
        assert not hasattr(super(), method_name), \
            "Superclass with method {} is masked".format(method_name)


class FileReporter(Reporter):

    MODES = ('x', 'w', 'w-', 'r', 'r+',)

    DEFAULT_MODE = 'x'

    # these are keywords that can be recognized by subclasses of
    # FileReporter in kwargs in order to bypass path methods, in order
    # to always support a direct file_path setting method. For example
    # in the ParametrizableFileReporter if you don't want to set the
    # parametrizable things then you just pass in one of the bypass
    # keywords and it will skip its generation of the file_paths
    # through components
    BYPASS_KEYWORDS = ('file_path', 'file_paths',)

    SUGGESTED_FILENAME_TEMPLATE = "{config}{narration}.{ext}"

    SUGGESTED_EXTENSION = 'report'

    def __init__(self, file_paths=None, modes=None,
                 file_path=None, mode=None,
                 **kwargs):

        # file paths

        assert not ((file_paths is not None) and (file_path is not None)), \
            "only file_paths or file_path kwargs can be specified"

        # if only one file path is given then we handle it as multiple
        if file_path is not None:
            file_paths = [file_path]

        self._file_paths = file_paths


        # modes

        assert not ((modes is not None) and (mode is not None)), \
            "only modes or mode kwargs can be specified"

        # if modes is None we make modes, from defaults if we have to
        if modes is None:

            # if mode is None set it to the default
            if modes is None and mode is None:
                mode = self.DEFAULT_MODE

            # if only one mode is given copy it for each file given
            modes = [mode for i in range(len(self._file_paths))]

        self._modes = modes

        super().__init__(**kwargs)

    def _bypass_dispatch(self, **kwargs):

        # check if we are bypassing the parametrization for
        # compatibility
        if any([True if key in self.BYPASS_KEYWORDS else False
                for key in kwargs.keys()]):

            # we just call the superclass methods then
            FileReporter.__init__(self, **kwargs)

            # unfortunately without doing metaclass weird stuff the
            # returned object will be an unparametrizable
            # ParamatrizableFileReporter but I think its okay for the
            # use cases it will be used for

            return True
        else:
            return False


    def _validate_mode(self, mode):
        if mode in self.MODES:
            return True
        else:
            return False

    @property
    def mode(self):
        if len(self._file_paths) > 1:
            raise ReporterError("there are multiple files and modes defined")

        return self._modes[0]

    @property
    def file_path(self):
        if len(self._file_paths) > 1:
            raise ReporterError("there are multiple files and modes defined")

        return self._file_paths[0]

    @property
    def file_paths(self):
        return self._file_paths

    @file_paths.setter
    def file_paths(self, file_paths):
        for i, file_path in enumerate(file_paths):
            self.set_path(i, file_path)

    def set_path(self, file_idx, path):
        self._paths[file_idx] = path

    @property
    def modes(self):
        return self._modes

    @modes.setter
    def modes(self, modes):
        for i, mode in enumerate(modes):
            self.set_mode(i, mode)

    def set_mode(self, file_idx, mode):

        if self._validate_mode(mode):
            self._modes[file_idx] = mode
        else:
            raise ValueError("Incorrect mode {}".format(mode))


    def reparametrize(self, file_paths, modes):

        self.file_paths = file_paths
        self.modes = modes


class ParametrizableFileReporter(FileReporter):

    PATH_TEMPLATE = "{work_dir}/{root_name}{suffix}.{extensions}"
    SUFFIX_TEMPLATE = "_{}"

    def __init__(self, root_names=None, work_dir=None, suffix=None, extensions=None,
                 **kwargs):

        # we check if we should bypass this class initialization and
        # just do the base stuff
        if self._bypass_dispatch(**kwargs):
            return None

        # otherwise we do the parametrization

        self._root_names = root_names

        if work_dir is None:
            self._work_dir = osp.realpath(osp.curdir)
        else:
            self._work_dir = work_dir

        if suffix is None:
            self._suffix = ""
        else:
            self._suffix = self.SUFFIX_TEMPLATE.format(suffix)

        self._extensions = extensions

        file_paths = []
        for file_idx, root_name in enumerate(self._root_names):
            # construct a path for this reporter
            file_path = self.PATH_TEMPLATE.format(root_name=root_name,
                                                  work_dir=osp.realpath(self._work_dir),
                                                  suffix=self._suffix,
                                                  extensions=self._extensions[file_idx].strip('.')
            )

            file_paths.append(file_path)

        super().__init__(file_paths=file_paths, **kwargs)

    @property
    def root_names(self):
        return self._root_names

    @property
    def work_dir(self):
        return self._work_dir

    @property
    def suffix(self):
        return self._suffix

    @property
    def extensions(self):
        return self._extensions


class ProgressiveFileReporter(FileReporter):
    """Super class for a reporter that will successively overwrite the
    same file over and over again. The base FileReporter really only
    supports creation of file one time.

    """

    def init(self, *args, **kwargs):

        super().init(**kwargs)

        # because we want to overwrite the file at every cycle we
        # need to change the modes to write with truncate. This allows
        # the file to first be opened in 'x' or 'w-' and check whether
        # the file already exists (say from another run), and warn the
        # user. However, once the file has been created for this run
        # we need to overwrite it many times forcefully.

        # go thourgh each file managed by this reporter
        for file_i, mode in enumerate(self.modes):

            # if the mode is 'x' or 'w-' we check to make sure the file
            # doesn't exist
            if self.mode in ['x', 'w-']:
                file_path = self.file_paths[file_i]
                if osp.exists(file_path):
                    raise FileExistsError("File exists: '{}'".format(file_path))

            # now that we have checked if the file exists we set it into
            # overwrite mode
            self.set_mode(file_i, 'w')
