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

    SUGGESTED_FILENAME_TEMPLATE = "{config}{narration}{reporter_class}.{ext}"

    DEFAULT_SUGGESTED_EXTENSION = 'report'

    FILE_ORDER = ()
    SUGGESTED_EXTENSIONS = ()

    def __init__(self, file_paths=None, modes=None,
                 file_path=None, mode=None,
                 **kwargs):

        # file paths

        assert not ((file_paths is not None) and (file_path is not None)), \
            "only file_paths or file_path kwargs can be specified"

        # if only one file path is given then we handle it as multiple
        if file_path is not None:
            file_paths = [file_path]


        # if any of the explicit paths are given (from the FILE_ORDER
        # constant) in the kwargs then we automatically add those to
        # the file_paths being sent to the super class constructor.

        # we use a flag to condition this, initialize and fall back to
        # using the 'file_paths' kwarg
        use_explicit_path_kwargs = False

        # we check the kwargs for the explicit file kwargs, and if
        # they are given then we check whether they are valid and if
        # they are, use them to set the 'file_paths' kwarg

        # make a list of the presence of the given explicit keys
        given_explicit_kwargs = [(True if file_key in kwargs else False)
                                for file_key in self.FILE_ORDER]

        # check that all the keys are present, if they aren't all
        # present then the flag will stay false and the fallback of
        # using the 'file_paths' kwarg will be used
        if all(given_explicit_kwargs):

            # then get the values and check them
            valid_explicit_kwargs = [(True if kwargs[file_key] is not None else False)
                                     for file_key in self.FILE_ORDER]


            # if they are all valid then we can use them
            if not all(valid_explicit_kwargs):

                use_explicit_path_kwargs = True

        # if only some were given this is wrong
        elif any(given_explicit_kwargs):

            raise ValueError("If you explicitly pass in the paths, all must be given explicitly")



        # if we use the explicit path kwargs, then we need to put them
        # into the 'file_paths' for superclass initialization
        if use_explicit_path_kwargs:

                file_paths = []
                for file_key in self.FILE_ORDER:

                    # add it to the file paths for superclass initialization
                    file_paths.append(kwargs[file_key])

        # otherwise we need to use the file_paths argument that should
        # have been given
        else:

            # make sure it is in kwargs and valid
            assert file_paths is not None, \
                "if no explicit file path is given the 'file_paths' must have a value"

            assert len(file_paths) == len(self.FILE_ORDER), \
                "you must give file_paths {} paths".format(len(self.FILE_ORDER))

        # using the file_path paths we got above we set them as
        # attributes in this object
        for i, file_key in enumerate(self.FILE_ORDER):
            setattr(self, file_key, file_paths[i])


        # set the underlying file paths
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

class ProgressiveFileReporter(FileReporter):
    """Super class for a reporter that will successively overwrite the
    same file over and over again. The base FileReporter really only
    supports creation of file one time.

    """

    def init(self, **kwargs):

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
            if mode in ['x', 'w-']:
                file_path = self.file_paths[file_i]
                if osp.exists(file_path):
                    raise FileExistsError("File exists: '{}'".format(file_path))

            # now that we have checked if the file exists we set it into
            # overwrite mode
            self.set_mode(file_i, 'w')
