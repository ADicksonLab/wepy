import os
import os.path as osp
import pickle
import logging

class ReporterError(Exception):
    """ """
    pass

class Reporter(object):
    """Abstract base class for wepy reporters.

    All reporters must customize and override minimally the 'report'
    method. Optionally the 'init' and 'cleanup' can be overriden.

    See Also
    --------

    wepy.sim_manager : details of calls to reporter methods.

    """


    def __init__(self, **kwargs):
        """Construct a reporter.

        Void constructor for the Reporter base class.

        Parameters
        ----------
        **kwargs : key-value pairs
            Ignored kwargs, but accepts them from subclass calls for
            compatibility.

        """
        pass

    def init(self, **kwargs):
        """Initialization routines for the reporter at simulation runtime.

        Initialize I/O connections including file descriptors,
        database connections, timers, stdout/stderr etc.

        Void method for reporter base class.

        Reporters can expect to have the following key word arguments
        passed to them during a simulation by the sim_manager in this
        call.


        Parameters
        ----------

        init_walkers : list of Walker objects
            The initial walkers for the simulation.

        runner : Runner object
            The runner that will be used in the simulation.

        resampler : Resampler object
            The resampler that will be used in the simulation.

        boundary_conditions : BoundaryConditions object
            The boundary conditions taht will be used in the simulation.

        work_mapper : WorkMapper object
            The work mapper that will be used in the simulation.

        reporters : list of Reporter objects
            The list of reporters that are in the simulation.

        continue_run : int
            The index of the run that is being continued within this
            same file.

        """
        method_name = 'init'
        assert not hasattr(super(), method_name), \
            "Superclass with method {} is masked".format(method_name)

    def report(self, **kwargs):
        """Given data concerning the main simulation components state, perform
        I/O operations to persist that data.

        Void method for reporter base class.

        Reporters can expect to have the following key word arguments
        passed to them during a simulation by the sim_manager.

        Parameters
        ----------

        cycle_idx : int

        new_walkers : list of Walker objects
            List of walkers that were produced from running their
            dynamics by the runner.

        warp_data : list of dict of str : value
            List of dict-like records for each warping event from the
            last cycle.

        bc_data : list of dict of str : value
           List of dict-like records specifying the changes to the
           state of the boundary conditions in the last cycle.

        progress_data : dict str : list
            A record indicating the progress values for each walker in
            the last cycle.

        resampling_data : list of dict of str : value
            List of records specifying the resampling to occur at this
            cycle.

        resampler_data : list of dict of str : value
            List of records specifying the changes to the state of the
            resampler in the last cycle.

        n_segment_steps : int
            The number of dynamics steps that were completed in the last cycle

        worker_segment_times : dict of int : list of float
            Mapping worker index to the times they took for each
            segment they processed.

        cycle_runner_time : float
            Total time runner took in last cycle.

        cycle_bc_time : float
            Total time boundary conditions took in last cycle.

        cycle_resampling_time : float
            Total time resampler took in last cycle.

        resampled_walkers : list of Walker objects
            List of walkers that were produced from the new_walkers
            from applying resampling and boundary conditions.

        """

        method_name = 'report'
        assert not hasattr(super(), method_name), \
            "Superclass with method {} is masked".format(method_name)

    def cleanup(self, **kwargs):
        """Teardown routines for the reporter at the end of the simulation.

        Use to cleanly and safely close I/O connections or other
        cleanup I/O.

        Use to close file descriptors, database connections etc.

        Reporters can expect to have the following key word arguments
        passed to them during a simulation by the sim_manager.

        Parameters
        ----------

        runner : Runner object
            The runner at the end of the simulation

        work_mapper : WorkeMapper object
            The work mapper at the end of the simulation

        resampler : Resampler object
            The resampler at the end of the simulation

        boundary_conditions : BoundaryConditions object
            The boundary conditions at the end of the simulation

        reporters : list of Reporter objects
            The list of reporters at the end of the simulation

        """
        method_name = 'cleanup'
        assert not hasattr(super(), method_name), \
            "Superclass with method {} is masked".format(method_name)


class FileReporter(Reporter):
    """Abstract reporter that handles specifying file paths for a
    reporter.

    This abstract class doesn't perform any operations that involve
    actually opening file descriptors, but only the validation and
    organization of file paths.

    This provides a uniform API for retrieving file paths from all
    reporters inheriting from it.

    Additionally, FileReporter implements an interface for performing
    a so-called reparametrization of the relevant values associated
    with each file specification (i.e. file path and mode).

    A reparametrization can be performed by calling the
    'reparametrize' method, and can be customized.

    Additionally, there are some customizable class constants than can
    be used in subclasses to control this process including:
    DEFAULT_MODE, SUGGESTED_FILENAME_TEMPLATE,
    DEFAULT_SUGGESTED_EXTENSION, FILE_ORDER, and SUGGESTED_EXTENSIONS.

    The intention is to allow the redefinition of file paths
    dynamically to adapt to changing runtime requirements. Such as
    execution on a separate subtree of a directory hierarchy.

    """

    MODES = ('x', 'w', 'w-', 'r', 'r+',)
    """Valid modes accepted for files."""

    DEFAULT_MODE = 'x'
    """The default mode to set for opening files if none is specified
    (create if doesn't exist, fail if it does.)"""


    SUGGESTED_FILENAME_TEMPLATE = "{config}{narration}{reporter_class}.{ext}"
    """Template to use for dynamic reparametrization of file path names.

    The fields in the template are:

    config : indicator of the runtime configuration used

    narration : freeform description of the instance

    reporter_class : the name of the class that produced the
        output. When no specific name is given for a file report generated
        from a reporter this is used to disambiguate, along with the
        extension.

    ext : The file extension, for multiple files produced from one
    reporter this should be sufficient to disambiguate the files.

    The 'config' and 'narration' should be the same across all
    reporters in the same simulation manager, and the 'narration' is
    considered optional.

    """

    DEFAULT_SUGGESTED_EXTENSION = 'report'
    """The default file extension used for files during dynamic
    reparametrization, if none is specified"""

    FILE_ORDER = ()
    """Specify an ordering of file paths. Should be customized."""

    SUGGESTED_EXTENSIONS = ()
    """Suggested extensions for file paths for use with the automatic
    reparametrization feature. Should be customized."""

    def __init__(self, file_paths=None, modes=None,
                 file_path=None, mode=None,
                 **kwargs):
        """Constructor for FileReporter.

        This constructor allows the specification of either a list of
        file names (and modes) via 'file_paths' and 'modes' key-word
        arguments or a single 'file_path' and 'mode'.

        The access API though is always a list of file paths and modes
        where order is important for associating other features.

        Parameters
        ----------

        file_paths : list of str
            The list of file paths (in order) to use.

        modes : list of str
            The list of mode specs (in order) to use.

        file_path : str
            If 'file_paths' not specified, the single file path to use.

        mode : str
            If 'file_path' option used, this is the mode for that file.

        """

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


    def _validate_mode(self, mode):
        """Check if the mode spec is a valid one.

        Parameters
        ----------
        mode : str

        Returns
        -------
        valid : bool

        """
        if mode in self.MODES:
            return True
        else:
            return False

    @property
    def mode(self):
        """For single file path reporters the mode of that file."""
        if len(self._file_paths) > 1:
            raise ReporterError("there are multiple files and modes defined")

        return self._modes[0]

    @property
    def file_path(self):
        """For single file path reporters the file path to that file spec."""
        if len(self._file_paths) > 1:
            raise ReporterError("there are multiple files and modes defined")

        return self._file_paths[0]

    @property
    def file_paths(self):
        """The file paths for this reporter, in order."""
        return self._file_paths

    @file_paths.setter
    def file_paths(self, file_paths):
        """Setter for the file paths.

        Parameters
        ----------
        file_paths : list of str

        """
        for i, file_path in enumerate(file_paths):
            self.set_path(i, file_path)

    def set_path(self, file_idx, path):
        """Set the path for a single indexed file.

        Parameters
        ----------
        file_idx : int
            Index in the listing of files.
        path : str
            The new path to set for this file

        """
        self._paths[file_idx] = path

    @property
    def modes(self):
        """The modes for the files, in order."""
        return self._modes

    @modes.setter
    def modes(self, modes):
        """Setter for the modes.

        Parameters
        ----------
        modes : list of str

        """
        for i, mode in enumerate(modes):
            self.set_mode(i, mode)

    def set_mode(self, file_idx, mode):
        """Set the mode for a single indexed file.

        Parameters
        ----------
        file_idx : int
            Index in the listing of files.
        mode : str
            The new mode spec.

        """

        if self._validate_mode(mode):
            self._modes[file_idx] = mode
        else:
            raise ValueError("Incorrect mode {}".format(mode))


    def reparametrize(self, file_paths, modes):
        """Set the file paths and modes for all files in the reporter.

        Parameters
        ----------
        file_paths : list of str
            New file paths for each file, in order.
        modes : list of str
            New modes for each file, in order.

        """

        self.file_paths = file_paths
        self.modes = modes

class ProgressiveFileReporter(FileReporter):
    """Super class for a reporter that will successively overwrite the
    same file over and over again. The base FileReporter really only
    supports creation of file one time.

    """

    def init(self, **kwargs):
        """Construct a ProgressiveFileReporter.

        This is exactly the same as the FileReporter.


        Parameters
        ----------

        file_paths : list of str
            The list of file paths (in order) to use.

        modes : list of str
            The list of mode specs (in order) to use.

        file_path : str
            If 'file_paths' not specified, the single file path to use.

        mode : str
            If 'file_path' option used, this is the mode for that file.

        See Also
        --------

        wepy.reporter.reporter.FileReporter

        """

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
