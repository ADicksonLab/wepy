
# these are the supported reporter types for reparametrizing
from wepy.reporter.reporter import FileReporter

# supported work mapper types
from wepy.work_mapper import WorkerMapper

class Configuration():

    SUPPORTED_REPORTERS = (FileReporter,)
    SUPPORTED_WORK_MAPPERS = (WorkerMapper,)

    def __init__(self, work_mapper=None, reporters=None):

        if work_mapper is not None:
            self._work_mapper = work_mapper
        else:
            # set as a default
            self._work_mapper = None

        if reporters is not None:
            self._reporters = reporters
        else:
            self._reporters = []

    @property
    def reporters(self):
        return deepcopy(self._reporters)

    @property
    def work_mapper(self):
        return deepcopy(self._work_mapper)


    def reparametrize_file_reporters(self, ):

        pass

    def reparametrize_reporters(self, ):

        new_reporters = []
        # go through the reporters, we access directly because we will
        # make copies ourselves
        for reporter in self._reporters:

            reporter_class = type(reporter)

            # check for the different possible cases and reparametrize
            # them accordingly
            if issubclass(reporter_class, FileReporter):
                new_reporter = self.reparametrize_file_reporter(reporter, )

            # if it is not any of the supported reporters we just copy
            # it
            else:
                new_reporter = deepcopy(reporter)

            new_reporters.append(new_reporter)

        return new_reporters

    def reparametrize(self, work_mapper_kwargs, reporter_kwargs):
        pass
