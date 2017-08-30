class Reporter(object):

    def __init__(self):
        pass

    def init(self):
        pass

    def report(*args, **kwargs):
        pass

class FileReporter(Reporter):

    def __init__(self, file_path, mode='x'):
        self.file_path = file_path
        self.mode = mode

    def init(self):
        pass

    def report(self):
        pass


class ObjectReporter(Reporter):

    def __init__(self):
        pass

    def init(self):
        pass

    def report(self, *args):
        return args

class PrintReporter(Reporter):

    def __init__(self):
        pass

    def init(self):
        pass

    def report(*args):
        for arg in args:
            print(arg)
