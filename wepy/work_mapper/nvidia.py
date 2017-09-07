import pynvml as nvml

class NVMLMapper(object):

    def __init__(self, uuids=None):

        nvml.nvmlInit()
        # if there were no specific UUIDs given just use the available GPUS
        if uuids is None:
            n_gpus = nvml.nvmlDeviceGetCount()
            self.handles = [nvml.nvmlDeviceGetHandleByIndex(idx) for idx in range(n_gpus)]
            #self.uuids = [nvml.nvmlDeviceGetUUID(handle) for handle in self.handles]
        # if specific UUIDs were given get the handles for them
        else:
            self.handles = [nvml.nvmlDeviceGetHandleByUUID(uuid) for uuid in uuids]

    def map(self, func, *iterables):
        pass
