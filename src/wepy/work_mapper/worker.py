"""Classes for workers and tasks for use with WorkerMapper. """

import multiprocessing as mp
import time
import logging
from wepy.work_mapper.mapper import ABCWorkerMapper, WrapperException, TaskException

# this whole thing should get refactored into a better name which
# should be something like ConsumerMapper because our workers act like
# consumers



# we can't move the WorkerMapper here until some of the pickles I have
# laying around don't expect it to be here. In the meantime, new
# software can expect it to be here so we import it here.
from wepy.work_mapper.mapper import WorkerMapper, Worker, WorkerException

