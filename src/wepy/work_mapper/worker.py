"""Classes for workers and tasks for use with WorkerMapper. """

# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import multiprocessing as mp
import time

# First Party Library
# we can't move the WorkerMapper here until some of the pickles I have
# laying around don't expect it to be here. In the meantime, new
# software can expect it to be here so we import it here.
from wepy.work_mapper.mapper import (
    ABCWorkerMapper,
    TaskException,
    Worker,
    WorkerException,
    WorkerMapper,
    WrapperException,
)

# this whole thing should get refactored into a better name which
# should be something like ConsumerMapper because our workers act like
# consumers
