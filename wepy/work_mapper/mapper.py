from multiprocessing import Queue, JoinableQueue

from wepy.work_mapper.worker import Worker, Task

PY_MAP = map

class Mapper(object):

    def __init__(self, func, *args, **kwargs):
        self.func = func

    def map(self, *args):

        return list(PY_MAP(self.func, *args))

class WorkerMapper(Mapper):

    def __init__(self, func, num_workers, worker_type=None):
        self.func = func
        self.num_workers = num_workers
        if worker_type is None:
            self.worker_type = Worker
        else:
            self.worker_type = worker_type


    def make_task(self, *args, **kwargs):
        return Task(self.func, *args, **kwargs)

    def map(self, *args, debug_prints=False):
        # Establish communication queues
        tasks = JoinableQueue()
        result_queue = Queue()

        # Start workers, giving them all the queues
        workers = [ self.worker_type(i, tasks, result_queue, debug_prints=debug_prints)
                        for i in range(self.num_workers) ]

        # start the workers
        for w in workers:
            w.start()

        # make tuples for the arguments to each function call
        task_args = zip(*args)

        num_tasks = len(args[0])
        # Enqueue the jobs
        for task_idx, task_arg in enumerate(task_args):

            # a task will be the actual task and its task idx so we can
            # sort them later
            tasks.put((task_idx, self.make_task(*task_arg)))

        # Add a poison pill (a stop signal) for each worker into the tasks
        # queue.
        for i in range(self.num_workers):
            tasks.put((None, None))

        # Wait for all of the tasks to finish
        print("Waiting for tasks to be run")
        tasks.join()

        # get the results out in an unordered way
        results = []
        while num_tasks:
            results.append(result_queue.get())
            num_tasks -= 1

        # sort the results according to their task_idx
        results.sort()

        # then just return the values of the function
        return [result for task_idx, worker_idx, result in results]
