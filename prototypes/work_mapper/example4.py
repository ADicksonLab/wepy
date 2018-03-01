# Originally an example from the multiprocessing docs, this makes the
# workers (known as consumers) subclasses of processes that both have
# access to the queues of the jobs (walkers in our example), thus
# there is communication between the workers.

from multiprocessing import Process, JoinableQueue, Queue
import time

PY_MAP = map

class Worker(Process):

    def __init__(self, task_queue, result_queue):

        # call the Process constructor
        Process.__init__(self)

        # the queues for work to be done and work done
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:

            # get the next task
            task_idx, next_task = self.task_queue.get()

            # check for the poison pill which is the signal to stop
            if next_task is None:


                print('{}: Exiting'.format(self.name))

                # mark the task as done
                self.task_queue.task_done()

                # and exit the loop
                break

            print('Worker: {}; task_idx : {}; args : {} '.format(self.name, task_idx, next_task.args))

            # run the task
            answer = next_task()

            # (for joinable queue) tell the queue that the formerly
            # enqued task is complete
            self.task_queue.task_done()

            # put the results into the results queue with it's task
            # index so we can sort them later
            self.result_queue.put((task_idx, answer))


class Task(object):

    def __init__(self, *args):
        self.args = args
        self.func = multiply

    def __call__(self):
        # do work that takes a long time
        print("Calling func")
        time.sleep(5)
        return self.func(*self.args)


class Mapper():

    def __init__(self, func):
        self.func = func

    def map(self, tasks):

        return list(PY_MAP(self.func, tasks))

class WorkerMapper(Mapper):

    def map(self, tasks):
        # Establish communication queues
        tasks = JoinableQueue()
        result_queue = Queue()

        # Start workers, giving them all the queues
        num_workers = 2
        print('Creating {} workers'.format(num_workers))
        workers = [ Worker(tasks, result_queue)
                      for i in range(num_workers) ]

        # start the workers
        for w in workers:
            w.start()

        # create jobs
        num_tasks = 10
        task_args = [(i, i) for i in range(num_tasks)]

        # Enqueue the jobs
        for task_idx, task_arg in enumerate(task_args):
            # a task will be the actual task and its job idx so we can
            # sort them later
            tasks.put((task_idx, Task(*task_arg)))

        # Add a poison pill (a stop signal) for each worker into the tasks
        # queue.
        for i in range(num_workers):
            tasks.put((None, None))

        # Wait for all of the tasks to finish
        tasks.join()

        # get the results out
        results = []
        while num_tasks:
            results.append(result_queue.get())
            num_tasks -= 1

        return results


# make a mapper above __main__
def multiply(a, b):
    return a*b

mapper = WorkerMapper(multiply)

if __name__ == '__main__':

    # create jobs
    num_tasks = 10
    task_args = [(i, i) for i in range(num_tasks)]

    results = mapper.map(task_args)
