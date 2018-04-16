from multiprocessing import Process
import multiprocessing as mp
import time

class Worker(Process):

    def __init__(self, worker_idx, task_queue, result_queue, debug_prints=False):

        # call the Process constructor
        Process.__init__(self)

        self.worker_idx = worker_idx

        # the queues for work to be done and work done
        self.task_queue = task_queue
        self.result_queue = result_queue

        self.debug_prints = debug_prints

    def run_task(self, task):
        return task()

    def run(self):

        if self.debug_prints:
            worker_process = mp.current_process()
            print("Worker process started as name: {}; PID: {}\n".format(worker_process.name,
                                                                       worker_process.pid))
        while True:

            # get the next task
            task_idx, next_task = self.task_queue.get()

            # # check for the poison pill which is the signal to stop
            if next_task is None:

                if self.debug_prints:
                    print('Worker: {}; received {} {}: FINISHED'.format(
                        self.name, task_idx, next_task))

                # mark the poison pill task as done
                self.task_queue.task_done()

                # and exit the loop
                break

            if self.debug_prints:
                print('Worker: {}; task_idx : {}; args : {} '.format(
                    self.name, task_idx, next_task.args))

            # run the task
            start = time.time()
            answer = self.run_task(next_task)
            end = time.time()
            task_time = end - start

            if self.debug_prints:
                print('Worker: {}; task_idx : {}; COMPLETED in {} s'.format(
                    self.name, task_idx, task_time))

            # (for joinable queue) tell the queue that the formerly
            # enqued task is complete
            self.task_queue.task_done()

            # put the results into the results queue with it's task
            # index so we can sort them later
            self.result_queue.put((task_idx, self.worker_idx, task_time, answer))


class Task(object):

    def __init__(self, func, *args):
        self.args = args
        self.func = func

    def __call__(self, **kwargs):
        # run the function passing in the args for running it and any
        # worker information in the kwargs
        return self.func(*self.args, **kwargs)
