import multiprocessing as mulproc

class GpuMapper:
    def __init__(self,n_walkers, n_workers ):

        self.free_workers= mulproc.Queue()
        self.lock = mulproc.Semaphore (n_workers)
        self.results_list = mulproc.Manager().list()
        for i in range(n_walkers):
            self.results_list.append(None)
        for i in range (n_workers):
            self.free_workers.put(i)

    def exec_call(self,_call_, index, *args):
        # gets a free GUP and calls the runable function
        self.lock.acquire()
        gpu_index = self.free_workers.get()
        #args += (gpuindex,)
        result = _call_(*args, gpu_index=gpu_index)
        # check for validation of MD simulation
        #if result is None:
         #  self.tasks.put(mulproc.Process(target=self.exec_call, args=(_call_,index,*args)))

        #else :
        self.free_workers.put(gpu_index)
        self.results_list[index] = result
        self.lock.release()


    def map(self,_call_, *iterables):

        walkers_pool = []
        # create processes and start to run
        index = 0

        for args in zip(*iterables):
            walkers_pool.append (mulproc.Process(target=self.exec_call, args=(_call_,index,*args)))
            index += 1

        for p in walkers_pool:
            p.start()

        for p in walkers_pool:
            p.join()


        return self.results_list


# ___________________________________________ new mapper
class task_runner(mulproc.Process):
    def __init__(self, task_queue, result_list, gpu_index):
        super(mulproc.Process, self).__init__()
        self.task_queue = task_queue
        self.result_list = result_list
        self.gpu_index = gpu_index

    def run(self):
        while self.task_queue.qsize() > 0:
            next_task = self.task_queue.get()

            index = next_task[1]
            arg1 = next_task[2]
            arg2 = next_task[3]
            print (self.gpu_index)
            answer = next_task[0](arg1,arg2, self.gpu_index)
            if answer == None:
                self.task_queue.task_done()
                self.task_queue.put(next_task)
                break
            self.result_list[index] = answer
            self.task_queue.task_done()
        return

class GpuMapper_2:
    def __init__(self,workers_idx):
        self.workers_idx = workers_idx

    def map(self,_call_, *iterables):

        tasks = mulproc.JoinableQueue()
        results = mulproc.Manager().list()

        # create processes and start to run
        index = 0
        processor_pool = [task_runner(tasks, results, i) for i in range(self.workers_idx)]


        for args in zip(*iterables):
            tasks.put ((_call_,index,*args))
            index += 1
            results.append(None)

        for p in processor_pool:
            p.start()

        # waite for all the tasks to finish
        tasks.join()

        return results
