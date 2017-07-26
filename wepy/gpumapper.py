import multiprocessing as mulproc


class GpuMapper:
    def __init__(self,n_workers):
        
        self.free_workers= mulproc.Queue()
        self.lock = mulproc.Semaphore (n_workers)
        self.results_list = mulproc.Manager().list()
        for i in range (n_workers):         
            self.free_workers.put(i)
            
    def exec_call (self,_call_, *args):
        # gets a free GUP and calls the runable function 
        self.lock.acquire()
        gpuindex = self.free_workers.get()
        args += (gpuindex,)
        result = _call_(*args)
        self.free_workers.put(gpuindex)
        self.lock.release()
        self.results_list.append(result)
        
    def map(self,_call_, *iterables):
        walkers_pool = []
        
        # create processes and start to run 
        for args in zip(*iterables):
            walkers_pool.append(mulproc.Process(target=self.exec_call, args=(_call_, *args)))
            
        for p in walkers_pool:
            p.start()
            
        for p in walkers_pool:
            p.join()
            
        return self.results_list
