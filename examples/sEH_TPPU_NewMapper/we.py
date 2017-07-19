import sys
import multiprocessing as mulproc
import threading

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

import scoop.futures

from wepy.sim_manager import Manager
from wepy.resampling.wexplore2  import WExplore2Resampler
from wepy.openmm import OpenMMRunner, OpenMMWalker ,OpenMMRunnerParallel



class hpcc:
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
        walkers_pool = [ ]
        
        # create processes and start to run 
        for args in zip(*iterables):
            p = mulproc.Process(target=self.exec_call, args=(_call_, *args))
            walkers_pool.append(p)
            p.start()
            
        for p in walkers_pool:
            p.join()

        return self.results_list
    
    
if __name__ == "__main__":


    psf = omma.CharmmPsfFile('sEH_TPPU_system.psf')

    # load the coordinates
    pdb = mdj.load_pdb('sEH_TPPU_system.pdb')

    # to use charmm forcefields get your parameters
    params = omma.CharmmParameterSet('all36_cgenff.rtf',
                                         'all36_cgenff.prm',
                                         'all36_prot.rtf',
                                         'all36_prot.prm',
                                         'tppu.str',
                                         'toppar_water_ions.str')

    # set the box size lengths and angles
    psf.setBox(82.435, 82.435, 82.435, 90, 90, 90)

        # create a system using the topology method giving it a topology and
        # the method for calculation
    system = psf.createSystem(params,
                                  nonbondedMethod=omma.CutoffPeriodic,
                                  nonbondedCutoff=1.0 * unit.nanometer,
                                  constraints=omma.HBonds)
    topology = psf.topology
        
    print("\nminimizing\n")
    # set up for a short simulation to minimize and prepare
    # instantiate an integrator
    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                            1/unit.picosecond,
                                            0.002*unit.picoseconds)
    # instantiate a OpenCL platform, in other platforms we can not have multiple simulation context 
    platform = omm.Platform.getPlatformByName('OpenCL')

     # instantiate a simulation object
    simulation = omma.Simulation(psf.topology, system, integrator,platform)
    # initialize the positions
    simulation.context.setPositions(pdb.openmm_positions(frame=0))
    # minimize the energy
    simulation.minimizeEnergy()
    # run the simulation for a number of initial time steps
    simulation.step(1000)
    print("done minimizing\n")

        # get the initial state from the context
    minimized_state = simulation.context.getState(getPositions=True,
                                                  getVelocities=True,
                                                  getParameters=True)

    # set up parameters for running the simulation
    num_workers = 8
    num_walkers = 8
    # initial weights
    init_weight = 1.0 / num_walkers
     # make a generator for the in itial walkers
    init_walkers = [OpenMMWalker(minimized_state, init_weight) for i in range(num_walkers)]

    # set up the OpenMMRunner with your system
    runner = OpenMMRunnerParallel(system, topology)

    # set up the Resampler (with parameters if needed)

    resampler = WExplore2Resampler(refrence_trajectory=pdb)

    # instantiate a hpcc object
    hpccmapper = hpcc (num_workers)

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          num_workers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper= hpccmapper.map)

    # run a simulation with the manager for 3 cycles of length 1000 each
    walker_records, resampling_records = sim_manager.run_simulation(3,
                                                                    [1000, 1000, 1000],
                                                                    debug_prints=True)
    # walker_records, resampling_records = sim_manager.run_simulation(1,
    #                                                                  [1000],
    
    #                                                                   debug_prints=True)
