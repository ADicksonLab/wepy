from __future__ import print_function
import mergeclone
import simtk.openmm.app  as app
import simtk.openmm as mm
import simtk.unit as unit
import numpy as np
import multiprocessing as mulproc
import random as rnd 
import mdtraj as md
import threading

#manager = mulproc.Manager()
#Walkers_List = manager.list()

class Walker_chr:
    positions  = None
    Walker_ID  = None
    Weight     = None
    restartpoint = None
    

class Calculate:
    def __init__(self,):
        #self.positions1 = None
        #self.positions2 = None 
        self.ref = md.load('seh_tppu_mdtraj.pdb')
        self.ref = self.ref.remove_solvent()
        self.lig_idx = self.ref.topology.select('resname "2RV"')
        self.b_selection = md.compute_neighbors(self.ref, 0.8, self.lig_idx)
        self.b_selection = np.delete(self.b_selection, self.lig_idx)
        #atom_indices = [atom.index for atom in ref.topology.atoms ]
        
    def __rmsd(self,traj,ref, idx):
        return np.sqrt(3*np.sum(np.square(traj.xyz[:,idx,:] - ref.xyz[:,idx,:]),
                                axis=(1, 2))/idx.shape[0])


    def __Make_Traj (self,Positions):
        
        Newxyz = np.zeros((1,self.ref.n_atoms, 3))
        for i in range(len(Positions)):
                  Newxyz[0,i,:] = ([Positions[i]._value[0],Positions[i]._value[1],
                                                        Positions[i]._value[2]])

        
        return md.Trajectory(Newxyz,self.ref.topology)
        
    def Calculate_Rmsd(self,positions1,positions2):
 
         #positions1 = positions1[0:self.ref.n_atoms]
         #positions2 = positions2[0:self.ref.n_atoms]
         ref_traj = self.__Make_Traj(positions1)
         traj = self.__Make_Traj(positions2)          
         traj=traj.superpose(ref_traj, atom_indices = self.b_selection)
         return self.__rmsd(traj,ref_traj,self.lig_idx)
        
 
class run_walker( mulproc.Process):
    
    def __init__(self, Params, Topology, Walker_ID, Initial):
        mulproc.Process.__init__(self)
        self.Params = Params
        self.Topology = Topology
        self.Walker_ID = Walker_ID
        self.Worker_ID = None
        self.initial  = Initial
        
    
       

    def run(self):
        Lock.acquire()
        self.Worker_ID = free_workers.get()
        print('Creating system')
        psf.setBox(82.435,82.435,82.435,90,90,90)
        system = psf.createSystem(self.Params, nonbondedMethod=app.CutoffPeriodic,
                     nonbondedCutoff=1.0* unit.nanometers, constraints=app.HBonds)

        print('Making INtegrator')

        seed=rnd.randint(0,1000000)
        integrator = mm.LangevinIntegrator(300*unit.kelvin, 1*(1/unit.picosecond), 
                                           0.002*unit.picoseconds)
        integrator.setRandomNumberSeed(seed)



        print('setting platform')

        platform = mm.Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('Precision', 'single')
        platform.setPropertyDefaultValue('DeviceIndex',str(self.Worker_ID))
        print('Making simulation object')    

        simulation = app.Simulation(self.Topology, system, integrator,platform)
        if self.initial:
            simulation.context.setPositions(pdb.positions)
            simulation.context.setVelocitiesToTemperature(300*unit.kelvin,100)
        else: 
            chk = Walkers_List[self.Walker_ID].restartpoint
            simulation.context.loadCheckpoint(chk)
           
        
        print('Minimizing ===========')    
        simulation.minimizeEnergy()
        print('End Minimizing ===========')

        #print('setting reporters')        
        #simulation.reporters.append(app.DCDReporter('traj{}.dcd'.format(Walker_ID), 1000))

        n_steps = 2000
         #n_steps = 8000000
        print('Starting to run {}'.format(n_steps))
         # 5 nano second 0.002pico 8 2.5 M step
        simulation.step(n_steps)
        print('Ending Walker {} on Worker {}'.format(self.Walker_ID,self.Worker_ID))
        # Saving Results
        item = Walkers_List[self.Walker_ID]
        item.restartpoint = simulation. context.createCheckpoint()
        item.positions = simulation.context.getState(getPositions=True ).getPositions() [ 0 :n_atoms]
        Walkers_List[self.Walker_ID]=  item 
        #print ("Weight ",Walkers_List[self.Walker_ID].Weight)

        free_workers.put(self.Worker_ID)
        Lock.release()
      #  return  state
    
if __name__ == '__main__':

    print("HELLO")
    print('Reading psf file ..')
    psf =app. CharmmPsfFile('fixed_seh.psf')

    print('Reading pdb file ..')
    pdb =app. PDBFile('seh_tppu_mdtraj.pdb')

    print('Reading ff parameters')
    params = app.CharmmParameterSet('top_all36_cgenff.rtf', 'par_all36_cgenff.prm', 
                                 'top_all36_prot.rtf', 'par_all36_prot.prm', 
                                'tppu.str', 'toppar_water_ions.str')

     
   
    n_walkers = 3
    n_workers = 2
    n_cycles = 2
    n_atoms = 5097
    initial = True
    queue = mulproc.Queue()
    manager = mulproc.Manager()
    Walkers_List = manager.list()
    initial = True 
    for i in range (n_walkers):
        new_walker = Walker_chr()
        new_walker.Walker_ID = i
        new_walker.Weight = 1 /n_walkers
        new_walker.restartpoint = None
        Walkers_List.append(new_walker)
        
    
    walkerwt = [ 1/n_walkers for i in range(n_walkers)]
    mergedist = 0.25 # 2.5 A
    # Make list of Walkers
    walkerwt=[]    
    for i in range(n_cycles):
        walkers_pool = [ run_walker(params, psf.topology, i , initial)
                         for i in range(n_walkers) ]
    
        free_workers= mulproc.Queue()
        Lock = mulproc.Semaphore (n_workers) 

        for i in range (n_workers):
            free_workers.put(i)

    
        for p in walkers_pool:
            p.start()
                   
        
        for p in walkers_pool :
            p.join()
        

        for w in Walkers_List:
            print ('Rsult ID= {} and Weight = {}\n'.format(w.Walker_ID ,w.Weight))
        initial = False
  
    
        
        a2a = np.zeros((n_walkers,n_walkers))
    
    # Calculating a2a Distance Matrix
        
        for i in range(n_walkers):
            walkerwt.append( Walkers_List[i].Weight )
            for j in range (i+1, n_walkers):
                Cal = Calculate()
                a2a[i][j] = Cal.Calculate_Rmsd(Walkers_List[i].positions, Walkers_List[j].positions)


        print (a2a)

      # merge and clone!  
    
        mcf= mergeclone.decision_maker(a2a, walkerwt, n_walkers, mergedist)
        mcf.mergeclone()
        #for i in range(n_walkers):
         #   print (' WalkerId ={ }  Weight = {}  amp= {}  parent= {} '.format( i, mcf.walkerwt[i], mcf.amp[i] , mcf#.copy_struct[i]))
     
 #   print (a2a) 

  
                

    
    
            

                
        
