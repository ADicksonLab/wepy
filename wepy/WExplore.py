# standard library imports
import multiprocessing as mulproc
import random as rnd

# external library imports
import numpy as np
import mdtraj as md
import simtk.openmm.app  as app
import simtk.openmm as mm
import simtk.unit as unit

# internal imports
import wepy.mergeclone

# should be using python3 only, this is for if you want to support
# python2, there are other things you have to do to support it though,
# so it is probably better to just fail quicker if a user tries to use
# python2
# from __future__ import print_function

#manager = mulproc.Manager()
#Walkers_List = manager.list()

class Walker_chr:
    positions = None
    walker_id = None
    weight = None
    restart_point = None


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

