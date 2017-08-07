import sys

import simtk.openmm.app as omma 
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

import scoop.futures


from wepy.sim_manager import Manager
from wepy.resampling.wexplore2 import WExplore2Resampler
from wepy.openmm import OpenMMRunner, OpenMMWalker ,OpenMMRunnerParallel
from wepy.gpumapper import GpuMapper
from wepy.gpumapper_old import GpuMapper as mapper_old
import numpy as np
import networkx as nx

from wepy.walker import Walker
from wepy.resampling.clone_merge import clone_parent_table, clone_parent_panel
from wepy.trajectory import trajectory_save

from tree import monte_carlo_minimization, make_graph

def print_initial_state(init_walkers,n_walkers):


    result_template_str = "|".join(["{:^10}" for i in range(n_walkers + 1)])

    # print the initial walkers
    print("The initial walkers:")

     # slots
    slot_str = result_template_str.format("slot", *[i for i in range(n_walkers)])
    print(slot_str)
 
    walker_weight_str = result_template_str.format("weight",
                                                    *[str(walker.weight) for walker in init_walkers])
    print(walker_weight_str)
         
def make_initial_minimized_state():
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
    print ('finished initialization')
    return minimized_state, system, topology, pdb
def make_graph_table(resampled_walkers, resampling_records):
    # write the output to a parent panel of all merges and clones within cycles
    parent_panel = clone_parent_panel(resampling_records)

    # make a table of the net parents for each cycle to a table
    parent_table = np.array(clone_parent_table(resampling_records))

    # write out the table to a csv
    np.savetxt("parents.dat", parent_table)

    # make a weights table for the walkers
    weights = []
    for cycle in resampled_walkers:
        cycle_weights = [walker.weight for walker in cycle]
        weights.append(cycle_weights)
    weights = np.array(weights)

    # create a tree visualization
    # make a distance array of equal distances from scratch
    distances = []
    for cycle in resampled_walkers:
        d_matrix = np.ones((len(cycle), len(cycle)))
        distances.append(d_matrix)
    distances = np.array(distances)

    node_positions = monte_carlo_minimization(parent_table, distances, weights, 50, debug=False)
    nx_graph = make_graph(parent_table, node_positions,
                              weight=weights)

    nx.write_gexf(nx_graph, "random_resampler_tree.gexf")
    
    
if __name__ == "__main__":

    
    minimized_state, system, topology, pdb = make_initial_minimized_state()
    # set up parameters for running the simulation
    num_workers = 8
    num_walkers = 48
    # initial weights
    init_weight = 1.0 / num_walkers
     # make a generator for the in itial walkers
    init_walkers = [OpenMMWalker(minimized_state, init_weight) for i in range(num_walkers)]
    print_initial_state(init_walkers, num_walkers)

    # set up the OpenMMRunner with your system
    runner = OpenMMRunnerParallel(system, topology)

    # set up the Resampler (with parameters if needed)

    resampler = WExplore2Resampler(reference_traj=pdb,seed=123, pmax=0.1)

    # instantiate a hpcc object
    #gpumapper  = GpuMapper(num_workers)
    gpumapper = mapper_old(num_walkers, num_workers)
    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          num_workers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=gpumapper.map)
    n_steps = 10000
    n_cycles =2

    # run a simulation with the manager for 50 cycles of length 1000 each
    steps = [ n_steps for i in range(n_cycles)]
    walker_records, resampling_records = sim_manager.run_simulation(n_cycles,
                                                                    steps,
                                                                    debug_prints=True)
    # make_graph_table(walker_records, resampling_records)
    
    # make_traj = trajectory_save(pdb.topology)

    # resampled_positions = [ [] for i in range(num_walkers)]#[init_walkers[i].positions] for i in range(num_walkers)]

    # for cycle_idx, cycles  in enumerate(walker_records):

    #     # trace each walker inside cycle
        
        
    #     for walker_idx, walker in enumerate(cycles):
            
    #         resampled_positions[walker_idx].append(walker.positions)
            
    # # length  [ 82.43499756,  82.43499756,  46.0130806 ]
    # # angles =  [  79.2771759 ,   79.2771759 ,  116.62015533],
    
    # #unitcell_lengths = np.array([pdb.unitcell_lengths[0] for i in range(n_cycles+1)])

    # #unitcell_angles = np.array([pdb.unitcell_angles[0] for i in range(n_cycles+1)])
    
    # for walker_idx in range(num_walkers):
    #     make_traj.save('traj{}.h5'.format(walker_idx), resampled_positions[walker_idx],
    #                    unitcell_lengths=None, unitcell_angles=None)
    
