import sys
from collections import namedtuple

import numpy as np
import pandas as pd

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import scoop.futures

import mdtraj as mdj

from wepy.sim_manager import Manager
from wepy.resampling.clone_merge import RandomCloneMergeResampler
from wepy.openmm import OpenMMRunner, OpenMMWalker
from wepy.resampling.clone_merge import clone_parent_table, clone_parent_panel

from resampling_tree.tree import monte_carlo_minimization, make_graph

from wepy_hdf5 import WepyHDF5, load_topo_dataset

if __name__ == "__main__":

    # write out an hdf5 storage of the system
    mdj_pdb = mdj.load_pdb('sEH_TPPU_system.pdb')
    mdj_pdb.save_hdf5('sEH_TPPU_system.h5')

    # set up the system which will be passed to the arguments of the mapped functions
    # the topology from the PSF
    psf = omma.CharmmPsfFile('sEH_TPPU_system.psf')

    # load the coordinates
    omm_pdb = omma.PDBFile('sEH_TPPU_system.pdb')

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

    print("\nminimizing\n")
    # set up for a short simulation to minimize and prepare
    # instantiate an integrator
    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                        1/unit.picosecond,
                                        0.002*unit.picoseconds)
    # instantiate a simulation object
    simulation = omma.Simulation(psf.topology, system, integrator)
    # initialize the positions
    simulation.context.setPositions(omm_pdb.positions)
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
    # make a generator for the initial walkers
    init_walkers = [OpenMMWalker(minimized_state, init_weight) for i in range(num_walkers)]

    # make a template string for pretty printing results as we go
    result_template_str = "|".join(["{:^10}" for i in range(num_walkers + 1)])

    # print the initial walkers
    print("The initial walkers:")
    # slots
    slot_str = result_template_str.format("slot", *[i for i in range(num_walkers)])
    print(slot_str)
    # weights
    walker_weight_str = result_template_str.format("weight",
        *[str(init_weight) for i in range(num_walkers)])
    print(walker_weight_str)

    # set up the OpenMMRunner with your system
    runner = OpenMMRunner(system, psf.topology)


    # set up the RandomResampler with the same random seed
    seed = 3247862378
    resampler = RandomCloneMergeResampler(3247862378)

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          num_workers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=scoop.futures.map)


    # run a simulation with the manager for 3 cycles of length 1000 each
    walker_records, resampling_records = sim_manager.run_simulation(3,
                                                                    [1000, 1000, 1000],
                                                                    debug_prints=True)

    # write the output to a parent panel of all merges and clones within cycles
    parent_panel = clone_parent_panel(resampling_records)

    # make a table of the net parents for each cycle to a table
    parent_table = np.array(clone_parent_table(resampling_records))

    # write out the table to a csv
    np.savetxt("parents.dat", parent_table, fmt='%i')

    # save the trajectories as hdf5 trajectories
    walker_trajs = [walker_records[0]]
    # the first one is the initial walkers
    for cycle_idx, cycle in walker_records[1:]:

    # make a dataframe out out of the resampling records
    # make a new record with all the info we need across the run
    DFResamplingRecord = namedtuple("DFResamplingRecord", ['cycle_idx', 'step_idx', 'walker_idx',
                                              'decision', 'instruction'])
    # make these records
    df_recs = []
    for cycle_idx, cycle in enumerate(resampling_records):
        # each cycle has multiple steps of resampling
        for step_idx, step in enumerate(cycle):
            for walker_idx, rec in enumerate(step):
                df_rec = DFResamplingRecord(cycle_idx=cycle_idx,
                                            step_idx=step_idx,
                                            walker_idx=walker_idx,
                                            decision=rec.decision.name,
                                            instruction=rec.instruction)
                df_recs.append(df_rec)
    # make a dataframe from them and write it out
    resampling_df = pd.DataFrame(df_recs)
    resampling_df.to_csv("resampling.csv")


    ## tree visualization
    # make a weights table for the walkers
    weights = []
    for cycle in walker_records:
        cycle_weights = [walker.weight for walker in cycle]
        weights.append(cycle_weights)
    weights = np.array(weights)

    # create a tree visualization
    # make a distance array of equal distances from scratch
    distances = []
    for cycle in walker_records:
        d_matrix = np.ones((len(cycle), len(cycle)))
        distances.append(d_matrix)
    distances = np.array(distances)

    node_positions = monte_carlo_minimization(parent_table, distances, weights, 50, debug=True)
    nx_graph = make_graph(parent_table, node_positions,
                              weight=weights)

    nx.write_gexf(nx_graph, "random_resampler_tree.gexf")
