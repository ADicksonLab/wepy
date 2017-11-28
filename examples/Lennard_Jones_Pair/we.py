import sys
from copy import copy

import numpy as np

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from openmmtools.testsystems import LennardJonesPair
import mdtraj as mdj

from wepy.sim_manager import Manager
from wepy.resampling.wexplore2 import WExplore2Resampler
from wepy.openmm import OpenMMRunner, OpenMMWalker
from wepy.openmm import UNIT_NAMES, GET_STATE_KWARG_DEFAULTS
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.reporter.hdf5 import WepyHDF5Reporter

import scoop.futures

if __name__ == "__main__":

    n_runs = int(sys.argv[1])
    n_steps = int(sys.argv[2])
    n_cycles = int(sys.argv[3])

    test_sys = LennardJonesPair()

#    integrator = omm.VerletIntegrator(2*unit.femtoseconds)
    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                        1/unit.picosecond,
                                        0.002*unit.picoseconds)

    context = omm.Context(test_sys.system, copy(integrator))
    context.setPositions(test_sys.positions)

    get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
    init_state = context.getState(**get_state_kwargs)

    thermostat = omm.AndersenThermostat(300.0 * unit.kelvin, 1/unit.picosecond)
    barostat = omm.MonteCarloBarostat(1.0*unit.atmosphere, 300.0*unit.kelvin, 50)

    runner = OpenMMRunner(test_sys.system, test_sys.topology, integrator, platform='Reference')

    num_walkers = 10
    init_weight = 1.0 / num_walkers

    init_walkers = [OpenMMWalker(init_state, init_weight) for i in range(num_walkers)]

    mdtraj_topology = mdj.Topology.from_openmm(test_sys.topology)
    resampler = WExplore2Resampler(topology=mdtraj_topology,
                                   ligand_idxs=np.array(test_sys.ligand_indices),
                                   binding_site_idxs=np.array(test_sys.receptor_indices),
                                   pmax=1.0,
                                   merge_dist=100.)

    ubc = UnbindingBC(cutoff_distance=0.5,
                      initial_state=init_walkers[0].state,
                      topology=mdtraj_topology,
                      ligand_idxs=np.array(test_sys.ligand_indices),
                      binding_site_idxs=np.array(test_sys.receptor_indices))

    json_top_path = 'pair.top.json'
    with open(json_top_path, 'r') as rf:
        json_str_top = rf.read()

    # make a dictionary of units for adding to the HDF5
    units = dict(UNIT_NAMES)

    report_path = 'results.wepy.h5'
    # open it in truncate mode first, then switch after first run
    reporter = WepyHDF5Reporter(report_path, mode='w',
                                decisions=resampler.DECISION,
                                instruction_dtypes=resampler.INSTRUCTION_DTYPES,
                                warp_dtype=ubc.WARP_INSTRUCT_DTYPE,
                                warp_aux_dtypes=ubc.WARP_AUX_DTYPES,
                                warp_aux_shapes=ubc.WARP_AUX_SHAPES,
                                bc_dtype=None,
                                bc_aux_dtypes=None,
                                bc_aux_shapes=None,
                                topology=json_str_top,
                                units=units)

    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=ubc,
                          work_mapper=map,
                          reporter=reporter)

    print("Number of steps: {}".format(n_steps))
    print("Number of cycles: {}".format(n_cycles))

    steps = [n_steps for i in range(n_cycles)]
    print("Running Simulations")

    for run_idx in range(n_runs):
        print("Starting run: {}".format(run_idx))
        sim_manager.run_simulation(n_cycles, steps, debug_prints=True)
        print("Finished run: {}".format(run_idx))

    print("Finished first file")


    print("Starting second file")
    # make a separate file with equal runs
    second_report_path = 'results_2.wepy.h5'
    # open it in truncate mode first, then switch after first run
    second_reporter = WepyHDF5Reporter(second_report_path, mode='w',
                                decisions=resampler.DECISION,
                                instruction_dtypes=resampler.INSTRUCTION_DTYPES,
                                warp_dtype=ubc.WARP_INSTRUCT_DTYPE,
                                warp_aux_dtypes=ubc.WARP_AUX_DTYPES,
                                warp_aux_shapes=ubc.WARP_AUX_SHAPES,
                                bc_dtype=None,
                                bc_aux_dtypes=None,
                                bc_aux_shapes=None,
                                topology=json_str_top,
                                units=units)

    second_sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=ubc,
                          work_mapper=map,
                          reporter=second_reporter)

    print("Running Simulations")

    for run_idx in range(n_runs):
        print("Starting run: {}".format(run_idx))
        second_sim_manager.run_simulation(n_cycles, steps, debug_prints=True)
        print("Finished run: {}".format(run_idx))

    print("Finished second file")
