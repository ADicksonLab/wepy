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
from wepy.boundary_conditions.unbinding import UnbindingBC
from wepy.reporter.hdf5 import WepyHDF5Reporter

import scoop.futures

if __name__ == "__main__":

    n_steps = int(sys.argv[1])
    n_cycles = int(sys.argv[2])

    try:
        platform = sys.argv[3]
    except IndexError:
        platform = False

    test_sys = LennardJonesPair()

    integrator = omm.VerletIntegrator(2*unit.femtoseconds)
    context = omm.Context(test_sys.system, copy(integrator))
    context.setPositions(test_sys.positions)
    init_state = context.getState(getPositions=True,
                                  getVelocities=True,
                                  getParameters=True,
                                  getForces=True,
                                  getEnergy=True,
                                  getParameterDerivatives=True)

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
                                   pmax=0.1)

    ubc = UnbindingBC(cutoff_distance=0.5,
                      initial_state=init_walkers[0],
                      topology=mdtraj_topology,
                      ligand_idxs=np.array(test_sys.ligand_indices),
                      binding_site_idxs=np.array(test_sys.receptor_indices))

    json_top_path = 'pair.top.json'
    with open(json_top_path, 'r') as rf:
        json_str_top = rf.read()

    report_path = 'wepy_results.h5'
    reporter = WepyHDF5Reporter(report_path, mode='w',
                                decisions=resampler.DECISION,
                                instruction_dtypes=resampler.INSTRUCTION_DTYPES,
                                warp_dtype=ubc.WARP_INSTRUCT_DTYPE,
                                warp_aux_dtypes=ubc.WARP_AUX_DTYPES,
                                warp_aux_shapes=ubc.WARP_AUX_SHAPES,
                                bc_dtype=None,
                                bc_aux_dtypes=None,
                                bc_aux_shapes=None,
                                topology=json_str_top)

    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=ubc,
                          work_mapper=map,
                          reporter=reporter)

    print("Number of steps: {}".format(n_steps))
    print("Number of cycles: {}".format(n_cycles))

    steps = [n_steps for i in range(n_cycles)]
    print("Running Simulation")
    sim_manager.run_simulation(n_cycles, steps, debug_prints=True)
