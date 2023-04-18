"""
Tests the ability to use openmm functions to build a system suitable for wepy simulation.
"""

import openmm as omm
import openmm.app as omma
import pickle as pkl

import os.path as osp
import os

from wepy.runners.openmm import gen_sim_state, OpenMMRunner, OpenMMState, OpenMMWalker, OpenMMCPUWalkerTaskProcess, OpenMMGPUWalkerTaskProcess
from wepy.resampling.distances.receptor import UnbindingDistance
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.boundary_conditions.receptor import UnbindingBC
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.work_mapper.task_mapper import TaskMapper
from wepy.sim_manager import Manager

def test_run_openmm_reference():
    run_openmm_system('Reference')

def test_run_openmm_cpu():
    run_openmm_system('CPU')

def test_run_openmm_CUDA():
    run_openmm_system('CUDA')
    
def run_openmm_system(platform,cleanup=True):

    box_len = 2.0 # nm
    temp = 303.25*omm.unit.kelvin
    pressure = 1.0*omm.unit.atmosphere
    fric = 1.0/omm.unit.picosecond
    dt = 0.002*omm.unit.picosecond
    
    base_folder = 'src/pytest_wepy/test_data'
        
    with open(osp.join(base_folder,'system.pkl'),'rb') as f:
        system = pkl.load(f)

    with open(osp.join(base_folder,'topology.pkl'),'rb') as f:
        omm_top = pkl.load(f)

    ion_idx1 = 738
    ion_idx2 = 739
        
    crd = omma.CharmmCrdFile(osp.join(base_folder,'step3_input.crd'))
    pos = crd.positions

    # make an integrator object that is constant temperature
    integrator = omm.LangevinIntegrator(temp, fric, dt)

    # generate a new simtk "state"
    new_simtk_state = gen_sim_state(pos, system, integrator)
    
    # set up the OpenMMRunner with your system
    runner = OpenMMRunner(system, omm_top, integrator, platform=platform)
    
    # set up parameters for running the simulation
    num_walkers = 4
    init_weight = 1.0 / num_walkers

    # generate the walker state in wepy format
    walker_state = OpenMMState(new_simtk_state)
        
    # make a list of the initial walkers
    init_walkers = [OpenMMWalker(walker_state, init_weight) for i in range(num_walkers)]

    # distance metric to be used in REVO
    unb_distance = UnbindingDistance([ion_idx1],
                                     [ion_idx2],
                                     walker_state)

    # set up the REVO Resampler with the parameters
    resampler = REVOResampler(distance=unb_distance,
                              init_state=walker_state,
                              weights=True,
                              pmax=1.0,
                              dist_exponent=4,
                              merge_dist=0.25,
                              char_dist=0.1)

    # set up the boundary conditions for a non-eq ensemble
    ubc = UnbindingBC(cutoff_distance=1.0,  # nm
                      initial_state=walker_state,
                      ligand_idxs=[ion_idx1],
                      receptor_idxs=[ion_idx2])

    h5_name = osp.join(base_folder,'tmp.wepy.results.h5')
    if osp.exists(h5_name): os.remove(h5_name)

    # set up the HDF5 reporter
    hdf5_reporter = WepyHDF5Reporter(save_fields=('positions','box_vectors'),
                                     file_path=h5_name,
                                     resampler=resampler,
                                     boundary_conditions=ubc,
                                     n_atoms=len(pos))

    if platform in ['Reference','CPU']:
        mapper = TaskMapper(walker_task_type=OpenMMCPUWalkerTaskProcess,
                            num_workers=2,
                            platform=platform)
    elif platform in ['OpenCL','CUDA']:
        mapper = TaskMapper(walker_task_type=OpenMMGPUWalkerTaskProcess,
                            num_workers=2,
                            platform=platform,
                            device_ids=[0,1])
        
    # build the simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=ubc,
                          work_mapper=mapper,
                          reporters=[hdf5_reporter])

    n_steps = 10
    n_cycles = 2

    # run a simulation with the manager for n_steps cycles of length 1000 each
    steps_list = [n_steps for i in range(n_cycles)]

    # and..... go!
    sim_manager.run_simulation(n_cycles,
                               steps_list)

    # read hdf5 and check if positions are different


    # cleanup if necessary
    if cleanup:
        os.remove(osp.join(base_folder,'tmp.wepy.results.h5'))
    
if __name__ == '__main__':
    run_openmm_system('Reference',cleanup=False)

