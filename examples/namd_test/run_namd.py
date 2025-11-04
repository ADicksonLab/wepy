from wepy.sim_manager import Manager
from wepy.resampling.resamplers.resampler import NoResampler
from wepy.runners.namd import NAMDRunner, NAMDWalker, prepare_initial_states

from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.work_mapper.task_mapper import TaskMapper, WalkerTaskProcess
from wepy.util.mdtraj import mdtraj_to_json_topology

import mdtraj as mdj

if __name__ == "__main__":
    
    runcmd = f"namd3 +p3"  # conf_file will be added by NAMDRunner
    inputs_path = './inputs' # all files in this dir will be copied to the work_dir
    conf_file = 'base.namd'  # needs to have 'TMP_INPUT_NAME', 'TMP_NSTEPS', 'TMP_OUTPUT_NAME',
                             # which will be replaced at runtime
    work_dir_path = '/scratch/...'  # stores temporary run files that will be cleaned periodically

    pdb_path = "inputs/system.pdb"
    
    # set up NAMD runner
    runner = NAMDRunner(runcmd, inputs_path, conf_file, work_dir_path, get_velocities=True, cycle_cache=2)

    # get the walker topology in a json format
    pdb = mdj.load_pdb(pdb_path)
    json_top = mdtraj_to_json_topology(pdb.top)

    # set up parameters for running the simulation
    num_walkers = 48
    init_weight = 1.0 / num_walkers

    # generate initial list of walkers
    coor_paths = ['inputs/init.coor' for i in range(num_walkers)]
    xsc_paths = ['inputs/init.xsc' for i in range(num_walkers)]
        
    init_walkers = prepare_initial_states(work_dir_path, coor_paths, xsc_paths, vel_paths=[])

    # set up the initial files
    runner.prep_initial_files(init_walkers)
    
    resampler = NoResampler()

    hdf5_reporter = WepyHDF5Reporter(save_fields=('positions','box_vectors','colvar1','velocities'),
                                     file_path='wepy.results.h5',
                                     resampler=resampler,
                                     topology=json_top)

    mapper = TaskMapper(walker_task_type=WalkerTaskProcess,
                        num_workers=4,
                        platform='CUDA',
                        device_ids=[0,1,2,3])

    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=mapper,
                          reporters=[hdf5_reporter])


    steps_list = [n_steps for i in range(n_cycles)]
    
    # and..... go!
    sim_manager.run_simulation(n_cycles,
                               steps_list)
    
