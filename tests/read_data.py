from wepy.wepy_reporter import GetResult
from wepy.hdf5 import TrajHDF5

import mdtraj as mdj

if __name__=='__main__':
    pdb = mdj.load_pdb('sEH_TPPU_system.pdb')
    topology = pdb.topology
    hd = TrajHDF5('/mnt/research/DicksonLab/share/sEH_Wepy_results/wepy_results.h5', mode='r')
    hd.open()
    hdf5 = hd.h5
    
    reporter = GetResult(hdf5)
    
    walker_num = 48
    walkers_list = [i for i in range(walker_num)]
    cycle_idx = 105
    weights =[]
    for cycle_idx in [103,104,105]:
        weights = reporter.get_cycle_data(0,cycle_idx, walker_num, 'weights')
        print('cycle{}'.format(cycle_idx))
        for i in range(walker_num):
            print('[{}]={}\n'.format(i, weights[i]))
    hd.close()
  
    #traj = reporter.make_traj(0, walkers_list, cycle_idx, topology)
    # save in dcd format
   # traj.save_dcd('traj{}.dcd'.format(cycle_idx))
    
    
