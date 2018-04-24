import sys
import os.path as osp

from wepy.hdf5 import WepyHDF5
from wepy.analysis.tree import ancestors
from wepy.resampling.resamplers.wexplore import WExploreResampler

if sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print("walker_lineage.py run_index walker_index output_DCD_path")
else:
    run_idx = int(sys.argv[1])
    walker_idx = int(sys.argv[2])
    dcd_path = sys.argv[3]

    outputs_dir = osp.realpath('../outputs')

    hdf5_filename = 'results.wepy.h5'

    hdf5_path = osp.join(outputs_dir, hdf5_filename)

    wepy_h5 = WepyHDF5(hdf5_path, mode='r')

    wepy_h5.open()

    cycle_idx = wepy_h5.traj(run_idx, walker_idx)['positions'].shape[0] - 1

    resampling_panel = wepy_h5.run_resampling_panel(run_idx)

    parent_panel = WExploreResampler.DECISION.parent_panel(resampling_panel)
    parent_table = WExploreResampler.DECISION.net_parent_table(parent_panel)

    lineage = ancestors(parent_table, cycle_idx, walker_idx)

    mdj_traj = wepy_h5.run_trace_to_mdtraj(run_idx, lineage)

    mdj_traj.save_dcd(dcd_path)
