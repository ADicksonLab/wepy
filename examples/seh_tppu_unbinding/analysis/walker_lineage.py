import sys
import os.path as osp

from wepy.hdf5 import WepyHDF5
from wepy.analysis.tree import ancestors
from wepy.resampling.wexplore1 import WExplore1Resampler

if sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print("walker_lineage.py run_index walker_index output_DCD_path")
else:
    run_idx = sys.argv[1]
    walker_idx = sys.argv[2]
    dcd_path = sys.argv[3]

    outputs_dir = osp.realpath('../outputs')

    hdf5_filename = 'results.wepy.h5'

    hdf5_path = osp.join(outputs_dir, hdf5_filename)

    wepy_h5 = WepyHDF5(hdf5_path, mode='r')

    wepy_h5.open()

    cycle_idx = wepy_h5.traj(run_idx, walker_idx)['positions'].shape[0]

    resampling_panel = wepy_h5.run_resampling_panel(run_idx)

    parent_panel = WExplore1.DECISION.parent_panel(resampling_panel)
    parent_table = WExplore1.DECISION.net_parent_panel(parent_panel)

    lineage = ancestors(parent_table, cycle_idx, walker_idx)

    mdj_traj = wepy_h5.trace_to_mdtraj(lineage)

    mdj_traj.save_dcd(dcd_path)
