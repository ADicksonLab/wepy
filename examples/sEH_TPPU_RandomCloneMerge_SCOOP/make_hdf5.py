import pandas as pd

from wepy.hdf5 import WepyHDF5, load_topo_dataset

if __name__ == "__main__":

    # load the resampling records
    resampling_df = pd.read_csv("resampling.csv", index_col=0)

    # load the topology from h5 system
    topo_h5 = load_topo_dataset('sEH_TPPU_system.h5')

    # make a wepy dataset
    wepy_h5 = WepyHDF5("wepy.h5", mode='w', overwrite=True)

    # add the trajectories for each walker to the dataset
