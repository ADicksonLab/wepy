from wepy.hdf5 import WepyHDF5

# initialize
wepy_hdf = WepyHDF5("test.h5", 'w')

# add a run with a name
run_grp = wepy_hdf.new_run(name="my run")

# add trajectories to the run
