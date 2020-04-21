import numpy as np

# this is a typical function for mapping over the dataset. I have
# added the extra argument scaling_factor just to demonstrate that you
# can have other arguments.
def traj_field_lj_dist(traj_data, scaling_factor):

    positions = traj_data['positions']

    # slice out positions for each LJ particle
    lj1 = positions[:,0,:]
    lj2 = positions[:,1,:]
    # compute distances with the scaling factor
    distances = scaling_factor * np.sqrt(
        (lj1[:,0] - lj2[:,0])**2 + (lj1[:,1] - lj2[:,1])**2 + (lj1[:,2] - lj2[:,2])**2)

    return distances



if __name__ == "__main__":
    from wepy.hdf5 import WepyHDF5

    # load the HDF5 file in read/write so we can save data to the
    # observables
    wepy_hdf5_path = "_output/we/results.wepy.h5"
    wepy_h5 = WepyHDF5(wepy_hdf5_path, mode='r+')

    with wepy_h5:
        wepy_h5.compute_observable(traj_field_lj_dist, ['positions'],
                                   (2.0,),
                                   save_to_hdf5='2*rmsd',
                                   map_func=map)

        wepy_h5.compute_observable(traj_field_lj_dist, ['positions'],
                                   (1.0,),
                                   save_to_hdf5='rmsd',
                                   map_func=map)
