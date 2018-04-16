import numpy as np

def traj_field_lj_dist(traj_data):

    positions = traj_data['positions']

    # slice out positions for each LJ particle
    lj1 = positions[:,0,:]
    lj2 = positions[:,1,:]
    # compute distances
    distances = np.sqrt((lj1[:,0] - lj2[:,0])**2 + (lj1[:,1] - lj2[:,1])**2 + (lj1[:,2] - lj2[:,2])**2)
    return distances



if __name__ == "__main__":
    from wepy.hdf5 import WepyHDF5

    # load the HDF5 file in read/write so we can save data to the
    # observables
    wepy_hdf5_path = "../outputs/results.wepy.h5"
    wepy_h5 = WepyHDF5(wepy_hdf5_path, mode='r+')

    print('test')
    with wepy_h5:
        wepy_h5.compute_observable(traj_field_lj_dist, ['positions'],
                                   save_to_hdf5='rmsd',
                                   map_func=map, debug_prints=True)
