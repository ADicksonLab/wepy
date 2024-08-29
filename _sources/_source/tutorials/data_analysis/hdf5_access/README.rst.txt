Working with HDF5
===================

The wepy.hdf5 module is part of the WEPY (Weighted Ensemble Pathway) framework, used for handling HDF5 files that store simulation data. This module helps in reading, writing, and managing the simulation data stored in HDF5 format.

To work with an HDF5 file, you need to create a WepyHDF5 object. It is a good idea to open the file "r+" mode which will append to the file instead of overwriting it.

.. code:: python

    from wepy.hdf5 import WepyHDF5

    # Open an existing HDF5 file or create a new one
    wepy_h5 = WepyHDF5('path_to_your_file.h5', mode='r+')  # 'r' for read, 'w' for write, 'a' for append

You can access the trajectories stored in the HDF5 file:

.. code:: python

    with wepy_h5:
        trajs = wepy_h5.get_run_trajs(run_idx=0)

To extract specific data, such as configurations or observables:

.. code:: python

    with wepy_h5
        configurations = wepy_h5.get_run_configurations(run_idx=0)
        observables = wepy_h5.get_run_observable_field(run_idx=0, obs_field='your_observable')


To write data into the HDF5 file:

.. code:: python

    with wepy_h5:
        wepy_h5.add_run(new_run)
        wepy_h5.add_run_configurations(run_idx=0, configurations=new_configurations)
        wepy_h5.add_run_observable_field(run_idx=0, obs_field='your_observable', data=new_data)

Always close the HDF5 file after operations:

.. code:: python
    
    wepy_h5.close()