Merge Multiple Simulation Data
==============================

In this section, we will cover how to merge multiple simulation data
files into a single HDF5 file using the wepy.hdf5 module. This process
is essential for combining the results of different simulation runs,
making it easier to analyze the overall data.

When running multiple simulations, it is often useful to merge the
resulting data files into a single file. This allows for a more
streamlined analysis and management of the data. The following example
demonstrates how to use the WepyHDF5 class to accomplish this task.

The following Python script shows how to initialize, clone, and link
multiple HDF5 files containing simulation results.

.. code:: python

   import os.path as osp
   from wepy.hdf5 import WepyHDF5

   print('running...', flush=True)

   # Define the base path where the merged file will be saved
   BASE_PATH = '<simulation-base-path>'

   # List of HDF5 files to be merged
   hdf5_filenames = [
       'run1.h5',
       'run2.h5',
       'run3.h5',
       ...
   ]

   # Initialization of a single HDF5 file
   init_wepy_h5 = WepyHDF5(filename=hdf5_filenames[0], mode='r')

   # Creating a clone (This has to be done every time you link all HDF5 files in as separate runs)
   CLONE_FILENAME = osp.join(BASE_PATH, 'merge.h5')

   with init_wepy_h5:
       wepy_file = init_wepy_h5.clone(path=CLONE_FILENAME, mode='w')

   # The linking step
   with wepy_file:
       for hdf5_filename in hdf5_filenames:
           wepy_file.link_file_runs(hdf5_filename)

The script performs the following steps:

#. **Initialization**:

   -  We start by importing the necessary modules and defining the base
      path where the merged file will be saved.
   -  A list of HDF5 filenames is provided, which contains the paths to
      the individual simulation result files.

#. **Cloning**:

   -  We initialize a WepyHDF5 object with the first HDF5 file in read
      mode.
   -  Using the clone method, we create a new HDF5 file that will serve
      as the merged output file.

#. **Linking**:

   -  We open the newly created HDF5 file in write mode.
   -  Each of the HDF5 files listed in ``hdf5_filenames`` is linked to
      the new file using the ``link_file_runs`` method.

By following these steps, you can merge multiple HDF5 files into a
single file, making it easier to handle and analyze your simulation
data.

Merging simulation data is a crucial step in data analysis workflows,
especially when dealing with multiple runs. The wepy.hdf5 module
provides a straightforward way to achieve this, ensuring that you can
efficiently combine and manage your simulation results.
