More on Reporters
=================

In our previous section, we introduced how to create a
``DashboardReporter``. Although it is a powerful tool for monitoring the
progress of your simulation, we will need other reporters for capturing
data and preparing it for analysis. In this section, we will introduce
two more reporters to our capture simulation data.

First item on our list is the ``WepyHDF5Reporter``. This for generating
an HDF5 format (``WepyHDF5``) data file that contains quantitative
information about each trajectory (positions, box vectors, velocities,
etc.) and how these trajectories are related (resampling records,
warping records). Below we have the basic structure for our ``WepyHDF5``
file. This can be thought of as a file system that has a set of nested
folders. We will go through how to retrieve data from this file in the
:any:`Tutorials <data_analysis>` section.

-  ``runs``: Group containing all primary data from simulations.

   -  ``0``: Group for run 0.

      -  ``trajectories``: Group for trajectory data.

         -  ``0``: Group for trajectory 0.

            -  ``positions``: Dataset for atomic positions.
            -  ``box_vectors``: Dataset for box vectors.
            -  ``velocities``: Dataset for atomic velocities.
            -  ``weights``: Dataset for walker weights.

         -  ``1``, ``2``, …, ``n``: Groups for other trajectories.

      -  ``resampling``: Record group for resampling data.
      -  ``resampler``: Record group for resampler data.
      -  ``warping``: Record group for warping data.
      -  ``progress``: Record group for progress data.
      -  ``boundary_conditions``: Record group for boundary conditions
         data.
      -  ``init_walkers``: Group for initial walker states.
      -  ``decision``: Record group containing resampling decisions.

   -  ``1``, ``2``, …, ``n``: Groups for other runs.

-  ``topology``: Dataset containing molecular topology as a JSON string.
-  ``_settings``: Group for type and configuration information.

Now that we have a grasp on what is in the HDF5 file, let's include it
in our simulation. Again, you can either copy and paste this code into
your ``wepy_script.py`` file, or use the ``wepy_script_hdf5.py`` file
found in ``examples/quick_start``.

.. code:: python

   # Import the reporter
   from wepy.reporter.hdf5 import WepyHDF5Reporter

   ...
   # Set up the HDF5 reporter
   hdf5_reporter = WepyHDF5Reporter(save_fields=('positions','box_vectors'),
                             file_path='wepy.results.h5',
                             resampler=resampler,
                             topology=json_top)
   ...

And it is that simple to create an ``HDF5Reporter`` in Wepy. Important
thing to notice is the ``save_fields`` parameter. It is a selection of
fields from the walker states to be stored. This allows us to omit some
fields that could be unnecessary, such as the velocities. If ``None``,
all fields will be saved.

Now that we have created our reporter, all it takes is a simple update
to the simulation manager. We need to add this reporter to the list of
reporters that the simulation manager will use. Below is the updated
code:

.. code:: python

   sim_manager = Manager(init_walkers,
                     runner=runner,
                     resampler=resampler,
                     reporters=[dashboard_reporter, hdf5_reporter]
                     )

Next item on the list is the ``WalkerPklReporter``. This is designed to
periodically save the states of all walkers in a simulation to a pickle
file for backup purposes. Whenever you want to continue a simulation
from a certain point, you can use the pickle file to load the state of
the simulation. Below is the code for creating such a reporter:

.. code:: python

   from wepy.reporter.walker_pkl import WalkerPklReporter

   ...
   pkl_reporter = WalkerPklReporter(save_dir = 'pkls',
                                 freq = 1,
                                 num_backups = 2)
   ...

The ``save_dir`` parameter is the directory where the pickle files will
be saved. The ``freq`` parameter is the frequency at which the reporter
will save the state of the walkers. ``num_backups`` is the number of
backups that the reporter will keep. If the number of backups exceeds
the specified number, the oldest backup will be deleted. Now we need to
include this reporter to our simulation manager as well:

.. code:: python

   sim_manager = Manager(init_walkers,
                 runner=runner,
                 resampler=resampler,
                 reporters=[dashboard_reporter, hdf5_reporter, pkl_reporter]
                 )

And thats it! We have all three of our reporters to (1) track the
progress of our simulation, (2) save the simulation data to an HDF5
file, and (3) save the state of the walkers to a pickle file. Now we can
run our simulation and check the results.

After running the simulation you should see a couple files and a folder
appear in your directory:

-  ``wepy.dash.org``
-  ``wepy.results.h5``
-  ``pkls``

Next, we will continue with including a resampler to our simulation.
