Introducing Resamplers
======================

A resampler is a computational algorithm used in WE simulations to
enhance sampling of particular states in a system. It typically operates
on an ensemble of walkers (representing different states) and performs
resampling operations such as cloning and merging. The main objective is
typically to promote sampling of low probability states that are
important to a process of interest.

In this chapter, we will introduce the REVO resampler. You can find more
detailed information in the paper `"REVO: Resampling of ensembles by
variation optimization" <https://pubmed.ncbi.nlm.nih.gov/31255090>`__
but briefly: REVO is a Weighted Ensemble based enhanced sampling
algorithm which uses cloning and merging to create ensembles of diverse
trajectories without defining any regions. It measures the diversity
using a quantity called the "trajectory variation", which depends on the
pairwise distances between the walkers, as well as their weights. REVO
solves this optimization problem using a greedy algorithm which at each
step selects best walkers for resampling operations (cloning and
merging) in order to maximize the variation.

Let's start our building our resampler, but first, we need to define a
way of measuring the distances between the trajectories, which we call
the "distance metric". This metric is important as it should correspond
to the meaningful differences between trajectories that you want to see
in your WE simulation. These are easy to customize for your system of
interest. Wepy also provides a few built-in distance metrics which you
can use. Below we define a simple distance metric for our NaCl system,
which uses the absolute value of the difference between the interatomic
distances.

.. code:: python

   from wepy.resampling.distances.distance import Distance
   from wepy.resampling.resamplers.revo import REVOResampler
   import numpy as np

   ...
   # we define a simple distance metric for this system, assuming the
   # positions are in a 'positions' field
   class PairDistance(Distance):

   def __init__(self):
       pass

   def image(self, state):
       return np.sqrt(np.sum(np.square(state['positions'][0] - state['positions'][1])))

   def image_distance(self, image_a, image_b):
       return np.abs(image_a - image_b)
   ...

   distance = PairDistance()

A few notes are important here.

#. Remember that the PairDistance is measuring distances between two
   different walkers. In other words, two different copies of the
   simulation. Here these simulations are very simple: each has only two
   atoms, but in general these will each have their own solvent,
   protein, box sizes, etc.
#. By convention the distance metric is split into two parts: the first
   is a function called image that extracts necessary information from
   the WalkerState. This is a dictionary that contains positions, but
   also box vectors, velocities and other quantities as well. Here, the
   "image" of each state is simply the distance between its two
   particles. The second part is a function called ``image_distance``,
   which performs the necessary calculations to get the distance between
   a pair of images. Note that while image is called O(N) times by the
   resampler (where N is the number of walkers), ``image_distance`` is
   called O(N\ :sup:`2`) times.

With the distance metric in hand, we can create our REVO resampler:

.. code:: python

   ...
   # Set up the REVO Resampler with the parameters
   resampler = REVOResampler(distance=distance,
                             init_state=init_state,
                             weights=True,
                             pmax=0.1,
                             dist_exponent=4,
                             merge_dist=0.25,
                             char_dist=0.1)
   ...

You can find more details about the parameters in the documentation.
Briefly:

-  ``distance``: The distance metric to compare walkers.
-  ``init_state``: A ``WalkerState`` object used for automatically
   determining state image shape.
-  ``weights``: Turns off or on the weight novelty in calculating the
   variation equation. When ``weights = False``, the value of the
   novelty function is set to 1 for all walkers.
-  ``pmax``: The maximum statistical weight. It prevents accumulation of
   excessive weight in one walker.
-  ``dist_exponent``: The distance exponent that modifies distance and
   weight novelty relative to each other in the variation equation.
-  ``merge_dist``: The merge distance threshold. Walkers farther than
   this distance will not be merged. Units should be the same as the
   distance metric.
-  ``char_dist``: The characteristic distance value. It is calculated by
   running a single dynamic cycle and then calculating the average
   distance between all walkers. Units should be the same as the
   distance metric.

It is also useful to add a ``REVODashboardSection`` to the
``DashboardReporter``:

.. code:: python

   from wepy.reporter.revo.dashboard import REVODashboardSection

   ...

   revo_dashboard_sec = REVODashboardSection(resampler)
   dashboard_reporter = DashboardReporter(file_path = dashboard_path,
                                         runner_dash = openmm_dashboard_sec,
                     resampler_dash = revo_dashboard_sec)

   ...

And that's it! You have created a REVO resampler. Feel free to test this
out for the NaCl example. You can paste these code sections in or find
the ``wepy_script_resampler.py`` file in the ``examples/quick_start``
folder.

Now you can use this resampler in your simulations. This is the final
chapter of our Quickstart guide, in the Tutorials section we will
introduce how to prepare and use your own data along with how you can
analyze the simulation results.
