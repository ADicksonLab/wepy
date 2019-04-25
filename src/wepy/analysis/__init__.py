"""Analysis tools for wepy datasets.

Analysis of weighted ensemble (WE) datasets are complicated by their
nonlinear branching structure, and thus in its raw form cannot be
analyzed by the tools available for linear trajectories. See the
documentation (TODO: link) for a more in depth explanation of the
general structure of weighted ensemble data.

These analysis modules do not aim to redo any particular analysis
calculation but rather to aid in accessing, summarizing, and
manipulating branched simulation data.

The starting point for this is the `ContigTree` class in the
`contig_tree` module. This is a general purpose container that
combines the data specifying the branching structure with the
underlying linear data structures. It is necessary for branched data
but can also be used for collections of linear data as well.

The main anlaysis routines we attempt to facilitate are:

1. Generating probability histograms over calculated
   observables. (TODO)

2. Generating contiguous trajectories of walkers that have reached
   target boundary conditions (TODO).

3. Visualizing the cloning and merging history of simulations
   (i.e. resampling trees) using the `parents` module and
   `network_layouts`.

4. Generating 'state networks' from frame labels and computing
   transition matrices with variable lag times using the `network` and
   `transitions` modules.

To calculate observables on the dataset you need to work directly with
the data storage API in the `wepy.hdf5.WepyHDF5` module using the
`compute_observable` method.

See Also
--------

`wepy.hdf5.WepyHDF5.compute_observable`

Notes
-----

References
----------

Examples
--------

"""
