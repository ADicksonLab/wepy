Introduction
============

Weighted Ensemble (WE)
----------------------

.. TODO For a background to weighted ensemble see:

Features
--------

-  State of the art WE resamplers: WExplore and REVO
-  Super fast molecular dynamics via OpenMM
-  One of a kind HDF5 storage format for WE data: ``WepyHDF5``
-  Analysis routines for:

   -  free energy profiles
   -  rate calculations
   -  computing trajectory observables
   -  extracting linear trajectories from clone-merge trees
   -  conformation state networks such as Markov State Models (MSMs)
   -  aggregating multiple runs

-  Orchestration framework for managing large number of simulations with
   simulation checkpointing, recovery, and continuations.
-  Expert friendly: Fully-featured framework for building and
   customizing simulations for exactly what you need.
-  Leverage the entire python ecosystem, you're limited to an old
   version of an embedded interpreter.
-  No complex *ad hoc* configuration files, everything is python.

.. _resources:
Getting Started
---------------

Once you have wepy :any:`installed <installation>` you can check out the
:any:`quickstart <quick_start>` to get a rough idea of how it works.

Then you can either read the :any:`user's guide <users_guide>` or head
on to the :any:`tutorials <../tutorials>` or execute the
`examples <https://github.com/ADicksonLab/wepy/tree/master/examples>`__
in the source code.

For a complete description of the modules and their components check out
the :any:`API documentation <../api/modules>`.

Contributed wepy libraries and other useful resources
-----------------------------------------------------

Here is a list of packages that are not in the main ``wepy`` repository
but may be of interest to users of wepy.

These include things like :

-  distance metrics and boundary conditions for different kinds of
   systems
-  resampler and runner prototypes
-  related analysis or utility libraries

They are:

-  `geomm <https://github.com/ADicksonLab/geomm>`__ : purely functional
   library for common numerical routines in computational biology and
   chemistry, with no dependency on specific file or topology formats.

-  `wepy-developer-resources <https://github.com/ADicksonLab/wepy-developer-resources>`__
   : Unofficial and miscellaneous materials related to wepy, including
   talks, workshops, contributed tutorials etc. May be out of date.

-  `mastic <https://github.com/ADicksonLab/wepy/blob/master/sphinx/source/introduction.org>`__
   : Library for doing general purpose "profiling" of intermolecular
   interactions. Useful for computing observables an experimental
   chemist understands. Also useful for building distance metrics.

-  `mdtraj <https://github.com/mdtraj/mdtraj>`__ : Excellent library
   with optimized code for numerical routines of interest in
   computational biology and chemistry. Differs from geomm in that it
   relies on their own topology format. The WepyHDF5 JSON topology
   format is borrowed from this library. Used in wepy as a utility
   writer of commonly used formats like PDBs, DCDs, etc.

-  `openmmtools <https://github.com/choderalab/openmmtools>`__ :
   Contributed components for OpenMM. Contains some ready-made test
   systems that are very convenient for testing and prototyping
   components in wepy.

-  `CSNAnalysis <https://github.com/ADicksonLab/CSNAnalysis>`__ : small
   library for aiding in the analysis of conformation state networks
   (CSNs) which can be generated from ``wepy`` data.

Alternatives
------------

``wepy`` is not the only WE framework package. Other packages have
different scopes and features. I have tried to provide a fair comparison
of ``wepy`` to them to help potential users make an informed decision.
If you feel a package is misrepresented contact the ``wepy`` devs or
submit a pull request with your desired changes.

`WESTPA <https://github.com/westpa/westpa>`__

Weighted ensemble package in Python 2.7. More reliant and integrated
with unix-like operating systems providing modularity through shell
scripting and python modules.

As an older project it has support for more MD engines (and non-MD
stochastic sampling engines, e.g. BioNetGen) and is currently better
suited for running simulations on large numbers of CPUs in a clustered
environment.

Support for WE algorithms closer to the original paper by Huber and Kim
with a focus on static tesselation of conformational space.

Has some support for adaptive binning algorithms like WExplore, but it
is a little more challenging to develop radically different resamplers
like REVO, which have no concept of bins at all.

`AWE: Accelerated Weighted
Ensemble <http://ccl.cse.nd.edu/software/awe/>`__

Another Python 2 library with a focus on the Accelerated WE resampling
algorithm and integration with a Work Queue library for distributed
jobs.
