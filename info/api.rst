API Overview
============

This is a listing of the important and most used for an exhaustive
(and potentially exhausting) listing see the module index:

   
Analysis
--------

* :any:`Free Energy Profiles of Observables </_api/wepy.analysis.profiles>`
* :any:`Contigs and Contig Tree/Forest </_api/wepy.analysis.contig_tree>`
* :any:`Parent Tables & Lineage Traces </_api/wepy.analysis.parents>`
* :any:`Warping Rates </_api/wepy.analysis.rates>`
* :any:`Macro-State Network </_api/wepy.analysis.network>`
* :any:`Network Transition Probabilities </_api/wepy.analysis.transitions>`



Data Storage
------------

* :any:`WepyHDF5 </_api/wepy.hdf5>`
      
Orchestration
-------------
      
* :any:`Orchestrator </_api/wepy.orchestration.orchestrator>`
* :any:`Configuration </_api/wepy.orchestration.configuration>`

Wepy Core
---------

* :any:`Simulation Manager </_api/wepy.sim_manager>`
* :any:`Walker </_api/wepy.walker>`

  ..
     Orchestration CLI
   -----------------

   TODO

   
Utilities
---------

* :any:`JSON Topology </_api/wepy.util.json_top>`
* :any:`Miscellaneous </_api/wepy.util.util>`
* :any:`MDTraj Interface </_api/wepy.util.mdtraj>`


Application Components
----------------------

Reporting on Simulations
^^^^^^^^^^^^^^^^^^^^^^^^

General Purpose:

* :any:`Text Dashboard </_api/wepy.reporter.dashboard>`
* :any:`WepyHDF5 </_api/wepy.reporter.hdf5>`
* :any:`Resampling Tree </_api/wepy.reporter.restree>`
* :any:`Last Walkers </_api/wepy.reporter.walker>`

WExplore and Image Based Resamplers:

* :any:`Images </_api/wepy.reporter.wexplore.image>`
* :any:`Dashboard </_api/wepy.reporter.wexplore.dashboard>`

Resamplers
^^^^^^^^^^

* :any:`WExplore </_api/wepy.resampling.resamplers.wexplore>`
* :any:`REVO </_api/wepy.resampling.resamplers.revo>`


Distance Metrics
^^^^^^^^^^^^^^^^

* :any:`Ligand Unbinding and Rebinding </_api/wepy.resampling.distances.receptor>`


Runners
^^^^^^^

* :any:`OpenMM </_api/wepy.runners.openmm>`

Work Mapper
^^^^^^^^^^^
   
* :any:`Single Process </_api/wepy.work_mapper.mapper>`
* :any:`Worker Mapper </_api/wepy.work_mapper.mapper>`

Parallel mapper via python multiprocessing that implements the
worker-consumer model. There will only be as many forked processes as
workers used.

* :any:`Task Mapper </_api/wepy.work_mapper.task_mapper>`

Parallel mapper via python multiprocessing that implements a task
based parallelism. Every walker every cycle will have a new process
created (forked) and workers are scheduled via a lock queue.

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

* :any:`Receptor-Ligand (Un)Binding </_api/wepy.boundary_conditions.receptor>`
   
