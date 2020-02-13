Glossary
========


.. glossary::

   idx
      short for index

   idxs
      short for indices

   weighted ensemble
      A class of enhanced sampling algorithms that iteratively perform
      parallel sampling coupled with resampling of the replicas, called
      walkers.

      A converged weighted ensemble simulation gives the exact probability
      distribution being sampled.

   walker
      The individual simulatio replicas of a weighted ensemble
      simulation. Each walker has a state and a weight corresponding to the
      probability.

   trajectory
      A contiguous block of trajectory frames that is not necessarily (or
      even usually) logically contiguous.

      In other words an implementation detail.

   trajectory fields
      The types of data associated with a trajectory is left largely
      unspecified making it a general purpose container for any type of
      runner or state that is produced through simulation.

      Typically, this includes positions, box vectors etc.

      Fields are named by strings, e.g. "positions" and special compound
      fields are separated by the HDF5 path separator
      e.g. "alt_reps/alt_rep1", "observables/rmsd".

   records
      Data associated with walkers and a run.

      Takes the form of a list of tuples (usually namedtuple records).

   continual records
      Records for a wepy simulation for which one is produced every cycle.

   sporadic records
      Records for a wepy simulation which may have none or many produced for
      a cycle.

   resampling records
      Records that report on the resampling of walkers i.e. cloning
      and merging. Contains the fields of a decision record. A
      destructured resampling panel.

   resampler records
      Records that report on the state of the resampler itself.

   resampling step
      A single cycle of resampling may be broken into several steps
      that are applied in order. A single step is a collection of
      decision records for each walker in the cohort. This, for
      example, allows for recursive clones, which would be very
      difficult to encode in a simple decision set.

   resampling panel
      This is the datastructure that describes the full process of
      resampling for an entire run or contig.

      A single step of resampling results in a resampling record for each
      walker and the list of these records (in order of walker ordering) is
      one "step" of a "cycle" resampling.

      A resampling panel is a list of lists where each element of the list
      contains the resampling steps of that cycle.

      The cycle element is a list of records.

      The structure for the resampling panel looks like this roughly:

      .. code:: python
      
          [
            cycle_0 [
                      step_0 [rec0, rec1, ...],
                      step_1 [rec0, rec1, ...],
                      ...
                    ]
            ...
          ]

   parent table
      A listing for each cycle snapshot of a weighted ensemble contig that
      specifies the parents of the next generation walkers. There can be
      different number of walkers in each cycle and thus is a list of lists
      of ints.

      Specifically:

      For a parent table P with M cycles, N_i is the number of walkers in
      cycle i has a listing P_i which has N_{i+1} elements. For walker j,
      P_i^j, has a domain 0 to N_i and specifies the index of the walker in
      cycle i that is the parent of walker j in cycle i+1.

   parent panel
      A parent panel is similar in structure to the resampling panel except
      that instead of resampling records as the atomic elements there are
      simply integers with the same meaning as in the parent table.

   warping records
      Records that report on events of walkers satisfying boundary
      conditions in the last cycle.

      Warping records are typically treated as either discontinuous
      (e.g. the walker's state is set to a specified other state) or
      continuous (e.g. the "color" of a walker changes as it reaches a
      boundary).

      The interpretation of this however is entirely domain specific.

   bc (boundary conditions) records
      Records that report on the state of the boundary conditions
      themselves.

   progress records
      Continual records that a boundary condition reports that provide some
      metrics on the walkers, such as their position along a progress
      coordinate.

   run
      A single weighted ensemble simulation. Is a collection of walker
      trajectories and records specifying parental relations, boundary
      condition events etc.

      Runs may be logically connected to other runs (through continuations)
      and can be thought of as a unit of computation.

   continuation
      A 2-tuple of the form (continuation_run_idx, continued_run_idx) that
      establishes a logical relationship between the end of one run
      (continued_run_idx) and the beginning of another run
      (continuation_run_idx).

      A collection of continuations is the underlying specification for a
      contig tree for run contigs.

   contig tree
      For a collection of weighted ensemble runs with continuations
      specified between them, the contig tree imposes the tree structure
      over them.

      Technically a forest since there can be multiple roots.

   contig
      A logically contiguous list of weighted ensemble cycle snapshots.

   spanning contig
      A contig from a contig tree that extends from a root to a leaf

   run contig
      A logically contiguous list of weighted ensemble runs. The only
      continuation relationship allowed is end-to-end.

   trace
      list of tuples which indexes things over either a contig tree, contig,
      or a run.

   run trace
      list of tuples of the form (run_idx, traj_idx, cycle_idx) over the
      contig tree

   contig trace
      list of tuples of the form (run_idx, cycle_idx) over the contig tree

   contig walker trace
      list of tuples of the form (traj_idx, cycle_idx) over a contig

   run cycle idx
      An index of an in-run cycle

   contig cycle idx
      An index of an in-contig cycle index (can include multiple runs).

   resampler
      An object that resamples walkers each cycle of a weighted ensemble
      simulation.

   runner
      The object that propogates the sampling of an individual walker.

   reporter
      A reporter is an object that receives all of the walker states and
      records at the end of the cycle and is allowed to write it out in some
      way.

   resampling decision
      A specification of a type of action to take on a sample within a sample set.

   decision record
      A specific key-value data structure that represents a resampling
      decision. Must contain a representation of the decision value
      (enum or its integer value) and any additional fields
      (instructions) that are needed to perform the resampling, such
      as the number of clones or which walker to merge with. Contained
      within a resampling record which adds metadata which places it
      within the context of an entire simulation.

   decision instruction record
      See decision record.

