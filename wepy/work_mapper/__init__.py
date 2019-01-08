"""Classes used for efficiently performing work at different stages of
the wepy simulation cycle.

Wepy simulations have four sequential stages of computation:

1. running segments
2. boundary conditions
3. resampling
4. reporting

Stage 1 involves propagating dynamics for each walker in a completely
independent using the same runner function (i.e. `run_segment`) and
can always be completely parallelized.

Steps 2 and 4, with a few more assumptions of independence (which are
most likely satisfied without overly complex behavior), are probably
also completely parallelizable over walkers and reporters
respectively.

Resampling may also have a degree of parallelizability but will vary
considerably between resamplers and so cannot be treated outside of
any individual framework for resamplers and so will not be treated
generally in this submodule.

Currently, wepy supports a worker based task queue implementation
(wepy.work_mapper.mapper.WorkerMapper) for running parallel segments
in stage 1, which is very useful for utilizing GPU compute sources.
It is pretty general and utilizes the multiprocessing library for
starting worker processes. Specific code for running OpenMM GPU
calculations is contained within the openmm runner module.

Currently general solutions for parallelizing stage 2 and 3 are not
provided or supported in the simulation manager (to avoid premature
optimization) but these are easy targets for improving the efficiency
of wepy code.

If you don't care about parallelizing your segment running (say for
simple test systems) you can use the reference implementation
(wepy.work_mapper.mapper.Mapper) which is basically just a wrapper
around a for-loop.

This sub-module provides reference implementations and/or abstract
base classes for a few interfaces.

The WorkMapper class interface has the following methods:

- init
- cleanup
- map

A single attribute for the segment_function that is set at runtime
with a call to `init`:

- segment_func

And an optional attribute to enable reporting on segment running performance:

- worker_segment_times : dict of int : list of float

Which should be a dictionary mapping a worker index to a list of float
values of time (in seconds) for each task (i.e. segment of walker
dynamics) it ran in the last cycle only (not cumulative over
consecutive cycles).

The `init` method is called at runtime by the simulation manager at
the beginning of the simulation and allows for performing such actions
as opening file handles or starting worker processes (as is the case
with the WorkerMapper).

The only necessary key-word argument to this method is 'segment_func'
which will be provided by the simulation manager (and is derived at
runtime from the runner).

The `cleanup` method likewise is called either at the end of a
succesful simulation or when an error occurs in the call to
`run_cycle` in the simulation manager (e.g. to allow killing of live
processes).

The `map` function acts similar to the python builtin except it does
not accept the function to map over the data. This is instead set
during the call to `init` from the 'segment_func' key-word argument.

The WorkerMapper class and related Worker and Task classes will be
provided as-is and are not really intended to be superclasses although
you are free to do so.

The only interfacing that facilitates their usage is that the
simulation manager will pass a required keyword argument 'num_workers'
to the call to `init`.

See the simulation manager module to see what fields are passed to the
mappers. These will likely not be removed in the future, although more
may be added.

"""
