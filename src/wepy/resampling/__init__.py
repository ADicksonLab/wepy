"""Resampling functionality for weighted ensemble simulations.

Role of Resampling in WE
------------------------

The defining feature of weighted ensemble simulations (as opposed to
simply running multiple independent simulations in parallel) is the
addition of a processing step that allows for communication of the
states of the parallel simulations. In this communication step we have
the opportunity to make decisions about the relative productivity of
each of them and attempt to bias the overall behavior of the ensemble
of simulations, while not explicitly modifying the governing laws
(e.g. equations of motion, force field, hamiltonian, etc.) of each
parallel dynamics simulation universe.

A resampler embodies this communication. At a basic level it takes as
input a set of samples derived from dynamics, called walkers. In
addition to the states (the random variable) the walkers also have an
associated probability or weight. At the beginning of a WE simulation
each walker in the ensemble has a uniform probability. But the
probabilities are free to change due to the action of the resampler,
that is due to a resampling process.

Resampling characterizes a potentially large number of algorithms that
may or may not be useful. In the original WE algorithm by Huber and
Kim (TODO add citation) this resampling is achieved through the
"cloning" and "merging" of walkers.

Outline of the resampler sub-package
------------------------------------

The design of the wepy framework allows for any user to implement
their own resampler. The only strict requirement that the framework is
that a resampler must have a method "resample" that receives a list of
walkers and returns some other list of walkers along with optional
records the resampling process and any changes to the state of the
resampler itself (while these records aren't strictly necessary for a
simulation to run, they are paramount to actually recording data on
the simulations).

However, useful resamplers share a lot of shared functionality. This
sub-package attempts to provide a sub-framework that provides this
functionality.

First is the definition of the "decisions" and the decision records.

A resampling process given a set of weighted samples should produce a
number of other samples where the set of states in the output is a
subset of those in the input.

For example the simplest form of resampling is subsampling, where we
simply drop some of the samples from the collection. Similar to how
outliers are removed from a dataset.
In the subsampling mechanism we might say that there are only two
decisions a resampler can make: DROP and KEEP.

These could then be embodied in a Decision class
(e.g. SubsamplingDecision) which enumerates the possibilities and
provides methods for performing the operations on a collection of
walkers given a set of these decisions.

Such a resampler wouldn't be very useful for WE since the simulation
would simply fade away as each walker was eventually eliminated.

A more complex example would be to add a possible decision to split up
a walker into two. In this there are now two samples of the same
random variable (state) but the total probability of that state is the
same since the weights are normalized and divided in half for each.



"""
