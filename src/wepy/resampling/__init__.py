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
their own resampler.

The only strict requirement that the framework is that a resampler
must have a method "resample" that receives a list of walkers and
returns some other list of walkers along with optional records the
resampling process and any changes to the state of the resampler
itself (while these records aren't strictly necessary for a simulation
to run, they are paramount to actually recording data on the
simulations).

However, useful resamplers share a lot of shared functionality. This
sub-package attempts to provide a sub-framework that provides this
functionality.

Decisions
=========

First is the definition of the "decisions" and the decision records.

A resampling process given a set of weighted samples should produce a
number of other samples where the set of states in the output is a
subset of those in the input.

For example a simple form of resampling is subsampling, where we
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

A more complex (and useful) example would be to add a possible
decision to split up a walker into two. In this there are now two
samples of the same random variable (state) but the total probability
of that state is the same since the weights are normalized and divided
in half for each.

These decisions might be called: DROP, KEEP, SPLIT. Now we can replace
dropped samples so that the number of samples is kept the same.

We could also allow the splitting of a walker into multiple walkers
instead of just two, in this case we replace SPLIT with CLONE. The
CLONE record now needs some additional information to parametrize it's
action. A record containing the decision value and the decision
parameters is called a 'decision record'. A CLONE decision record would
look like for example:

(decision=CLONE, n_children=3)

If we add data about which walker it applies to on which cycle and
which step of resampling it would be a 'resampling record'.


One issue with the use of a pure subsampling scheme is that we must
renormalize the weights of the walkers when any are dropped. In WE the
sample distribution is preserved at each step by allowing the
'merging' of walkers instead of just removing them. Merging involves
choosing a walker/sample to remove and adding that weight to another
walker, before discarding the sample value.

This cloning-merging strategy is the primary Decision class used in
wepy and the choice of most resamplers. The full listing of decisions
is:

- NOTHING
- CLONE
- SQUASH
- KEEP_MERGE

Where NOTHING and CLONE are as above, SQUASH is the walker that will
be discarded and have it's weight donated, and KEEP_MERGE is a walker
that will be receiving weight from one or more squashed walkers.

Resamplers
==========

Abstractly a resampler in the context of wepy is an algorithm that
given a sample of walkers produces instructions on how to resample
them. The instruction set is defined and implemented by a decision
object.

Concretely (and state above), the Resampler class is parametrized by a
Decision class and has a method 'resample' which receives a set of
walkers and returns a resampled collection of walkers.

Resamplers in the sub-framework also produce resampling records in
roughly the format described above so that storage formats can record
the lineages (family trees) of walkers.

Changes in the state of the resampler can also be recorded for
profiling and analysis purposes if it is desired, but is completely
specific to the resampling algorithm.


It is important to note that the choice of resampler is entirely
independent of:

- the type of sampling used e.g. molecular dynamics, monte carlo
- the system being simulated e.g. proteins, polymers

The type of sampling being used is encapsulated in the Runner class.

For the second point we would like our resamplers to be as resuable as
possible between different systems.

For example, wepy has implementations of both the WExplore and REVO
resampling algorithms which are broadly applicable and interchangable.

For instance if you want to simulate ligand unbinding for a particular
protein-ligand system you can choose either WExplore or REVO. This
choice can be made after you have already determined a metric for
discerning novelty in you protein-ligand system, as long as the metric
is encapsulated in a Distance class.

Distances
=========

A distance metric (in the context of wepy) is simply a function that
returns a single value when given the states of two walkers.

For such a function to be considered a proper distance metric extra
criterion are required but wepy does not enforce these in
implementation.

A Distance class is simply a class with a single method 'distance'. To
allow for certain basic performance requirements, it is also required
that the 'image' and 'image_distance' methods be overriden. This
allows for the transformation (which is potentially compute intensive)
of a walker state to a reduced form called an image. The
'image_distance' method should produce the same results as the
'distance' method for the same walker states.

"""
