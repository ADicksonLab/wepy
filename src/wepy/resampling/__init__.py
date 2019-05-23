"""Resampling functionality for weighted ensemble simulations.

The defining feature of weighted ensemble simulations (as opposed to
simply running multiple independent simulations in parallel) is the
addition of a processing step that allows for communication of the
states of the parallel simulations. In this communication step we have
the opportunity to make decisions about the relative productivity of
each of them and attempt to bias the overall behavior of the ensemble
of simulations, while not explicitly modifying the governing laws
(e.g. equations of motion, force field, hamiltonian, etc.) of each
parallel dynamics simulation universe.

The 'bias' in weighted ensemble (WE) simulations is only a bias in the
degree of sampling that is performed, and not in the revealed
content. The goal of any WE simulation (not done for fun) is to
estimate some property of a dynamical system (e.g. the rate ligand
unbinding).  The accurate estimation of such a property is typically
only dependent on a finite subset of the space that is being
sampled. Indeed in infinite spaces, such properties cannot be
calculated in finite time.

Thus the values of the finite subspace that is necessary for an
accurate estimate of the parameter is all that is needed. Thus, we can
restrict our attention to only this region of the space.

Given infinite time for a dynamical process that satisfies an
ergodicity assumption, we can expect to gather an exact sampling of
the desired finite subspace if we know that we are bound to sampling a
finite superset of the desired space. This is better than the previous
situation of sampling an infinite space. However, depending on the
size of this space, the available initial samples, and waiting times
for traversing different regions of the space the time needed to
obtain a sufficient sample may be very large and impractical to
calculate in reality.

To be continued...

"""
