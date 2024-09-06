Starting Parallel Runs
======================

In Wepy you can run two seperate scripts with same initial condition or
you can use different initial conditions for parallel runs. It is simply
running two simulation scripts that build their own simulation objects
and run them.

An example usage for this case would be using a randomized subset of
trajectories for setting up the walkers. This way the walkers have
different starting points. It is important to note that it is a good
practice to have different scripts for each of these simulation runs.
This way you can keep track of the changes you made to the scripts and
the results you obtained from them.
