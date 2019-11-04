Development Setup
=================

Get the source code:

.. code:: bash

    git clone https://github.com/ADicksonLab/wepy --recurse-submodules
    cd wepy

Install a virtual environment for it:

.. code:: bash

    wepy_dev_env_refresh () {

        package='wepy'
        conda deactivate
        dev_env="${package}-dev"
        conda env remove -y -n "$dev_env"
        conda create -y -n "$dev_env" python=3
        conda activate "$dev_env"

        # we need openmm but can't get it from pip
        conda install -y -c omnia openmm openmmtools

        # install in editable mode, we need to avoid using pep517 which
        # doesn't allow editable installs
        pip install -r requirements_dev.txt 
        pip install --no-use-pep517 -e .[all]

    }

.. code:: bash

    wepy_dev_env_refresh

Currently, for installing mdtraj we use a forked repository which
handles pip installations better that allows for seamless dependecy
resolution and doesn't require manual intervention to install cython.

This is specified in the requirements.txt file which should be used for
specifying the "concrete" requirements of the project (i.e. the literal
repo or index URL that packages should be retrieved from).

"Abstract" requirements should also be listed in setup.py.

For development specific requirements, we have the separate
requirements\ :sub:`dev`.txt.

Because at this multiple packages are developed simultaneously we
require that geomm be installed in the same directory as wepy for using
the dev requirements.

Releasing Package
-----------------

Test the installation process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for doing this:

.. code:: bash

    wepy_test_build () {
        package='wepy'
        build_env="test-${package}-build"
        conda deactivate
        conda env remove -y -n "$build_env"
        conda create -y -n "$build_env" python=3
        conda activate "$build_env"
        pip install -r requirements_dev.txt
        rm -rf dist/*
        python setup.py build sdist
        conda deactivate
        conda env remove -y -n "$build_env"

    }

    wepy_test_install () {

        package='wepy'
        conda deactivate
        install_env="test-${package}-install"
        conda env remove -y -n "$install_env"
        conda create -y -n "$install_env" python=3
        conda activate "$install_env"
        pip install dist/"$package"-*.tar.gz
        conda deactivate
        conda env remove -y -n "$install_env"

    }

Update versions
~~~~~~~~~~~~~~~

Before we build the package we need to bump the version in all those
places it is written down at, which is achieved with the bumpversion
tool:

.. code:: bash

    bumpversion patch # possible: major / minor / patch

Make sure to tag in git (I typically use magit in emacs but the command
is):

.. code:: bash

    git tag -a vX.Y.Z -m "release message"
    git push gitlab vX.Y.Z

Deploying
~~~~~~~~~

To deploy to PyPI (if you have access)

.. code:: bash

    conda activate wepy-dev
    rm -rf dist/*
    python setup.py sdist
    twine upload dist/*

Building Docs
-------------

Install pandoc for converting org-mode files to rst.

You can follow the instructions on the site or just use anaconda:

.. code:: bash

    conda install pandoc

Then run the build script. This uses the make file and additionally runs
api-doc, and converts org-mode source files to rst using pandoc.

.. code:: bash

    pushd sphinx
    chmod u+x build.sh
    ./build.sh
    popd

This will build the HTML files in the ``sphinx/_build/html`` directory
and if you point your web browser there you can view them.

Deploying Docs
--------------

To run the current deployments of the docs run the deploy script:

.. code:: bash

    pushd sphinx
    chmod u+x deploy.sh
    ./deploy.sh
    popd

Currently we are using github pages, and to avoid putting the build
artifacts of the website into the master development branch we are using
the gh-pages branch.

To make this work you need to pull the gh-pages branch:

Testing
-------

Getting the wepy-tests submodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tests for wepy are included as a submodule because some of the
associated data is large and we want to make the install base for the
program smaller than that. Development of this is tracked in
https://gitlab.com/salotz/wepy-tests.

If you cloned without the recurse-submodules flag you can always pull
them in later like this:

.. code:: bash

    git submodule update --init --recursive

WARNING: before you start editing the ``wepy-tests`` submodule you need
to check out master.

.. code:: bash

    git checkout master

How many times I have edited it before I checked out master...

If you do edit and commit try to get the hash of the commit and then
merge with master. If you don't then you need to figure out which commit
that was.

Test Suite
~~~~~~~~~~

We are using pytest so just run that from the main directory:

.. code:: bash

    pytest

We use a special marker for interacting with test fixtures. We find this
more useful in many cases where you just want to spin up a test fixture
with the newest changes and inspect it, perhaps to help in writing real
tests. We incorporate this with the testing suite so we only have to
implement the boilerplate code of setting up test fixtures once, and we
gain that it is now version controlled.

To select just the interactive tests (which just have a fixture and a
breakpoint) run:

.. code:: bash

    pytest -m interactive

To run automated tests:

.. code:: bash

    pytest -m 'not interactive'

TODO: we will probably add more categories in the future for selecting
particular fixtures.

We are also using tox to test against different python versions. To test
against all of the versions they must be installed on the machine in a
directory here called \`PREFIX\`. To let tox see them they must be on
your path so run tox with a modified environment so we don't have to
dingle with the path in an interactive shell and confuse ourselves:

.. code:: bash

    env PATH="$PREFIX/bin:$PATH" tox

To install these different pythons download, unpack and build the python
configuring it to be installed to the prefix:

.. code:: bash

    wget "https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz"
    tar --extract -f Python-3.7.3
    cd Python-3.7.3
    ./configure --prefix=$PREFIX
    make -j 8
    make install

To run tox for a specific environment check which environment names are
possible by looking in the \`tox.ini\` file:

.. code:: bash

    env PATH="$PREFIX/bin:$PATH" tox -r -e py37

Where the \`-r\` option recreates it from scratch.

Code Quality
~~~~~~~~~~~~

You can also lint the code with flake8:

.. code:: bash

    flake8 src/wepy wepy-tests

And get reports on the complexity of our code:

TODO

Profiling
~~~~~~~~~

We also have tests for profiling the performance sensitive parts of our
code.

You will need to install graphviz for this to get nice SVGs of the call
graphs. On ubuntu and debian:

.. code:: bash

    sudo apt install -y graphviz

Testing examples and tutorials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also want to make sure that the tutorials and examples work.

For this we want to emulate the experience of somebody installing it
from scratch and running the examples.

.. code:: bash

    wepy_test_user_install () {

        package='wepy'
        conda deactivate
        install_env="test-${package}-user-install"
        conda env remove -y -n "$install_env"
        conda create -y -n "$install_env" python=3
        conda activate "$install_env"
        conda install -y -c omnia openmm openmmtools
        pip install wepy[all]==1.0.0rc0
    }

    wepy_test_user_master_install () {

        package='wepy_master'
        conda deactivate
        install_env="test-${package}-user-install"
        conda env remove -y -n "$install_env"
        conda create -y -n "$install_env" python=3
        conda activate "$install_env"
        conda install -y -c omnia openmm openmmtools
        pip install mdtraj
        pip install git+https://github.com/ADicksonLab/wepy.git
    }

#. Examples

Writing Tests
~~~~~~~~~~~~~

If you add a feature ideally you should add some sort of test to make
sure it works.

We currently don't do extensive tests at fine grained levels like unit
tests. Largely, I think these are a waste of time for a project like
wepy without a full time developer. These are welcome contributions
however, if anyone finds the time to write them.

Our tests do however try to do some basic integration tests where we
just try to build up and run simulations and perhaps run some analysis
routines just to make sure that your changes or new component can be run
without errors somewhere down the line.

Aside from the automated tests which get run by pytest there are a
number of other useful pieces of code that tend to be useful during the
development or perhaps maintenance cycle. This is a little different
from other repos I have seen, and perhaps adds a little bit of messiness
to the whole thing. It should add however, some value to dealing with
difficult and slippery problems that at least I have encountered in the
day to day of developing a project. Our goal is to have clear boundaries
for quarantining our messiness so that it doesn't inevitably bleed into
the perfectly crystalline purity of the main code base. A complete lack
of messiness (IMO) is either a sign of immense maturity (unlikely) or
premature optimization. So we aim to start treating it as a first class
citizen.

These categories and the related folders are:

tests
    Proper tests that would get run by pytest and your CI/CD pipeline
examples and mocks (harnesses)
    Well-behaved "context" scripts for prototyping, bugfixing, and
    showcasing how to accomplish very specific tasks.
troubleshooting
    Misbehaved "context" scripts for broad domain problem solving. This
    is more oriented towards improving the operation of the repo
    tooling, how installations are failing, how builds are failing etc.
scrapyard
    If you feel too much apprehension in burninating your lovely
    prototype or script park it here to rust in peace.

The harnesses, troubleshooting, and scrapyard folders should be flat.
That means don't nest directories for categorization, instead put it in
the file name. If you can use an org-mode file to contain explanations,
instructions, or multiple code blocks, please do so. It helps immensely
to have all of the necessary context in one artifact if possible. If you
absolutely must have more than one file (if you have config files small
inputs that must be read from the program to operate etc.) for the unit
go ahead and make a directory.

For data that should be stored in git LFS (large file storage) please
put them in either:

lfs-data
    for the automatic 'tests' data. These are relied on being available
    to run the tests and should be kept organized and clean.
lfs-misc
    for all the files that are used by the harnesses, troubleshooting,
    and scrapyard. Although try not to store large data at all for these
    things, or when it is no longer need it remove them from the repo
    and untrack with LFS.

#. Tests

   This is the thing that most developers think of. Basically we run
   pytests and you can write tests like you would for that, so go read
   that documentation.

   I do offer some insights into our focus however. Because we do not
   have unit testing we focus more on building up a collection of useful
   fixtures, which build on each other. This is to approximate some
   integration testing where all the components must work to even get
   the end product.

   The favorite test system is the Lennard-Jones (lj) pair, for which we
   can build a system with no input files, along with a dependency on
   openmmtools.

   The integration tests for this basically amount to just importing the
   fixtures. If the fixture generation part works, then we just pass the
   test.

   We also have a series of special test cases which are tagged as
   'interactive' (which also appears in the test name).

#. Harnesses

   These are scripts and code blocks that build up a mock system or
   "harness" to allow interactive prototyping. These should share code
   through copy-paste. be independent, and never assumed to work how you
   think. Usually you will use one of these for developing the feature
   or component. These should not have a module structure and should be
   copy-pastable into an IPython session or notebook and run using the
   dev virtual environment. If you have explanations or other
   instructions please put them into an org-mode document along with the
   the script in a code block which can be copy-pasted or tangled.

   Because, wepy doesn't have config files and other such things you
   should be able to put everything into a single python code block.
   This is kind of the litmus test for whether it belongs in the
   harnesses or not. If you have to set new virtual envs or do
   reinstallations etc. your problem is in troubleshooting. The only
   exception is if you are prototyping something that brings in a new
   library, which should be rare for core wepy. If this is the case,
   consider that you should be doing this in a separate repo. The idea
   is that these code blocks should be runnable version to version and
   the only thing that might break is the API calls.

#. Troubleshooting

   Scripts and code block/prose for specific contexts that used for
   problem solving. Ideally, once a troubleshooting problem is fixed
   there should be no need for the file, so go ahead and remove it,
   unless you suspect the problem will rear its ugly head again. These
   contexts, typically are for pathological cases and as such may
   involve tweaking environmental knobs like virtual envs, package
   versions, OS env variables, etc. So you probably should be writing an
   org mode file (or I guess markdown; whatever floats your boat) that
   is very detailed in the process providing copy-pasted outputs from
   your terminal etc. Try to include a date or commit that you are
   working from so future devs know what to clean out based on how old
   it is. If that is you don't clean up your own mess.

#. Scrapyard

   Really this is just a dumping ground for half-baked, forgotten, or
   dead end things that never went anywhere. They can be prototypes,
   deprecated modules, harnesses, troubleshooting scripts, anything. We
   make no effort to organize anything in here at all. The idea is that
   if you have this nagging feeling in the back of your mind that you
   really shouldn't completely delete that thing and lose it forever. Of
   course if it is in the git history it is safe (sort of), but no one
   goes digging in git history for parts and pieces, its more useful for
   merging branches and recovering when things go horribly wrong.

   That said don't be offended if your old scrap pieces get deleted.
   There are no naming conentions and there never should be here. If you
   have "picked parts" they probably should go in harnesses.

Contributing
------------

TBD

Architecture
============

Record Groups
-------------

The protocol by which non-trajectory data is given by the resampler and
boundary conditions (BC) is unified that makes it simpler to save in
formats like HDF5.

The resampler and BC both have multiple record groups:

-  resampler

   -  resampling
   -  resampler

-  BC

   -  warping
   -  progress
   -  boundary conditions

A record group can be thought of as a single table in a relational
database. Each record group corresponds to a class of events that occur
and each record in a record group corresponds to one event.

Record groups can be **continual** or **sporadic**.

A continual record is recorded once per cycle. A continual record
reports on the event of a cycle.

A sporadic record can be reported 0 or many times per cycle and responds
to the event determined by the record group.

-  continual

   -  progress

-  sporadic

   -  resampler
   -  resampling
   -  warping
   -  boundary conditions

As you can see currently most records are sporadic. This distinction is
really only used internally within the ``WepyHDF5`` class to distinguish
how it stores them, but this distinction is useful in data analysis as
well.

Resampling Records

The ='resampling'= records are probably the most important records for
``wepy`` because they are what records the cloning and merging of
walkers.

Without the ='resampling'= your ``wepy`` simulation would have been
wasted since you no longer will know the history of any given frame. You
will just have a bag full of unconnected pictures.

Records for ='resampling'= happen for each "assignment" event of a
walker during resampling, this minimally should contain two fields:
='decision\ :sub:`id`'= and ='target\ :sub:`idxs`'=.

The ='decision\ :sub:`id`'= is an integer corresponding to an
enumeration of the possible decisions that can be made as to the fate of
the walker during resampling. While technically these decisions are also
modular it is likely that 99.9% of all users will use the
``CloneMergeDecision``.

Detailed knowledge of this formalism is not usually needed in the
practice of writing resamplers that behave well, which is another topic,
and the next few paragraphs can be safely skipped.

The enumerated decisions in this are:

+------------------+-----+
| ``NOTHING``      | 1   |
+------------------+-----+
| ``CLONE``        | 2   |
+------------------+-----+
| ``SQUASH``       | 3   |
+------------------+-----+
| ``KEEP_MERGE``   | 4   |
+------------------+-----+

The ``NOTHING`` decision means don't clone or merge this walker.

``CLONE`` means clone this walker.

``SQUASH`` and ``KEEP_MERGE`` are related in that both involve merging.

A single merge includes a set of walkers that will be merged together,
there must be at least 2 such walkers in this "merge group".

From the merge group only a single *state* will be preserved in the
single resulting walker, while the weight of the final walker will be
the sum of all those walkers.

The state of the final walker will be drawn from the set of walkers in
the merge group based on the behavior of the resampler (usually a choice
weighted by their weights), but will always be identical to one of the
walkers. The walker with the chosen state is the ``KEEP_MERGE`` walker.
The rest are the ``SQUASH`` walkers.

The second field, ='target\ :sub:`idxs`'=, actually determines which
walkers will be merged with what other walkers, and is a tuple of
integers indicating the location, or slot.

A 'slot' is simply an available position in the lineup of walkers that
will be simulated in a single cycle of WE. The number of slots is the
number of walkers that will be simulated in the next cycle.

As an aside: In general the number of walkers used in a WE simulation is
not specified (other than there needs to be more than 1). You can have a
constant number of walkers, or a dynamic one with the number fluctuating
during the simulation.

If you have too small a number of walkers then you will have a
relatively sparse coverage of the sample space.

If you have too many the cycle throughput will be very slow.

Additionally, simulations run with GPUs will want to have a number of
walkers each cycle that is a multiple of the number of GPUs or a number
of the GPUs will be lying idle when the task queue of running walker
runner segments is depleted.

So typically there is some constraint on the the number of slots
available in the next WE cycle. The constraint is decided on and
enforced by the resampler. So if there is a mismatch in the resampling
records and the walkers produced the ``wepy`` simulation manager will
not complain.

WARNING: Currently the ``WepyHDF5`` storage backend and reporter do not
support dynamic numbers of simulations. While technically the none of
the other code has any problem with this.

The ='target\ :sub:`idxs`'= value for ``NOTHING`` and ``KEEP_MERGE`` is
a 1-tuple of the integer index of slot where the resultant walker will
be placed.

The ='target\ :sub:`idxs`'= for ``CLONE`` is an n-tuple of integer
indices of slots where n is the number of children of the clone and n
must be at least 2 (or it would've been a ``NOTHING``).

The ='target\ :sub:`idxs`'= of ``SQUASH`` is also a 1-tuple like
``NOTHING`` except since a ``SQUASH`` has no child it indicates the
``KEEP_MERGE`` walker that it's weight is added to. Note that this slot
index is the slot index that the ``KEEP_MERGE`` record itself specifies
and not the slot the ``KEEP_MERGE`` walker previously occupied (as that
index is of no consequence to the current collection of walkers).

Thus a ``KEEP_MERGE`` walker defines a single merge group, and the
members of that merge group are given by which ``SQUASH`` targets.

Critically, the ='step\ :sub:`idx`'= and ='walker\ :sub:`idx`'= (slot
index of walker in last cycle) fields should also be supplied so that
the lineage histories can be generated.

In addition to the Decision class record fields any other amount of data
can be attached to these records to report on a resampling event.

For example in the WExplore resampler the region the walker was assigned
to is also given.

Warping Records

The next most important record is the warping records.

These are of course only relevant if you are using boundary conditions,
but among the three BC these are the principal object.

Warping records determine the action that was taken on a walker after it
met the criteria for a boundary condition event.

Minimally it should specify the ='walker\ :sub:`idx`'= that was acted
on, and if any warping event can be discontinuous the 'weight' of it so
this can be accounted for in analysis.

The rest of the specification for boundary conditions does not have a
protocol similar to the one for cloning and merging records and is left
up to the developer of the class to decide.

For simple boundary conditions where there is only one result an
additional field is not even necesary.

The colored trajectories examples provides a possible example. In this
case you could have a field called ='color'= which is the new "color" of
the walker which indicates the last boundary it crossed and could be a
string or an integer enumeration.

Boundary Condition Records

This and all the other record groups are really optional.

A single boundary condition record reports on the event of a change in
the state of the boundary condition object.

For example if the cutoff value for a ligand unbinding boundary
condition changes during a simulation.

Resampler Records

These records report on events changing of the state of the resampler.

For example in WExplore a single record is generated every time a new
region/image is defined giving details on the values that triggered this
event as well as the image that was created.

This interpretation is semantically useful but in practice this reporter
could also report on collective attributes of the walkers, such as
all-to-all distances or histograms of the current batch of walkers.

Its up to the writer of the resampler to decide.

Progress Records

Progress records are provided mainly as a convenience to get on-line
data analysis of walkers during a simulation.

For instance in ligand unbinding the progress may be the distance to the
cutoff, or RMSD to the original state.

While the active observer may note that these calculations may also have
been implemented in a reporter as well.

There are a few tradeoffs for that approach though.

One, the value may have already been calculated in the process of
evaluating walkers for warping and double calculation is potentially
unacceptably wasteful (although one might imagine complex systems where
reporters perform their actions asynchronously to the flow of the
simulation manager moving onto new cycles).

Second, the flow of data will be forked. For example when using the
``WepyHDF5Reporter`` all the data it will report on is assumed to be
contained in records returned by the runner, resampler, and boundary
conditions and can't know of another reporter. Nor is it easy nor wise
to have two reporters acting on the same database.

Perhaps such analysis could be implemented as analysis submodules in the
``WepyHDF5Reporter`` to keep a single stream of data, if you think that
way go ahead and make a pull request.

Specifying Record Group Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each record group should have three class constants defined for it.

This is strictly not necessary from the perspective of either the
simulation manager or the primary consumer of these records, the
``WepyHDF5Reporter``, but is a very good practice as it will help catch
bugs and will clarify the results your BC or resampler will produce for
those inspecting them.

The three definitions are:

-  field names
-  shapes
-  dtypes

Each should be defined as a class constant prefixed by the name of the
record group followed by the definition type, for example the resampling
record group of WExplore looks like this:

.. code:: python

    DECISION = MultiCloneMergeDecision
    RESAMPLING_FIELDS = DECISION.FIELDS + ('step_idx', 'walker_idx', 'region_assignment',)
    RESAMPLING_SHAPES = DECISION.SHAPES + ((1,), (1,), Ellipsis,)
    RESAMPLING_DTYPES = DECISION.DTYPES + (np.int, np.int, np.int,)

For the "fields" this is the name of the field and should be a string.
In the example we are using fields defined from the
``MultiCloneMergeDecision`` class.

The shapes are the expected shapes of a single element of the field.
Three types of values are accepted here:

A. A tuple of ints that specify the shape of the field element array.

B. Ellipsis, indicating that the field is variable length and limited to
being a rank one array (e.g. ``(3,)`` or ``(1,)``).

C. None, indicating that the first instance of this field will not be
known until runtime. Any field that is returned by a record producing
method will automatically interpreted as None if not specified here.

Note that the shapes must be tuple and not simple integers for rank-1
arrays.

It is suggested that if possible use option A. Option B will use a
special datatype in HDF5 for variable length datasets that can only be 1
dimensional, in addition to being much less efficient to store.

Option C is not advisable but is there because I know people will be
lazy and not want to define all these things. By defining things ahead
of time you will reduce errors by catching differences in what you
expect a field to look like and what you actually receive at runtime.

If you are actually saving the wrong thing and don't specify the shape
and dtype then you may run weeks of simulations and never realize you
never saved the right thing there.

The dtypes have similar options but there is no Ellipsis option.

Each non-None dtype should be a numpy dtype object. This is necessary
for serializing the datatype to the HDF5 (using the
``numpy.dtype.descr`` attribute).

Record Fields
~~~~~~~~~~~~~

One additional class constant can be defined to make analysis in the
future easier.

When accessing records from a ``WepyHDF5`` object you can automatically
generate ``pandas.DataFrames`` from the records, which will select from
a subset of the fields for a record group. This is because large arrays
don't fit well into tables!

So you can define a subset of fields to be used as a nice "table" report
that could be serialized to CSV. For instance in WExplore's resampler
record group we leave out the multidimensional ='image'= field:

.. code:: python

    RESAMPLER_FIELDS = ('branching_level', 'distance', 'new_leaf_id', 'image')
    RESAMPLER_SHAPES = ((1,), (1,), Ellipsis, Ellipsis)
    RESAMPLER_DTYPES = (np.int, np.float, np.int, None)

    # fields that can be used for a table like representation
    RESAMPLER_RECORD_FIELDS = ('branching_level', 'distance', 'new_leaf_id')

Again, its not necessary, but its there to use.
