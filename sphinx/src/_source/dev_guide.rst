Development Guide
=================

Overview
--------

In this project we use a bunch of extra tools that simplify the drudgery
of manual maintenance tasks so we can get more coding done. Its also
probably not how your used to.

This "middleware" includes:

`invoke <https://www.pyinvoke.org/>`__
   for creating the runnable endpoints or targets.
`jubeo <https://github.com/salotz/jubeo.git>`__
   for importing and updating a standard set of project independent
   invoke targets and endpoints.

``invoke`` is common but ``jubeo`` is a creation of my own and may still
have some rough edges.

Furthermore, this project is made from a `cookiecutter
template <https://github.com/salotz/salotz-py-cookiecutter.git>`__ to
bootstrap it. You may find some odd stubs around and that is why. Feel
free to get rid of them. If you ever want them back you can transcribe
from the source.

Ideally you won't have to do much outside of running the ``invoke``
targets (i.e. any command that starts with ``inv``).

To see all of the commands run:

.. code:: bash

   inv -l

You should read and understand the ``jubeo`` documentation so that you
know how to add your own project-specific targets via ``invoke``.

Getting Set Up
--------------

Obtaining the source code
~~~~~~~~~~~~~~~~~~~~~~~~~

For the source code:

.. code:: bash

   git clone {{ cookiecutter.dev_url }}
   cd {{ cookiecutter.project_name }}

Tooling
~~~~~~~

The project comes configured for use with ``jubeo`` (see the ``.jubeo``
directory) but without the tasks imported.

To get started you will need to install ``jubeo`` and then run this from
the project directory:

.. code:: bash

   jubeo init .

This will download the appropriate ``invoke`` tasks organized into a
special structure along with the necessary dependency specs to use them.

To be able to run the tasks you should install these dependencies:

.. code:: bash

   pip install -r .jubeo/requirements.in

If you add to the libraries needed (through plugins discussed later) you
will need to edit the ``.jubeo/requirements.txt`` file and recompile the
``.jubeo/requirements.in`` file by either manually running
``pip-compile`` or using the ``inv core.pin-tool-deps`` target.

Configuring
~~~~~~~~~~~

We typically manage configuration values in the ``tasks/config.py`` and
``tasks/sysconfig.py`` files as opposed to global system environment
variables.

See the ``jubeo`` documentation on how to use these configuration files.

For this python project template we do need to set some values before
all of the features will work. We also avoid setting these in shell
configuration variables and as of now it is just up to the user to
customize these. To the ``tasks/config.py`` file add the following:

.. code:: python

   PROJECT_SLUG = "{{cookiecutter.project_slug}}"
   VERSION="{{cookiecutter.initial_version}}"

Just make sure to update the version string here when you do releases
(included in the checklist for releases).

Virtual Environments
~~~~~~~~~~~~~~~~~~~~

There are helpers for pinning and (re)generating python virtual
environments which are helpful in developing and testing this project,
and not necessarily just for running it as a user. See *Managing
Dependencies* for details on managing dependencies of the installable
project.

If an environment has been already been written and compiled you need
only create it locally and then activate it.

To create an env called ``dev`` just run the ``env`` (``env.make``)
target from ``invoke``:

.. code:: bash

   inv env -n dev

If it fails double check that all the dependencies have been compiled.

If it still fails, likely the environment is meant to be used for
simultaneous development of multiple projects. You can check which
installable source repos are expected in which locations by looking at
the ``self.requirements.txt`` file. If there are simultaneous dev
requirements make sure these source repos can be found at those
locations.

Then follow the activation instructions that are printed as different
projects might use different backends.

For pure python projects the default ``venv`` tool should be used, but
``conda`` is also an option.

For ``venv`` envs they will be stored in a directory called ``_venvs``
and for conda ``_conda_envs`` (this is customizable however). Simply:

.. code:: bash

   source _venvs/dev/bin/activate_

or

.. code:: bash

   conda activate _conda_envs/dev

In any case the environments are not stored with other user-level
environments, what we call *ambient* environments, and are instead
stored in the project directory.

If you ever have problems with an environment just rerun the
``env.make`` target to get a clean one. A practice we encourage to do
frequently so that developers don't diverge in their envs with local
modifications. So while you can make your env, try to use this one
unless you have problems.

We maintain a number of preconfigured environments in the ``envs``
directory which are used for different purposes. Calling ``inv env -n
dev`` is the same as ``inv dev`` since it is the default, but any other
environment can be created by passing the matching name. For instance
there is an environment that mimics the user's installation environment
so that we can test experiences upon install, to make sure we haven't
accidentally depended on something in the dev env:

.. code:: bash

   inv env -n test_install

Maintenance Tasks
-----------------

Managing Dependencies
~~~~~~~~~~~~~~~~~~~~~

#. Quick Reference

   To initially pin an environment or when you add requirements run this
   target:

   .. code:: bash

      inv env.deps-pin -n dev

   To update it (should be accompanied by a reason why):

   .. code:: bash

      inv env.deps-pin-update -n dev

   The best practice here is to make initial pinning and updating a
   single commit so that it can easily be rolled back or patched e.g.:

   .. code:: bash

      git add envs/*
      git commit -m "Updates dev environment"

#. Explanation

   Reminder that there are two separate goals of managing dependencies
   and where they are managed:

   Python Libraries
      These dependencies are managed in ``setup.py`` and in PyPI or
      other indices.
   Python Applications/Deployments
      These are dependencies managed in ``requirements.in`` and
      ``requirements.txt`` and used for developer environments and
      deployment environments.

   In this template project there are a number of different places
   dependencies are managed according to both of these purposes. As far
   as the python library specs are concerned it is simpler and well
   documented elsewhere. In this template we introduce a few other
   mechanisms for managing development environments. They are as follows
   with the specific purpose of them:

   ``setup.py``
      specifying high level requirements for installation of a released
      version from an index by a user or system integrator.
   ``tools.requirements.txt``
      A bare minimum high-level listing of dependencies necessary to
      bootstrap the creation of development environments from the
      project tooling itself. You are free to install these in any
      ambient environment you see fit. We suggest using something like
      ``pyenv-virtualenv``.
   ``envs/env_name`` dirs
      a directory with a set of files that are used to reproduce
      development environments the full structure will be discussed
      separately. There can be any number of these but they shouldn't
      start with a double-underscore '__' which is used for temporary
      utility environments.
   ``requirements.in``
      An optional high-level specification of install dependencies
      readable from other projects for simultaneous development. Should
      be the same as ``setup.py`` install dependencies.

   The biggest concern for developers is writing env specs in the
   ``envs`` dir. These add a few features a simple
   ``requirements.in/requirements.txt`` file can't solve alone. Here is
   the full listing of possible files that can be authored by the
   developer in this directory:

   ``requirements.in``
      (required) abstract specification of packages
   ``self.requirements.txt``
      (required) how to install packages actively being worked on
   ``dev.requirements.list``
      A list of paths to other ``requirements.in`` files that will be
      included in dependency compilation with this env.
   ``pyversion.txt``
      the python version specified (if supported)

   This also supports the use of ``conda`` for managing environments,
   although this isn't recommended for python packages which are not
   intended to be distributed via ``conda``. This is however, useful for
   projects like the ``analytics-cookiecutter`` project which won't
   actually be distributed to others as general purpose. For this you
   need only add another file for the abstract conda dependencies:

   -  ``env.yaml`` (required for conda managed envs) an abstract
      specification for dependencies. Compiled to ``env.pinned.yaml``

   All the other files are still valid for conda environments still.

   #. Abstract Requirements

      ::

         requirements.in

      The basic part of this spec is the ``requirements.in`` and
      ``self.requirements.txt`` files.

      The ``requirements.in`` file is as described in the ``pip-tools``
      documentation (i.e. ``pip-compile requirements.in``).

      Running ``inv env.deps-pin`` will compile this file to a
      ``requirements.txt`` file, which can then be used to create an
      environment via ``inv env`` (i.e.
      ``pip install -r requirements.txt``).

      It should look something like this:

      .. code:: pyreq

         requests
         networkx >= 2

      There should be no entries like ``-e .`` for installing the
      package or any local file paths. This should be portable between
      machines and developers.

   #. Development Project Installation Spec

      ::

         self.requirements.txt

      The ``self.requirements.txt`` file instead is where these kinds of
      specifications should be.

      At its simplest it may look like this:

      .. code:: pyreq

         -e .

      Which means just to install the package of this current repo.

      However, it is often that you are working on multiple separate
      projects at once in different version control repos and want to
      develop simultaneously without either releasing them every time
      you want to make changes or even push them to a git repo. You can
      then write a ``self.requirements.txt`` file that looks like this:

      .. code:: pyreq

         -e .

         -e ../other_project
         -e $HOME/dev/util_project

   #. Simultaneous Project Development

      ::

         dev.requirements.list

      During simultaneous development however, the dependencies of these
      other repos won't be included in the compilation of the
      ``requirements.txt`` file.

      Your options are to:

      #. manually transcribe their dependencies into the env's
         ``requirements.in`` file (not recommended)
      #. write top-level ``requirements.in`` files for each project and
         include paths to these files in the
         ``envs/env_name/dev.requirements.list`` file.

      The tooling here provides support for the second one. For this you
      must write a ``list`` text file (see
      `rfc:salotz/016\ trivial-plaintext-formats <https://github.com/salotz/rfcs/blob/master/rfcs/salotz.016_trivial-plaintext-formats.org#a-list-file>`__
      for a discussion of the format), where each line should be a path
      to a ``requirements.in`` file, e.g.:

      .. code:: trivial-list

         ../other_project/requirements.in
         $HOME/dev/util_project/requirements.in

      This will include each of these files in the dependency
      compilation step. Note that the ``requirements.in`` can come from
      any location and is not a specification other projects *must*
      support.

   #. Meta-Tools Installation Spec

      ::

         tools.requirements.txt

      Use this to "pin" or constrain versions of tools which won't be or
      can't be managed by the pinning tool (i.e. ``pip-tools``, meaning
      ``pip``, ``setuptools`` etc.).

      The main use of this to pin the version of ``pip`` in case it
      breaks some other tools.

   #. Specifying Python Version

      ::

         pyversion.txt

      This file should only contain the text that specifies the version
      of python to use that is understood by the env method (e.g.
      ``conda``).

      E.g.:

      .. code:: fundamental

         3.7.6

      Only the ``conda`` method supports this as of now.

      For the ``venv`` method it is still encouraged to write this file
      though, as a warning will be generated to remind you.

      For managing different python versions we recommend using
      something like ``pyenv`` and we may integrate with this or
      manually specifiying interpreter paths in the future.

Documentation and Website
~~~~~~~~~~~~~~~~~~~~~~~~~

#. Writing Documentation

   The primary source of the documentation is in the ``info`` folder and
   is written in Emacs org mode.

   Because of the powerful wiki-like capabilities of org mode it can
   serve as a primary source for reading docs. This obviously serves the
   devs more than end user's expecting an HTML website it is a good
   first measure for writing docs.

   Org-mode documents can be converted to RestructuredText files for use
   in generators like ``Sphinx`` (for documentation) or ``Nikola`` (for
   static sites) using the ``pandoc`` tool which we expect to be
   installed.

   Furthermore, org-mode & Emacs provides excellent facilities for
   writing foreign source code blocks which allow for literate documents
   which can easily be tangled into code files that can then be tested
   automatically.

   The documentation can roughly be broken down into three major parts:

   pages
      Documents the actual project this repo is about. Should always be
      tested with the same version of the software it is released with.
      Should not include extra dependencies.
   examples & tutorials
      Extended documentation of the project, however this may include
      extra dependencies of the project. These are tested separately
      from the pages documentation.
   meta
      General information about the project management itself. Will not
      be tested and should only contain source code in extremely small
      doses.

   If you write code blocks in your documentation (which is highly
   recommended) you **must** at least write tests which run the code to
   make sure it at least runs.

   When you write code blocks you should use this format:

   .. code:: org

      #+begin_src python :tangle ex0.py
        print("Hello!!!")
      #+end_src

   Notice there is no extra paths to get the tangling right. The tooling
   for running the tests will take care of setting up an environment for
   tangling scripts as the docs shouldn't really be tangled in place in
   the ``info`` tree.

#. Writing Examples & Tutorials

   For our purposes as devs examples and tutorials are almost the same
   in structure. The distinction is mostly for end users that have
   different expecations from examples over tutorials.

   Examples should be provide less explanation whereas tutorials are
   likely to be long form prose documents with literate coding and may
   even provide media like graphs and pictures.

   Examples can also be literate but they are restricted to formats like
   org mode, whereas the tutorials may also be in formats like Jupyter,
   which integrate well with Sphinx docs.

   #. Initializing a Tutorial/Example

      To write examples and tutorials that play nice with testing and
      the basic rules of the examples (described in the
      `users\ guide <./users_guide.org>`__) there are some templates
      available in the ``templates`` directory for
      ``templates/examples``, ``templates/tutorials``, and environment
      specs ``templates/envs``.

      You can either just copy these templates over or use the targets:

      .. code:: bash

         inv docs.new-example --name='myexample' --template='org'
         cp -r -u templates/envs/conda_blank info/examples/myexample/env
         inv docs.new-tutorial  --name='mytutorial --template=jupyter'
         cp -r -u templates/envs/conda_blank info/examples/mytutorial/env

      After you have your directory set up there are some things to keep
      in mind while you are constructing your tutorial.

   #. Managing Dependencies and Envs

      First, write source either in the literate document
      (``README.org``) or in the source file. Not both, unless you
      intend to test both separately. For tutorials you should prefer to
      write them directly in the literate doc, but particularly long and
      uninteresting pieces of code can be put into the source.

      As you write the code pay attention to your dependencies and
      virtual environment. If you add new dependencies, add them to the
      ``requirements.in`` or ``env.yaml`` file and compile:

      .. code:: bash

         cd $PROJECT_DIR
         inv docs.pin-example -n 'myexample'

      You can then make the env 2 ways (the latter is intended to be run
      by users who don't want to be overwhelmed by all the dev options):

      .. code:: bash

         cd $PROJECT_DIR
         inv docs.env-example -n 'myexample'

      or

      .. code:: bash

         inv env

   #. Writing Code Examples

      When writing examples and tutorials you should manually write the
      tangle targets to be the ``_tangle_source`` folder:

      .. code:: org

         Here is some code I am explaining that you will run:

         #+begin_src python :tangle _tangle_source/tutorial.py
           print("Hello!")
         #+end_src

      As stated in the user's guide if you don't follow this rule (or
      any others) then **it is wrong** and an issue should be filed.

      When using input files, please copy them to the ``input`` dir and
      reference them relative to the example dir. So that when you
      execute a script:

      .. code:: bash

         python source/script.py

      The code for reading a file would look like:

      .. code:: python

         with open("input/data.csv", 'r') as rf:
             table = rf.read()

      and not:

      .. code:: python

         with open("../data.csv", 'r') as rf:
             table = rf.read()

      Similarly writing and creating files should be done into the
      ``_output`` dir:

      .. code:: python

         with open("_output/test.txt", 'w') as wf:
             wf.write("Hello!!")

   #. Adding to the built documentation

      The tutorial README files will be automatically converted to
      ReStructuredText and built into the documentation, but in order to
      have links to them from the Tutorials page you will need to
      manually add them to the table of contents section in the
      ``sphinx/tutorials_index.rst`` file, e.g.:

      .. code:: rst

         .. toctree::
            :maxdepth: 1

            tut0/README
            tut1/README

#. Testing Documentation

   There is a folder just for tests that target the docs
   ``tests/test_docs``. You should be able to run them after tangling:

   .. code:: bash

      inv docs.tangle
      inv docs.test-example
      inv docs.test-tutorial
      inv docs.test-pages

   See these targets for more fine-grained tests or to run them using
   ``nox`` for the python version matrix or just to have a more minimal
   and reproducible test environment.

   .. code:: bash

      inv -l | grep docs.test

#. Building Documentation

   To compile and build the docs just run:

   .. code:: bash

      inv py.docs-build

   Which will output them to a temporary build directory
   ``_build/html``.

   You can clean this build with:

   .. code:: bash

      inv py.clean-docs

   To view how the docs would look as a website you can point your
   browser at the ``_build/html`` folder or run a python http web server
   with this target:

   .. code:: bash

      inv py.docs-serve

#. Building and testing the website

   For now we only support deploying the sphinx docs as a website and on
   github pages (via the ``gh-pages`` branch, see *Website Admin*).

   So to view your undeployed docs just run:

   .. code:: bash

      inv py.docs-serve

   And open the local URL.

   Once you are happy with the result, **you must commit all changes and
   have a clean working tree** then you can push to github pages:

   .. code:: bash

      inv py.website-deploy

   Basically this checks out the ``gh-pages`` branch merges the changes
   from ``master`` builds the docs, commits them (normally these files
   are ignored), and then pushes to github which will render them.

   We may also support other common use cases in the future as well like
   Gitlab pages or a web server (via rsync or scp).

   We also will support a more traditional static site generator
   workflow instead of relying in addition to the sphinx docs.

   #. 

#. Deploying the website

   We are using github pages. To avoid having to keep the entire built
   website in the main tree we use the alternate ``gh-pages`` branch. To
   make this process easy to deploy we have a script
   ``sphinx/deploy.sh`` that checks the ``gh-pages`` branch out, does
   some necessary cleaning up, and copies the built website to the
   necesary folder (which is the toplevel), commits the changes and
   pushes to github, and then returns to your working branch.

   The invoke target is:

   .. code:: bash

      inv docs.website-deploy

Testing
~~~~~~~

This is about testing the actual source tree (see *Testing
Documentation* for testing the docs).

#. Testing in the Dev Cycle

   You can either test in the ``dev`` (or ``test``) environment while
   working:

   .. code:: bash

      inv py.tests-all

   There are specific commands for each section of tests, primarily:

   .. code:: bash

      inv py.tests-integration
      inv py.tests-unit

   If you use the ``-t`` option you can specify a tag. The tag will be
   used as an identifying string for output to reports etc. Currently it
   will generate test results into the ``reports``

#. Automated Test Matrix

   We use ``nox`` as the runner for parametrizing and running tests in
   isolated environments for the test matrix. See the ``noxfile.py`` on
   how this is configured.

   You can run the "session" directly since there are other session
   definitions for docs etc.:

   .. code:: bash

      nox -s test

   There is also a target for this:

   .. code:: bash

      inv py.tests-nox

#. Auxiliary "tests"

   We also have two other "testing" targets for the benchmarks and the
   "interactive" tests.

   Benchmarks have a special toolchain for recording and publishing them
   as metrics.

   The 'interactive' tests are just tests which have something like a
   ``breakpoint()`` in them. This is kind of an experimental thing, and
   probably more useful for you to write and call them individually for
   different purposes. The idea is that you can write "tests" that
   generate something like realistic live environment (kind of like
   integration tests) that you can drop into a debugger with and poke
   around in.

Code Quality Metrics
~~~~~~~~~~~~~~~~~~~~

Just run the end target:

.. code:: bash

   inv quality

This will write files to ``metrics``.

Releases
~~~~~~~~

The typical pre-requisites for a release are that:

-  the documentation has been updated and tested
-  the tests have been run and results are recorded
-  the quality metrics have been run and are recorded
-  the changelog has been written

Making a release then follows these steps:

#. test the build
#. make a pre-release build and publish
#. make the release build and publish
#. build and publish documentation, website, etc.

#. Writing and/or Generating the Changelog and Announcement

   Simply go into the ``info/changelog.org`` file and write it.

   There are conventions here per-project. Follow them.

#. Choosing a version number

   There are some semantics around changing the version number beyond
   the 'semver' sense of the 'v.X.Y.Z' meanings.

   To make a release do some changes and make sure they are fully tested
   and functional and commit them in version control. At this point you
   will also want to do any rebasing or cleaning up the actual commits
   if this wasn't already done in the feature branch.

   If this is a 'dev' release and you just want to run a version control
   tag triggered CI pipeline go ahead and change the version numbers and
   commit. Then tag the 'dev' release.

   If you intend to make a non-dev release you will first want to test
   it out a little bit with a release-candidate prerelease.

#. Changing the version number

   You can check the current version number with this command:

   .. code:: bash

      inv py.version-which

   The places where an actual version are needed are:

   -  ``setup.py``
   -  ``sphinx/conf.py``
   -  ``src/package/__init__.py``
   -  ``tasks/config.py``
   -  ``conda/conda-forge/meta.yaml`` (optional)
   -  the git tag

   The ``setup.py`` and ``src/package/__init__.py`` version is handled
   by ``versioneer`` using the git tag for the release. This allows for
   fine-grained versions using git hashes on "dirty" releases.

   The ``sphinx/conf.py`` just gets the current version from
   ``__init__.py`` so it is also downstream of versioneer.

   So currently only the ``tasks/config.py`` and conda versions need to
   be updated manually.

   In this project we never like to initiate configuration tasks at the
   REPL/shell so we never actually run ``git tag`` under normal
   circumstances.

   Instead we configure the desired version "bump" in one place
   ``tasks/config.py`` and then generate the rest downstream through
   ``invoke`` endpoints.

   So simply edit the ``tasks/config.py`` ``VERSION`` variable and then
   run:

   .. code:: bash

      inv git.release

   Which will write the git tag in the correct format. ``versioneer``
   takes over from there.

   Here then is the checklist of manually edited versions (currently the
   conda packaging stuff is not supported):

   -  [ ] edit ``tasks/config.py``
   -  [ ] commit
   -  [ ] run ``inv git.release``

   Changing the version may happen a few times through the release
   process in order to debug wrinkles in the process so its useful to
   have this workflow in mind.

#. Release Process

   #. Testing the build

      To test a build with whatever work you have go ahead and run:

      .. code:: bash

         inv py.build

      And then try to install it from an empty environment:

      .. code:: bash

         inv env -n test_install

      Activate the environment e.g.:

      .. code:: bash

         source _venv/test_install/bin/activate

      or

      .. code:: bash

         conda activate _conda_envs/test_install

      then run it for each build, e.g.:

      .. code:: bash

         pip install dist/BUILD.tar.gz

      They should all succeed. You should also test the installation
      somehow so that we know that we got the dependencies correct.

   #. Making the Pre-Release

      All releases should be preceded by a release candidate just to
      make sure the process is working as intended.

      So after this testing of your potentially "dirty" tree (which is
      anything that is not equal to a 'vX.Y.Z.etc' git tag) change the
      versions to have 'rc0' at the end of the new intended (semantic)
      number, e.g. ``v1.0.0.rc0``.

      Then go ahead and commit the changes with a message like this:

      .. code:: fundamental

         1.0.0rc0 release preparation

      Then do the git release (just tags it doesn't 'publish' it) and
      rebuild before doing the next steps:

      .. code:: bash

         inv git.release
         inv py.build

      Once you have built it and nothing is wrong go ahead and publish
      it to the test indexes (if available):

      .. code:: bash

         inv py.publish-test

      You can test that it works from the index using the same
      ``test_install`` environment above.

      And install the package from the test repo with no dependencies:

      .. code:: bash

         pip install --index-url https://test.pypi.org/simple/ --no-deps package

      Then you can publish this pre-release build.

      Publishing the results will vary but you can start with publishing
      the package to PyPI and the VCS hosts with the real publish
      target:

      .. code:: bash

         inv git.publish
         inv py.publish

   #. The final public release

      Edit the version number to something clean that won't be hidden on
      PyPI etc.

      Then:

      .. code:: bash

         inv git.release
         inv py.build
         inv py.publish
         inv git.publish

Initializing this repository
----------------------------

These are tasks that should only be done once at the inception of the
project but are described for posterity and completeness.

Version Control
~~~~~~~~~~~~~~~

First we need to initialize the version control system (``git``):

.. code:: bash

   inv git.init

If you want to go ahead and add the remote repositories for this
project. We don't manage this explicitly since ``git`` is treated mostly
as first class for these kinds of tasks and is better left to special
purpose tools which are well integrated and developed.

Python Packaging
~~~~~~~~~~~~~~~~

There is a target to initialize python specific packaging things. This
is because some tools (like ``versioneer``) need to be generated after
project instantiation.

Make sure you have a clean tree so you can see the changes then:

.. code:: bash

   inv py.init

then inspect and commit.

Compiling Dependencies
~~~~~~~~~~~~~~~~~~~~~~

Then add any extra dependencies you want to the development environment
`requirements.in <../envs/dev/requirements.in>`__ file and then compile
and pin them:

.. code:: bash

   inv env.deps-pin -n dev env.deps-pin -n test_install

Then commit this.

Creating Environments
~~~~~~~~~~~~~~~~~~~~~

Then just create the virtual environment. For portability we use the
builin ``venv`` package, but this is customizable.

.. code:: bash

   inv env

Then you can activate it with the instructions printed to the screen.

Website Admin
~~~~~~~~~~~~~

We use Github Pages by default since it is pretty easy. Because we don't
want to clutter up the master branch with website build artifacts we use
the ``gh-pages`` branch approach.

If you just run the ``inv py.website-deploy`` target this will
idempotently take care of setting this up for you.

However, you will need to create it and push it before you can set this
in the github settings for the page.
