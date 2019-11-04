Installation
============

To install from pip (which may be out of date):

.. code:: bash

    pip install wepy[all]

Which will install most dependencies.

To install the optional, but likely desired feature, OpenMM you should
probably just use the omnia channel on conda, otherwise you will need to
compile it yourself.

.. code:: bash

    conda install -c omnia openmm

There are some optional features you can install as well using the
"extras" spec in pip. The base package aims to be very easy to install
but lacks a lot of functionality that makes wepy truly useful.
Currently, these are:

mdtraj
    allows export of structures to the mdtraj format which has many
    writers for serialization, e.g. pdb, dcd, etc.
all
    installs all extras

Check the setup.py file under 'extras:sub:`requirements`' for the full
listing.

.. code:: bash

    pip install wepy[mdtraj]

You can always install from git as well for the latest:

.. code:: bash

    pip install git+https://github.com/ADicksonLab/wepy.git

If installation went alright you should have this command line interface
for working with orchestration available:

.. code:: bash

    wepy --help

