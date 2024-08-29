Installation Guide
==================

Welcome to the installation guide for `wepy`. Follow the steps below to set up the environment and install `wepy`.

Prerequisites
-------------

Ensure that you are using **Python 3.10**. It is recommended to set up a conda environment for managing dependencies and ensuring compatibility.

.. code-block:: bash

   conda create -n wepy-env python=3.10
   conda activate wepy-env

Installing `wepy`
-----------------

You can install `wepy` from the latest git repository.

.. Installing from Pip:

.. .. code-block:: bash

..    pip install wepy

Installing from Git (for the latest version):

.. code-block:: bash

   pip install git+https://github.com/ADicksonLab/wepy.git

.. Installing Optional Features
.. ----------------------------

.. `wepy` offers several optional features that enhance its functionality. You can install these using the "extras" spec in pip.

.. **Molecular Dynamics (md):** extra packages for molecular dynamics.

..   .. code-block:: bash

..      pip install wepy[md]

.. **Distributed Analysis (distributed):** extra packages to allow for distributed analysis.

..   .. code-block:: bash

..      pip install wepy[distributed]

.. **Prometheus Monitoring (prometheus):** for monitoring simulations via [[https://prometheus.io][Prometheus]]

..   .. code-block:: bash

..      pip install wepy[prometheus]

.. **All Optional Features (all):** installs all extras

..   .. code-block:: bash

..      pip install wepy[all]

.. For a full listing of optional packages, check the ``extras_requirements`` section in the ``setup.py`` file.

Installing OpenMM and mdtraj
-----------------

To install OpenMM, which is highly recommended for running molecular dynamics simulations:

.. code-block:: bash

   conda install -c conda-forge openmm=8.0.0

If the conda version of openmm doesn't work for your compute environment, consult the `OpenMM documentation <http://docs.openmm.org/latest/userguide/application.html#installing-openmm>`__.

The mdtraj library is used by wepy to create and manipulate system topologies.  It can also be installed via conda:

.. code-block:: bash

   conda install -c conda-forge mdtraj=1.9.9

These version numbers work correctly at the time of this writing and avoid some incompatibility issues with numpy and pandas dependencies.

Verifying Installation
----------------------

After installation, you should be able to import `wepy` in python. Verify it by running:

.. code-block:: bash

   python -c "import wepy"

If the installation was successful, you should not see any errors.
