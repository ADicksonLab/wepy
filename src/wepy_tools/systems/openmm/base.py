import os
import os.path
import numpy as np

import scipy
import scipy.special
import scipy.integrate

import openmm
import openmm.unit as unit
import openmm.app as omma

class TestSystem(object):

    """Abstract base class for test systems, demonstrating how to implement a test system.

    Parameters
    ----------

    Attributes
    ----------
    system : openmm.System
        System object for the test system
    positions : list
        positions of test system
    topology : list
        topology of the test system

    Notes
    -----

    Unimplemented methods will default to the base class methods, which raise a NotImplementedException.

    Examples
    --------

    Create a test system.

    >>> testsystem = TestSystem()

    Retrieve a deep copy of the System object.

    >>> system = testsystem.system

    Retrieve a deep copy of the positions.

    >>> positions = testsystem.positions

    Retrieve a deep copy of the topology.

    >>> topology = testsystem.topology

    Serialize system and positions to XML (to aid in debugging).

    >>> (system_xml, positions_xml) = testsystem.serialize()

    """

    def __init__(self, **kwargs):
        """Abstract base class for test system.

        Parameters
        ----------

        """

        # Create an empty system object.
        self._system = openmm.System()

        # Store positions.
        self._positions = unit.Quantity(np.zeros([0, 3], float), unit.nanometers)

        # Empty topology.
        self._topology = omma.Topology()
        # MDTraj Topology is built on demand.
        self._mdtraj_topology = None

    @property
    def system(self):
        """The openmm.System object corresponding to the test system."""
        return self._system

    @system.setter
    def system(self, value):
        self._system = value

    @system.deleter
    def system(self):
        del self._system

    @property
    def positions(self):
        """The openmm.unit.Quantity object containing the particle positions, with units compatible with openmm.unit.nanometers."""
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value

    @positions.deleter
    def positions(self):
        del self._positions

    @property
    def topology(self):
        """The openmm.app.Topology object corresponding to the test system."""
        return self._topology

    @topology.setter
    def topology(self, value):
        self._topology = value
        self._mdtraj_topology = None

    @topology.deleter
    def topology(self):
        del self._topology

    @property
    def mdtraj_topology(self):
        """The mdtraj.Topology object corresponding to the test system (read-only)."""
        import mdtraj as md
        if self._mdtraj_topology is None:
            self._mdtraj_topology = md.Topology.from_openmm(self._topology)
        return self._mdtraj_topology
    
 
    def construct_restraining_potential(self, particle_indices, K):
        """Make a CustomExternalForce that puts an origin-centered spring on the chosen particles"""

        # Add a restraining potential centered at the origin.
        energy_expression = '(K/2.0) * (x^2 + y^2 + z^2);'
        energy_expression += 'K = %f;' % (K / (unit.kilojoules_per_mole / unit.nanometers ** 2))  # in OpenMM units
        force = openmm.CustomExternalForce(energy_expression)
        for particle_index in particle_indices:
            force.addParticle(particle_index, [])
        
        return force 