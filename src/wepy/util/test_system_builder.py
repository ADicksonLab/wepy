import os
import os.path
import numpy as np

import scipy
import scipy.special
import scipy.integrate

import openmm
from openmm import unit
from openmm import app

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
        self._topology = app.Topology()
        # MDTraj Topology is built on demand.
        self._mdtraj_topology = None

        return

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
    
 
def construct_restraining_potential(particle_indices, K):
    """Make a CustomExternalForce that puts an origin-centered spring on the chosen particles"""

    # Add a restraining potential centered at the origin.
    energy_expression = '(K/2.0) * (x^2 + y^2 + z^2);'
    energy_expression += 'K = %f;' % (K / (unit.kilojoules_per_mole / unit.nanometers ** 2))  # in OpenMM units
    force = openmm.CustomExternalForce(energy_expression)
    for particle_index in particle_indices:
        force.addParticle(particle_index, [])
    return force 
    
class NaCLPair(TestSystem):

    """Create a non-periodic rectilinear grid of Lennard-Jones particles in a harmonic restraining potential.

    Parameters
    ----------
    nx : int, optional, default=3
        number of particles in the x direction
    ny : int, optional, default=3
        number of particles in the y direction
    nz : int, optional, default=3
        number of particles in the z direction
    K : openmm.unit.Quantity, optional, default=1.0 * unit.kilojoules_per_mole/unit.nanometer**2
        harmonic restraining potential
    cutoff : openmm.unit.Quantity, optional, default=None
        If None, will use NoCutoff for the NonbondedForce.  Otherwise,
        use CutoffNonPeriodic with the specified cutoff.
    switch_width : openmm.unit.Quantity, optional, default=None
        If None, the cutoff is a hard cutoff.  If switch_width is specified,
        use a switching function with this width.

    Examples
    --------

    Create Lennard-Jones cluster.

    >>> cluster = LennardJonesCluster()
    >>> system, positions = cluster.system, cluster.positions

    Create default 3x3x3 Lennard-Jones cluster in a harmonic restraining potential.

    >>> cluster = LennardJonesCluster(nx=10, ny=10, nz=10)
    >>> system, positions = cluster.system, cluster.positions
    """

    def __init__(self, nx=3, ny=3, nz=3, K=1.0 * unit.kilojoules_per_mole / unit.nanometer**2, cutoff=None, switch_width=None, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Default parameters for Na and Cl
        mass_Na = 22.99 * unit.amu
        mass_Cl = 35.45 * unit.amu
        q_Na = 1.0 * unit.elementary_charge
        q_Cl = -1.0 * unit.elementary_charge
        sigma_Na = 2.0 * unit.angstrom
        sigma_Cl = 4.0 * unit.angstrom
        epsilon_Na = 0.1 * unit.kilojoule_per_mole
        epsilon_Cl = 0.2 * unit.kilojoule_per_mole

        scaleStepSizeX = 1.0
        scaleStepSizeY = 1.0
        scaleStepSizeZ = 1.0

        natoms = nx * ny * nz

        system = openmm.System()

        nb = openmm.NonbondedForce()

        if cutoff is None:
            nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        else:
            nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
            nb.setCutoffDistance(cutoff)
            nb.setUseDispersionCorrection(False)
            nb.setUseSwitchingFunction(False)
            if switch_width is not None:
                nb.setUseSwitchingFunction(True)
                nb.setSwitchingDistance(cutoff - switch_width)

        positions = unit.Quantity(np.zeros([natoms, 3], np.float32), unit.angstrom)

        atom_index = 0
        for ii in range(nx):
            for jj in range(ny):
                for kk in range(nz):
                    if (atom_index % 2) == 0:  # Alternating Na and Cl
                        mass = mass_Na
                        q = q_Na
                        sigma = sigma_Na
                        epsilon = epsilon_Na
                        element = app.Element.getBySymbol('Na')
                        atom_name = 'Na'
                    else:
                        mass = mass_Cl
                        q = q_Cl
                        sigma = sigma_Cl
                        epsilon = epsilon_Cl
                        element = app.Element.getBySymbol('Cl')
                        atom_name = 'Cl'
                    
                    system.addParticle(mass)
                    nb.addParticle(q, sigma, epsilon)
                    x = sigma * scaleStepSizeX * (ii - nx / 2.0)
                    y = sigma * scaleStepSizeY * (jj - ny / 2.0)
                    z = sigma * scaleStepSizeZ * (kk - nz / 2.0)

                    positions[atom_index, 0] = x
                    positions[atom_index, 1] = y
                    positions[atom_index, 2] = z
                    atom_index += 1

        system.addForce(nb)

        topology = app.Topology()
        chain = topology.addChain()
        for _ in range(system.getNumParticles()):
            residue = topology.addResidue(atom_name, chain)
            topology.addAtom(atom_name, element, residue)
        self.topology = topology

        # Add a restraining potential centered at the origin.
        system.addForce(construct_restraining_potential(particle_indices=range(natoms), K=K))

        self.system, self.positions = system, positions
