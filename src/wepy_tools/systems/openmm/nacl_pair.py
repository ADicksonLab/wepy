from wepy_tools.systems.openmm.base import TestSystem

import numpy as np

import openmm
import openmm.unit as unit
import openmm.app as omma

class NaClPair(TestSystem):

    """Create a non-periodic rectilinear grid of NaCl pair in a harmonic restraining potential.

    Parameters
    ----------
    nx : int, optional, default=3
        number of particles in the x direction
    ny : int, optional, default=3
        number of particles in the y direction
    nz : int, optional, default=3
        number of particles in the z direction

    Attributes
    ----------
    MASS_Na : openmm.unit.Quantity
        Mass of a Na atom.
    MASS_Cl : openmm.unit.Quantity
        Mass of a Cl atom.
    Q_Na : openmm.unit.Quantity
        Charge of a Na ion.
    Q_Cl : openmm.unit.Quantity
        Charge of a Cl ion.
    SIGMA_Na : openmm.unit.Quantity
        Lennard-Jones sigma parameter for Na.
    SIGMA_Cl : openmm.unit.Quantity
        Lennard-Jones sigma parameter for Cl.
    EPSILON_Na : openmm.unit.Quantity
        Lennard-Jones epsilon parameter for Na.
    EPSILON_Cl : openmm.unit.Quantity
        Lennard-Jones epsilon parameter for Cl.
    CUTOFF : openmm.unit.Quantity or None
        Class-level default cutoff distance. If None, no cutoff is used.
    SWITCH_WIDTH : openmm.unit.Quantity or None
        Class-level default switching width.
    SCALE_STEP_SIZE_X : float
        Step size scaling in the x dimension (default=1.0).
    SCALE_STEP_SIZE_Y : float
        Step size scaling in the y dimension (default=1.0).
    SCALE_STEP_SIZE_Z : float
        Step size scaling in the z dimension (default=1.0).

    """

    MASS_Na = 22.99 * unit.amu
    MASS_Cl = 35.45 * unit.amu
    Q_Na = 1.0 * unit.elementary_charge
    Q_Cl = -1.0 * unit.elementary_charge
    SIGMA_Na = 2.0 * unit.angstrom
    SIGMA_Cl = 4.0 * unit.angstrom
    EPSILON_Na = 0.1 * unit.kilojoule_per_mole
    EPSILON_Cl = 0.2 * unit.kilojoule_per_mole
    K = 1.0 * unit.kilojoules_per_mole / unit.nanometer**2

    CUTOFF = None
    SWITCH_WIDTH = None

    SCALE_STEP_SIZE_X = 1.0
    SCALE_STEP_SIZE_Y = 1.0
    SCALE_STEP_SIZE_Z = 1.0


    def __init__(self, nx=3, ny=3, nz=3, **kwargs):
        super().__init__(**kwargs)
        
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.natoms = nx * ny * nz

        self.construct_system()

    def construct_system(self):
        system = openmm.System()

        nb = openmm.NonbondedForce()

        if self.CUTOFF is None:
            nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        else:
            nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
            nb.setCutoffDistance(self.CUTOFF)
            nb.setUseDispersionCorrection(False)
            nb.setUseSwitchingFunction(False)
            if self.SWITCH_WIDTH is not None:
                nb.setUseSwitchingFunction(True)
                nb.setSwitchingDistance(self.CUTOFF - self.SWITCH_WIDTH)

        positions = unit.Quantity(np.zeros([self.natoms, 3], np.float32), unit.angstrom)

        atom_index = 0
        for ii in range(self.nx):
            for jj in range(self.ny):
                for kk in range(self.nz):
                    if (atom_index % 2) == 0:  # Alternating Na and Cl
                        mass = self.MASS_Na
                        q = self.Q_Na
                        sigma = self.SIGMA_Na
                        epsilon = self.EPSILON_Na
                        element = omma.Element.getBySymbol('Na')
                        atom_name = 'Na'
                    else:
                        mass = self.MASS_Cl
                        q = self.Q_Cl
                        sigma = self.SIGMA_Cl
                        epsilon = self.EPSILON_Cl
                        element = omma.Element.getBySymbol('Cl')
                        atom_name = 'Cl'
                    
                    system.addParticle(mass)
                    nb.addParticle(q, sigma, epsilon)
                    x = sigma * self.SCALE_STEP_SIZE_X * (ii - self.nx / 2.0)
                    y = sigma * self.SCALE_STEP_SIZE_Y * (jj - self.ny / 2.0)
                    z = sigma * self.SCALE_STEP_SIZE_Z * (kk - self.nz / 2.0)

                    positions[atom_index, 0] = x
                    positions[atom_index, 1] = y
                    positions[atom_index, 2] = z
                    atom_index += 1

        system.addForce(nb)

        topology = omma.Topology()
        chain = topology.addChain()
        for _ in range(system.getNumParticles()):
            residue = topology.addResidue(atom_name, chain)
            topology.addAtom(atom_name, element, residue)
        self.topology = topology

        # Add a restraining potential centered at the origin.
        system.addForce(self.construct_restraining_potential(particle_indices=range(self.natoms), K=self.K))

        self.system, self.positions = system, positions
