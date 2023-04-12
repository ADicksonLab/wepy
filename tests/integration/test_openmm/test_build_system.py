"""
Tests the ability to use openmm functions to build a system suitable for wepy simulation.
"""

import openmm as omm
import openmm.app as omma
import pickle as pkl

import os.path as osp

def test_build_system(write=False):

    base_folder = 'src/pytest_wepy/test_data'

    psf = omma.CharmmPsfFile(osp.join(base_folder,'step3_input.psf'))
    crd = omma.CharmmCrdFile(osp.join(base_folder,'step3_input.crd'))

    box_len = 2.0 # nm
    temp = 303.25*omm.unit.kelvin
    pressure = 1.0*omm.unit.atmosphere
    fric = 1.0/omm.unit.picosecond
    dt = 0.002*omm.unit.picosecond

    params = omma.CharmmParameterSet(osp.join(base_folder,'toppar_water_ions.str'))
    psf.setBox(box_len,box_len,box_len)

    system = psf.createSystem(params,nonbondedMethod=omma.CutoffPeriodic)

    barostat = omm.MonteCarloBarostat(pressure, temp)
    system.addForce(barostat)

    integrator = omm.LangevinIntegrator(temp, fric, dt)
    
    platform = omm.Platform.getPlatformByName('Reference')

    # Build simulation context
    simulation = omma.Simulation(psf.topology, system, integrator, platform)
    simulation.context.setPositions(crd.positions)

    simulation.step(10)
    omm_state = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, getIntegratorParameters=True)

    if write:
        pkl.dump(omm_state,open(osp.join(base_folder,'omm_state.pkl'),'wb'))

if __name__ == '__main__':
    test_build_system(write=True)
