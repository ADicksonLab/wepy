"""This script was used to produce the inputs that would be used in
the examples and shows an example of how you might go from a crystal
structure of a protein ligand complex to a set of inputs for
wepy. This is by no means the only way to do this. Furthermore, a few
of the things written out will be deprecated in the future as
necessary, or are only used for convenience here.

"""


if __name__ == "__main__":

    import os
    import os.path as osp
    import pickle
    import sys

    import numpy as np
    import h5py

    import simtk.openmm.app as omma
    import simtk.openmm as omm
    import simtk.unit as unit

    import mdtraj as mdj

    from wepy.runners.openmm import OpenMMState
    from wepy.util.mdtraj import mdtraj_to_json_topology

    # input paths
    inputs_dir = osp.realpath('./inputs')
    charmm_psf_filename = 'sEH_TPPU_system.psf'
    starting_coords_pdb = 'sEH_TPPU_system.pdb'
    charmm_param_files = ['all36_cgenff.rtf',
                          'all36_cgenff.prm',
                          'all36_prot.rtf',
                          'all36_prot.prm',
                          'tppu.str',
                          'toppar_water_ions.str']

    # output paths
    json_top_filename = "sEH_TPPU_system.top.json"
    omm_state_filename = "initial_openmm_state.pkl"

    # normalize the paths
    charmm_psf_path = osp.join(inputs_dir, charmm_psf_filename)
    pdb_path = osp.join(inputs_dir, starting_coords_pdb)
    charmm_param_paths = [osp.join(inputs_dir, filename) for filename
                          in charmm_param_files]

    json_top_path = osp.join(inputs_dir, json_top_filename)
    omm_state_path = osp.join(inputs_dir, omm_state_filename)

    # load the charmm file for the topology
    psf = omma.CharmmPsfFile(charmm_psf_path)

    # load the coordinates
    pdb = mdj.load_pdb(pdb_path)

    # convert the mdtraj topology to a json one
    top_str = mdtraj_to_json_topology(pdb.topology)

    # write the JSON topology out
    with open(json_top_path, mode='w') as json_wf:
        json_wf.write(top_str)


    # to use charmm forcefields get your parameters
    params = omma.CharmmParameterSet(*charmm_param_paths)

    # set the box size lengths and angles
    lengths = [8.2435*unit.nanometer for i in range(3)]
    angles = [90*unit.degree for i in range(3)]
    psf.setBox(*lengths, *angles)

    # create a system using the topology method giving it a topology and
    # the method for calculation
    system = psf.createSystem(params,
                              nonbondedMethod=omma.CutoffPeriodic,
                              nonbondedCutoff=1.0 * unit.nanometer,
                              constraints=omma.HBonds)

    # we want to have constant pressure
    barostat = omm.MonteCarloBarostat(1.0*unit.atmosphere, 300.0*unit.kelvin, 50)

    topology = psf.topology

    print("\nminimizing\n")
    # set up for a short simulation to minimize and prepare
    # instantiate an integrator
    temperature = 300*unit.kelvin
    integrator = omm.LangevinIntegrator(temperature,
                                        1/unit.picosecond,
                                        0.002*unit.picoseconds)
    platform = omm.Platform.getPlatformByName('OpenCL')

    # instantiate a simulation object
    simulation = omma.Simulation(psf.topology, system, integrator, platform)
    # initialize the positions
    simulation.context.setPositions(pdb.openmm_positions(frame=0))
    # minimize the energy
    simulation.minimizeEnergy()

    # run the simulation for a number of initial time steps
    simulation.step(1000)
    print("done minimizing\n")

    # get the initial state from the context
    minimized_state = simulation.context.getState(getPositions=True,
                                                  getVelocities=True,
                                                  getParameters=True,
                                                  getForces=True,
                                                  getEnergy=True,
                                                  getParameterDerivatives=True)

    # pickle it for use in seeding simulations
    with open(omm_state_path, mode='wb') as wf:
        pickle.dump(minimized_state, wf)

    print('finished initialization')
