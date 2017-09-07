import numpy as np

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from wepy.walker import Walker
from wepy.runner import Runner

# for the sim manager, dependencies will change
import sys
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import h5py

from wepy.sim_manager import Manager

# default inputs
from wepy.resampling.resampler import NoResampler
from wepy.runner import NoRunner
from wepy.boundary_conditions.boundary import NoBC


class OpenMMRunner(Runner):

    def __init__(self, system, topology, platform=None):
        self.system = system
        self.topology = topology
        self.platform_name = platform

    def run_segment(self, walker, segment_length, **kwargs):

        # TODO move this out to the constructor
        # instantiate an integrator
        integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                            1/unit.picosecond,
                                            0.002*unit.picoseconds)

        # if a platform was given we use it to make a Simulation object
        if self.platform_name is not None:
            # get the platform by its name to use
            platform = omm.Platform.getPlatformByName(self.platform_name)
            # set properties from the kwargs if they apply to the platform
            for key in kwargs:
                if key in platform.getPropertyNames():
                    platform.setPropertyDefaultValue(key, str(kwargs[key]))

            # instantiate a simulation object
            simulation = omma.Simulation(self.topology, self.system, integrator, platform)

        # Otherwise just use the default or environmentally defined one
        else:
            simulation = omma.Simulation(self.topology, self.system, integrator)

        # set the state to the context from the walker
        simulation.context.setState(walker.state)

        # Run the simulation segment for the number of time steps
        simulation.step(segment_length)

        # save the state of the system with all possible values
        new_state = simulation.context.getState(getPositions=True,
                                            getVelocities=True,
                                            getParameters=True,
                                            getParameterDerivatives=False,
                                            getForces=False,
                                            getEnergy=False,
                                            enforcePeriodicBox=False
                                            )

        # create a new walker for this
        new_walker = OpenMMWalker(new_state, walker.weight)

        return new_walker



class OpenMMWalker(Walker):

    def __init__(self, state, weight):
        super().__init__(state, weight)

        self._keys = ['positions', 'velocities', 'forces', 'kinetic_energy',
                      'potential_energy', 'time', 'box_vectors', 'box_volume',
                      'parameters', 'parameter_derivatives']

    def dict(self):
        """return a dict of the values."""
        return {'positions' : self.positions_values(),
                'velocities' : self.velocities_values(),
                'forces' : self.forces_values(),
                'kinetic_energy' : self.kinetic_energy_value(),
                'potential_energy' : self.potential_energy_value(),
                'time' : self.time_value(),
                'box_vectors' : self.box_vectors_values(),
                'box_volume' : self.box_volume_value(),
                'parameters' : self.parameters_values(),
                'parameter_derivatives' : self.parameter_derivatives_values()
                    }

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        if key == 'positions':
            return self.positions_values()
        elif key == 'velocities':
            return self.velocities_values
        elif key == 'forces':
            return self.forces_values()
        elif key == 'kinetic_energy':
            return self.kinetic_energy_value()
        elif key == 'potential_energy':
            return self.potential_energy_value()
        elif key == 'time':
            return self.time_value()
        elif key == 'box_vectors':
            return self.box_vectors_values()
        elif key == 'box_volume':
            return self.box_volume_value()
        elif key == 'parameters':
            return self.parameters_values()
        elif key == 'parameter_derivatives':
            return self.parameter_derivatives_values()
        else:
            raise KeyError('{} not an OpenMMWalker attribute')

    @property
    def positions(self):
        try:
            return self.state.getPositions()
        except TypeError:
            return None

    @property
    def positions_unit(self):
        return self.positions.unit

    def positions_values(self):
        return np.array(self.positions.value_in_unit(self.positions_unit))

    @property
    def velocities(self):
        try:
            return self.state.getVelocities()
        except TypeError:
            return None

    @property
    def velocities_unit(self):
        return self.velocities.unit

    def velocities_values(self):
        velocities = self.velocities
        if velocities is None:
            return None
        else:
            return np.array(self.velocities.value_in_unit(self.velocities_unit))

    @property
    def forces(self):
        try:
            return self.state.getForces()
        except TypeError:
            return None

    @property
    def forces_unit(self):
        return self.forces.unit

    def forces_values(self):
        forces = self.forces
        if forces is None:
            return None
        else:
            return np.array(self.forces.value_in_unit(self.forces_unit))

    @property
    def kinetic_energy(self):
        try:
            return self.state.getKineticEnergy()
        except TypeError:
            return None

    @property
    def kinetic_energy_unit(self):
        return self.kinetic_energy.unit

    def kinetic_energy_value(self):
        kinetic_energy = self.kinetic_energy
        if kinetic_energy is None:
            return None
        else:
            return np.array(self.kinetic_energy.value_in_unit(self.kinetic_energy_unit))

    @property
    def potential_energy(self):
        try:
            return self.state.getPotentialEnergy()
        except TypeError:
            return None

    @property
    def potential_energy_unit(self):
        return self.potential_energy.unit

    def potential_energy_value(self):
        potential_energy = self.potential_energy
        if potential_energy is None:
            return None
        else:
            return np.array(self.potential_energy.value_in_unit(self.potential_energy_unit))

    @property
    def time(self):
        try:
            return self.state.getTime()
        except TypeError:
            return None

    @property
    def time_unit(self):
        return self.time.unit

    def time_value(self):
        time = self.time
        if time is None:
            return None
        else:
            return np.array(self.time.value_in_unit(self.time_unit))

    @property
    def box_vectors(self):
        try:
            return self.state.getPeriodicBoxVectors()
        except TypeError:
            return None

    @property
    def box_vectors_unit(self):
        return self.box_vectors.unit

    def box_vectors_values(self):
        box_vectors = self.box_vectors
        if box_vectors is None:
            return None
        else:
            return np.array(self.box_vectors.value_in_unit(self.box_vectors_unit))

    @property
    def box_volume(self):
        try:
            return self.state.getPeriodicBoxVolume()
        except TypeError:
            return None

    @property
    def box_volume_unit(self):
        return self.box_volume.unit

    def box_volume_value(self):
        box_volume = self.box_volume
        if box_volume is None:
            return None
        else:
            return np.array(self.box_volume.value_in_unit(self.box_volume_unit))

    @property
    def parameters(self):
        try:
            return self.state.getParameters()
        except TypeError:
            return None

    # TODO test this, this is jsut a guess because I don't use parameters
    @property
    def parameters_unit(self):
        param_units = {key : val.unit for key, val in self.parameters.items()}
        return param_units

    # TODO test this, this is jsut a guess because I don't use parameters
    def parameters_values(self):
        if self.parameters is None:
            return None

        param_arrs = {key : np.array(val.value_in_unit(val.unit)) for key, val
                          in self.parameters.items()}

        # return None if there is nothing in this
        if len(param_arrs) == 0:
            return None
        else:
            return param_arrs

    @property
    def parameter_derivatives(self):
        try:
            return self.state.getEnergyParameterDerivatives()
        except TypeError:
            return None

    # TODO test this, this is jsut a guess because I don't use parameters
    @property
    def parameter_derivatives_unit(self):
        param_units = {key : val.unit for key, val in self.parameter_derivatives.items()}
        return param_units

    # TODO test this, this is jsut a guess because I don't use parameter_derivatives
    def parameter_derivatives_values(self):

        if self.parameter_derivatives is None:
            return None

        param_arrs = {key : np.array(val.value_in_unit(val.unit)) for key, val
                          in self.parameter_derivatives.items()}

        # return None if there is nothing in this
        if len(param_arrs) == 0:
            return None
        else:
            return param_arrs

    def to_mdtraj(self):
        """ Returns an mdtraj.Trajectory object from this walker's state."""
        raise NotImplementedError
        import mdtraj as mdj
        return mdj.Trajectory(self.positions_values,
                              time=self.time_value, unitcell_vectors=self.box_vectors)

# class OpenMMManager(Manager):
#     def __init__(self, init_walkers, num_workers,
#                  runner = NoRunner(),
#                  resampler = NoResampler(),
#                  ubc = NoBC(),
#                  work_mapper = map):
#         super().__init__(init_walkers, num_workers,
#                         runner,
#                         resampler,
#                         work_mapper)
#         self.ubc = ubc

#     def run_simulation(self, n_cycles, segment_lengths, debug_prints=False):
#         """Run a simulation for a given number of cycles with specified
#         lengths of MD segments in between.

#         Can either return results in memory or write to a file.
#         """


#         if debug_prints:
#             sys.stdout.write("Starting simulation\n")
#         walkers = self.init_walkers

#         resampling_handler = pd.HDFStore(os.getcwd()+'/resampling_records.h5',mode='w')
#         walker_handler = h5py.File(os.getcwd()+'/walkers_records.h5',mode='w')
#         dist_handler = h5py.File(os.getcwd()+'/dist_records.h5',mode='w')
#         #save initial state
#         self.save_walker_records(walker_handler,-1, walkers)
#         for cycle_idx in range(n_cycles):
#             if debug_prints:
#                 sys.stdout.write("Begin cycle {}\n".format(cycle_idx))


#             # run the segment

#             walkers = self.run_segment(walkers, segment_lengths[cycle_idx],
#                                            debug_prints=debug_prints)


#             # calls wexplore2 ubinding boundary conditions
#             if debug_prints:
#                 sys.stdout.write("Start  boundary Conditions")


#             resampled_walkers, warped_walkers_records, ubc_data= self.ubc.warp_walkers(walkers)
#             # ubc_data is min_distances
#             # record changes in state of the walkers
#             if debug_prints:
#                 sys.stdout.write("End  BoundaryConditions")
#                 print ('warped_walkers=',warped_walkers_records)

#             if debug_prints:
#                 sys.stdout.write("Start Resampling")


#             # resample based walkers
#             resampled_walkers, cycle_resampling_records, resample_data = \
#                             self.resampler.resample(resampled_walkers, debug_prints=debug_prints)

#             # resample_data includes distance_matrix and last_spread
#             self.save_dist_records(dist_handler,cycle_idx, resample_data, ubc_data)

#             # save resampling records in a hdf5 file
#             self.save_resampling_records(resampling_handler,
#                                          cycle_idx,
#                                          cycle_resampling_records,
#                                          warped_walkers_records)
#             if debug_prints:
#                 sys.stdout.write("End  Resampling")

#             # prepare resampled walkers for running new state changes
#             # save walkers positions in a hdf5 file
#             self.save_walker_records(walker_handler, cycle_idx, resampled_walkers)
#             walkers = resampled_walkers.copy()

#         # saving last velocities
#         for walker_idx, walker  in enumerate(resampled_walkers):
#             walker_handler.create_dataset(
#                 'cycle_{:0>5}/walker_{:0>5}/velocities'.format(cycle_idx,walker_idx),
#                 data = self.mdtraj_positions(walker.velocities))

#         resampling_handler.close()
#         walker_handler.close()
#         dist_handler.close()
#         if debug_prints:
#             sys.stdout.write("End cycle {}\n".format(cycle_idx))


#     def mdtraj_positions(self, openmm_positions):

#         n_atoms = len (openmm_positions)

#         xyz = np.zeros(( n_atoms, 3))



#         for i in range(n_atoms):
#             xyz[i,:] = ([openmm_positions[i]._value[0], openmm_positions[i]._value[1],
#                                                     openmm_positions[i]._value[2]])

#         return xyz

#     def save_resampling_records(self, hdf5_handler,
#                                 cycle_idx,
#                                 cycle_resampling_records,
#                                 warped_walkers_records):

#         # save resampling records in table format in a hdf5 file
#         DFResamplingRecord = namedtuple("DFResamplingRecord", ['cycle_idx', 'step_idx',
#                                                                'walker_idx', 'decision',
#                                                                'instruction', 'warped_walker'])
#         df_recs = []
#         warped_walkers_idxs =[]
#         for record in warped_walkers_records:
#             warped_walkers_idxs.append(record[0])

#         for step_idx, step in enumerate(cycle_resampling_records):
#             for walker_idx, rec in enumerate(step):

#                 if  walker_idx  in warped_walkers_idxs:
#                     decision = True
#                 else:
#                     decision = False
#                 df_rec = DFResamplingRecord(cycle_idx=cycle_idx,
#                                             step_idx=step_idx,
#                                             walker_idx=walker_idx,
#                                             decision=rec.decision.name,
#                                             instruction = rec.value,
#                                             warped_walker = decision)
#                 df_recs.append(df_rec)

#         resampling_df = pd.DataFrame(df_recs)

#         hdf5_handler.put('cycle_{:0>5}'.format(cycle_idx), resampling_df, data_columns= True)
#         hdf5_handler.flush(fsync=True)




#     def save_walker_records(self, walker_handler, cycle_idx, resampled_walkers):

#         walker_handler.create_dataset('cycle_{:0>5}/time'.format(cycle_idx),
#                                       data=resampled_walkers[0].time._value)
#         for walker_idx, walker in enumerate(resampled_walkers):
#             walker_handler.create_dataset(
#                 'cycle_{:0>5}/walker_{:0>5}/positions'.format(cycle_idx,walker_idx),
#                                           data = self.mdtraj_positions(walker.positions))
#             box_vector = np.array(((walker.box_vectors._value[0],
#                                     walker.box_vectors._value[1],
#                                     walker.box_vectors._value[2])))
#             walker_handler.create_dataset(
#                 'cycle_{:0>5}/walker_{:0>5}/box_vectors'.format(cycle_idx,walker_idx),
#                 data=box_vector)
#             walker_handler.create_dataset(
#                 'cycle_{:0>5}/walker_{:0>5}/weight'.format(cycle_idx,walker_idx),
#                 data=walker.weight)

#             walker_handler.flush()


#     def read_resampling_data(self,):

#         hdf = pd.HDFStore(os.getcwd()+'/resampling_records.h5',
#                             mode ='r')
#         keys = list (hdf.keys())
#         for key in keys:
#             df = hdf.get(key)
#             print (df)
#         hdf.close()

#     def save_dist_records(self, dist_handler,cycle_idx, resample_data, ubc_data):
#         dist_handler.create_dataset(
#             'cycle_{:0>5}/dist_matrix'.format(cycle_idx),
#             data=resample_data['distance_matrix'])
#         dist_handler.create_dataset(
#             'cycle_{:0>5}/spread'.format(cycle_idx),
#             data=resample_data['spread'])
#         dist_handler.create_dataset(
#             'cycle_{:0>5}/min_distances'.format(cycle_idx),
#             data=ubc_data['min_distances'])

#         dist_handler.flush()
