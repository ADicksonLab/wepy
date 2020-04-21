"""This module sets up a random walk system then runs a simulation using
the given resampler. This profiles the performance of the resampler
using the data produced in the simulation.

The simulation data is saved using the WepyHDF5 reporter. To create a
WepyHDF5 reporter we create a dummy topology for the random walk
system. This dummy system consists of one atom with a position vector
in an N-dimensional space.

The resmapler quality is measured using the following values:

    - P(x): The average predicted probability at position x.

    - Accuracy: This is calculated using the target probability and predicted
    probabilities of walkers at position x.

   - Range: The maximum range that is observed by the given resampler is calculated
     by determining the largest x values visited along each dimension,
     then averaging them.

You can find detailed information about random walk parameters in the papers:

"WExplore: Hierarchical Exploration of High-Dimensional Spaces
Using the Weighted Ensemble Algorithm" and
"REVO: Resampling of Ensembles by Variation Optimization".

"""

import os
import sys
import json

import numpy as np
import pandas as pd
import h5py

import mdtraj as mdj

from wepy.resampling.resamplers.resampler import NoResampler
from wepy.work_mapper.mapper import Mapper
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.sim_manager import Manager
from wepy.walker import Walker, WalkerState
from wepy.runners.randomwalk import RandomWalkRunner, UNIT_NAMES
from wepy.hdf5 import WepyHDF5
from wepy.util.mdtraj import mdtraj_to_json_topology


PRECISION = 3

SAVE_FIELDS = ('positions',)
UNITS=UNIT_NAMES


np.set_printoptions(precision=PRECISION)



class RandomwalkProfiler(object):
    """A class to implement RandomWalkProfilier."""


    RANDOM_WALK_TEMPLATE=\
"""* Random walk simulation:
-Number of runs: {n_runs}
-Number of cycles: {n_cycles}
-Number of walkers:{n_walkers}
-Move-forward probability:{prob}
-Dimension:{dimension}
"""
    RUN_RESULTS_TEMPLATE=\
"""* Run {run_idx} results:
-Maximum range: {max_range}
-Maximum range of dimensions:
{max_dim_range}
-Accuracy: {accuracy}
-Average Predicted probability:
{predicted_probabilty}

"""

    def __init__(self, resampler=None, dimension=None, probility=0.25,
                 hdf5_filename='rw_results.wepy.h5',
                 reporter_filename='randomwalk.org'):
        """Constructor for RandomwalkProfiler.

        Parameters
        ----------
        resampler:
            The dimension of the random walk space.
             (Default = 2)

        probabilty: float
            "Probability" is defined here as the forward-move
             probability only. The backward-move probability is
             1-probability.
             (Default = 0.25)

        """

        assert resampler is not None,  "Resampler object must be given."
        self.resampler = resampler

        assert dimension is not None,  "The dimension of random walk  must be given."
        self.dimension = dimension

        self.probability = probility

        self.hdf5_filename = hdf5_filename

        self.reporter_filename = reporter_filename

    def generate_topology(self):
        """Creates a one-atom, dummy trajectory and topology for
        the randomwalk system using the mdtraj package.  Then creates a
        JSON format for the topology. This JSON string is used in making
        the WepyHDF5 reporter.

        Returns
        -------
        topology: str
            JSON string representing the topology of system being simulated.

        """
        n_atoms = 1
        data = []
        for i in range(n_atoms):
            data.append(dict(serial=i, name="H", element="H",
                             resSeq=i + 1, resName="UNK", chainID=0))

        data = pd.DataFrame(data)

        xyz = np.zeros((1, 1, 3))
        unitcell_lengths = 0.943 * np.ones((1, 3))
        unitcell_angles = 90 * np.ones((1, 3))

        top = mdj.Topology.from_dataframe(data, bonds=np.zeros((0, 2), dtype='int'))

        json_top_str = mdtraj_to_json_topology(top)

        return json_top_str


    def run(self, num_runs=1, num_cycles=200, num_walkers=100):
        """Runs a random walk simulation and profiles the resampler
        performance.

        Parameters
        ----------
        num_runs: int
            The number independet simulations.

        num_cycles: int
            The number of cycles that will be run in the simulation.

        num_walkers: int
            The number of walkers.

        """
        # set the random walk simulation repreter string
        randomwalk_string = self.RANDOM_WALK_TEMPLATE.format(
            n_runs=num_runs,
            n_cycles=num_cycles,
            n_walkers=num_walkers,
            prob=self.probability,
            dimension=self.dimension
            )

        # calls the runner
        self._run(num_runs, num_cycles, num_walkers)

        # calls the profiler
        self.analyse(randomwalk_string)


    def _run(self, num_runs, num_cycles, num_walkers):
        """Runs a random walk simulation.

        Parameters
        ----------
        num_runs: int
            The number independet simulations.

        num_cycles: int

            The number of cycles that will be run in the simulation.

        num_walkers: int
            The number of walkers.

        """

        print("Random walk simulation with: ")
        print("Dimension = {}".format(self.dimension))
        print("Probability = {}".format(self.probability))
        print("Number of Walkers = {}".format(num_walkers))
        print("Number of Cycles ={}".format(num_cycles))

        # set up initial state for walkers
        positions = np.zeros((1, self.dimension))

        init_state = WalkerState(positions=positions, time=0.0)


        # create list of init_walkers
        initial_weight = 1/num_walkers
        init_walkers = []

        init_walkers = [Walker(init_state, initial_weight)
                        for i in range(num_walkers)]

        # set up raunner for system
        runner = RandomWalkRunner(probability=self.probability)

        units = dict(UNIT_NAMES)
        # instantiate a revo unbindingboudaryconditiobs
        segment_length = 10

        # set up the reporter
        randomwalk_system_top_json = self.generate_topology()

        hdf5_reporter = WepyHDF5Reporter(file_path=self.hdf5_filename,
                                         mode='w',
                                         save_fields=SAVE_FIELDS,
                                         topology=randomwalk_system_top_json,
                                         resampler=self.resampler,
                                         units=dict(UNITS),
                                         n_dims=self.dimension)
        # running the simulation
        sim_manager = Manager(init_walkers,
                              runner=runner,
                              resampler=self.resampler,
                              work_mapper=Mapper(),
                              reporters=[hdf5_reporter])


        # run a simulation with the manager for n_steps cycles of length 1000 each
        steps = [segment_length for i in range(num_cycles)]
        ### RUN the simulation
        for run_idx in range(num_runs):
            print("Starting run: {}".format(run_idx))
            sim_manager.run_simulation(num_cycles, steps)
            print("Finished run: {}".format(run_idx))


        print("Finished Simulation")



    def kronecker_delta(self, x):
        """Implements the the Kronecker delta function.

        Parameters
        ----------
        x: int
            Input value of the function. Here, this is the random walk position.


        Returns
        -------
        y: int
            The output of the the Kronecker delta function.

        """

        if x == 0:
            return 1
        else:
            return 0

    # Measure accuracy
    def accuracy(self, x, Px):
        """Calculate the accuracy at position x.

        Parameters
        ----------
        x: int
            The position.

        Returns
        -------

        accuracy: float
            The value that specifies how accurate the resampler is at point x.
            The highest accuracy is achived when P(X) = Pt(x).


        """

        if Px == 0:
            return 0
        elif np.log(Px) > 2* np.log(self.Pt(x)):
            return 1 + np.abs( np.log(self.Pt(x)) - np.log(Px)) / np.log(self.Pt(x))
        else:
            return 0


    def Pt(self, x):
        """Calculate the target probability at position x.

        Parameters
        ----------

        x: int
            The position.

        Returns
        -------

        accuracy : float
            The value of the target probability when the
            forward-move probability p.

        """

        # p/q where q is (1-p)
        ratio = self.probability/(1-self.probability)

        if self.probability > (1-self.probability):
            return 1

        else:

            return (1 - ratio) * np.power(ratio, x)


    def get_max_range(self, wepy_h5, run_idx=0):
        """Finds the furthest range (position) visited by the walkers in the
        simulation.

        Parameters
        ----------
        wepy_h5: Hdf5 file object
            The hdf5 file that the simulation data is stored in.

        run_idx: int
            The index of the run.

        Returns
        -------

        max_range: int
            The maximum range that is visited by all walkers in
            all dimensions.

        max_dim_range: arraylike of shape (1, dimension)
            The miximum value of range in each dimension.

        """

        max_ranges = []
        num_cycles = wepy_h5.num_run_cycles(run_idx)
        num_walkers = wepy_h5.num_run_trajs(run_idx)

        for walker_idx in range(num_cycles):

            # makes the trace for the current walker
            walker_trace = [(walker_idx, cycle_idx) for cycle_idx in range(num_cycles)]

            # gets the data of given fields for the walker
            traj_data = wepy_h5.get_run_trace_fields(run_idx, walker_trace,
                                                     ['positions'])
            # gets the walker posiotions
            positions = traj_data['positions']

            max_ranges.append(np.max(positions, axis=0)[0])


        return np.max(np.array(max_ranges)), np.max(np.array(max_ranges), axis=0)



    def get_predicted_probability(self, wepy_h5, run_idx, max_range):
        """This projects all dimensions in the simulation on a
           1-dimensional space. It then calculates the predicted
           probability for all positions in in this 1D space in the
           range 0 to max_x.


        Parameters
        ----------
        walker: object implementing the Walker interface
            The individual walker for which dynamics will be propagated.

        segment_length : int
            The numerical value that specifies how much dynamical steps
            are to be run.

        Returns
        -------
        predicted_probabilty: list of floats
            The average predicted probability distribution.

        """

        predited_probibilty = np.zeros((max_range))

        num_cycles = wepy_h5.num_run_cycles(run_idx)
        num_walkers = wepy_h5.num_run_trajs(run_idx)

        for walker_idx in range(num_cycles):
            # makes the trace for the current walker
            walker_trace = [(walker_idx, cycle_idx) for cycle_idx in range(num_cycles)]
            # gets the data of given fields for the current walker
            traj_data = wepy_h5.get_run_trace_fields(run_idx, walker_trace, ['positions', 'weights'])

            weight = traj_data['weights']
            positions = traj_data['positions']

            # calculates for all x in range 0 to max_range
            for x in range(max_range):
                for cycle_idx in range(num_cycles):
                    for dim_idx in range(self.dimension):
                        predited_probibilty[x] += weight[walker_idx] * \
                        self.kronecker_delta(x - positions[walker_idx, 0, dim_idx])

        return 1/(num_cycles * self.dimension) * predited_probibilty

    def get_accuracy(self, predicted_probabilty):
        """Calculates the accuracy of the resampler for a random walk simulation
        run.

        Parameters
        ----------
        predicted_probabilty: list of floats
            Average predicted probability distribution.

        Returns
        -------
        accuracy: float
            The accuracy of the resampler determined using
            the accuracy equation.
        """


        accuracy_value = 0
        for x, px in enumerate(predicted_probabilty):
            accuracy_value += self.accuracy(x, px)

        return accuracy_value

    def analyse(self, randomwalk_string):
        """Calculates all quality metrics for the random walk simulation
        including pridicted probabilities, accuracy, and maximum
        average range.

        Parameters
        ----------
        walker: object implementing the Walker interface
            The individual walker for which dynamics will be propagated.

        Returns
        -------
        results: dict of str/arraylike

        """

        wepy_h5 = WepyHDF5(self.hdf5_filename, mode='r')
        wepy_h5.open()


        # get the number of euns
        num_runs = wepy_h5.num_runs


        results = []
        for run_idx in range(num_runs):

            max_range, max_dim_ranges = self.get_max_range(wepy_h5, run_idx)

            predicted_probabilty = self.get_predicted_probability(wepy_h5, run_idx,
                                           int(max_range))

            accuracy_value = self.get_accuracy(predicted_probabilty)

            run_results = {'max_range': max_range,
                       'max_dim_range': max_dim_ranges,
                       'predicted_probabilty': predicted_probabilty,
                       'accuracy': accuracy_value}

            results.append(run_results)

        self.write_report(randomwalk_string, results)

        return results


    def run_string(self, run_idx, run_results):
        """Creates a formated string for writing results of a randomwalk run.

        Parameters
        ----------
        run_idx: int
            The index of the run.

        Returns
        -------
        results: str
            The formated results.

        """

        run_result_string = self.RUN_RESULTS_TEMPLATE.format(
            run_idx=run_idx,
            max_range=run_results['max_range'],
            max_dim_range=np.array2string(run_results['max_dim_range'],
                                        separator=','),
            accuracy=round(run_results['accuracy'], PRECISION),

            predicted_probabilty=np.array2string(run_results['predicted_probabilty'],
                                                 separator=',')
            )


        return run_result_string



    def write_report(self, randomwalk_string, results):
        """Write the dashboard to the file."""

        mode = 'w'

        report_string = randomwalk_string

        for run_idx, run_results in enumerate(results):
            report_string += self.run_string(run_idx, run_results)

        with open(self.reporter_filename, mode=mode) as reporter_file:
            reporter_file.write(report_string)
