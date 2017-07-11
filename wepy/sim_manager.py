from scoop import futures

from wepy.decision import Decision, DecisionModel
from wepy.decision import NoCloneMerge
from wepy.resampling import StubResampler
from wepy.runner import NoRunner

class Manager(object):

    def __init__(self, init_walkers,
                 num_walkers, num_workers,
                 runner=NoRunner,
                 decision_model=NoCloneMerge,
                 resampler=StubResampler,
                 work_mapper=futures.map):

        # the runner is the object that runs dynamics
        self.runner = runner
        # the number of walkers to use
        self.num_walkers = num_walkers
        # TODO make this useful with SCOOP
        # the number of CPUs/GPUs to use
        self.num_workers = num_workers
        # the decision making class
        assert issubclass(decision_model, DecisionModel), \
            "decision_model is not a subclass of DecisionModel"
        self.decision_model = decision_model

        # the resampler
        self.resampler = resampler

        # the function for running work on the workers
        self.map = work_mapper

    def run_segment(self, walkers, segment_length):
        """Run a time segment for all walkers using the available workers. """

        new_walkers = list(self.map(self.runner.run_segment,
                                    walkers,
                                    (segment_length for i in range(self.num_workers))))

        return results

    def run_simulation(self, n_cycles, segment_lengths, output="memory"):
        """Run a simulation for a given number of cycles with specified
        lengths of MD segments in between.

        Can either return results in memory or write to a file.
        """

        resampling_records = []
        walkers = self.init_walkers
        walker_records = [walkers]
        for cycle_idx in range(n_cycles):
            new_walkers = self.run_segment(walkers, segment_lengths[cycle_idx])
            decisions = self.decision_model(new_walkers)
            resampled_walkers, cycle_resampling_records = self.resampler.resample(new_walkers,
                                                                      decisions)
            resampling_records.append(cycle_resampling_records)
            walker_records.append(new_walkers)

        return walker_records, resampling_records

    def write_results(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError

