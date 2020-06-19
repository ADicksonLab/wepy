import logging

from pympler.asizeof import asizeof
import prometheus_client as prom

class SimMonitor:
    """A simulation monitor using a prometheus http server"""

    DEFAULT_PORT = 9001

    def __init__(self,
                 tag="",
                 port=None,
                 reporter_order=(),
    ):

        self.port = port
        self.tag = tag

        self.reporter_order = reporter_order


    def _init_metrics(self):

        logging.info(f"SimMonitor ({self}): Initializing monitoring metrics")

        ## progress tracking
        self.cycle_counter = prom.Counter('wepy_cycle_idx', "",
                                          ['tag']
        )

        # TODO unbinding events

        # TODO largest progresses

        ## object sizes
        self.walker_size_g = prom.Gauge('wepy_walker_single_size_bytes', "",
                                          ['tag']
        )
        self.ensemble_size_g = prom.Gauge('wepy_walker_ensemble_size_bytes', "",
                                          ['tag']
        )

        self.runner_size_g = prom.Gauge('wepy_runner_size_bytes', "",
                                          ['tag']
        )
        self.resampler_size_g = prom.Gauge('wepy_resampler_size_bytes', "",
                                          ['tag']
        )
        self.bc_size_g = prom.Gauge('wepy_bc_size_bytes', "",
                                          ['tag']
        )
        self.mapper_size_g = prom.Gauge('wepy_mapper_size_bytes', "",
                                          ['tag']
        )

        self.sim_manager_size_g = prom.Gauge('wepy_sim_manager_size_bytes', "",
                                          ['tag']
        )

        self.reporter_size_g = prom.Gauge('wepy_reporters_size_bytes',
                                          "",
                                          ['tag', "name"],
        )

        ## timings

        # TODO: convert to Summaries or Histograms for samples

        # components

        self.bc_time_g = prom.Gauge('wepy_bc_cycle_time_seconds', "",
                                          ['tag']
        )
        self.resampling_time_g = prom.Gauge('wepy_resampling_cycle_time_seconds', "",
                                          ['tag']
        )

        # runner splits
        self.runner_precycle_time_g = prom.Gauge('wepy_runner_precycle_time_seconds', "",
                                          ['tag']
        )
        self.runner_postcycle_time_g = prom.Gauge('wepy_runner_postcycle_time_seconds', "",
                                          ['tag']
        )
        self.sim_manager_segment_time_g = prom.Gauge('wepy_sim_manager_segment_time_seconds', "",
                                          ['tag']
        )
        self.sim_manager_segment_overhead_time_g = prom.Gauge(
            'wepy_sim_manager_segment_overhead_time_seconds',
            "",
            ['tag']
        )

        # runner segment splits
        self.runner_segment_gen_sim_time_g = prom.Gauge(
            'wepy_runner_segment_gen_sim_time_seconds',
            "",
            ['tag', 'segment_idx',]
        )

        self.runner_segment_steps_time_g = prom.Gauge(
            'wepy_runner_segment_steps_time_seconds',
            "",
            ['tag', 'segment_idx',]
        )

        self.runner_segment_get_state_time_g = prom.Gauge(
            'wepy_runner_segment_get_state_time_seconds',
            "",
            ['tag', 'segment_idx',]
        )

        self.runner_segment_run_segment_time_g = prom.Gauge(
            'wepy_runner_segment_run_segment_time_seconds',
            "",
            ['tag', 'segment_idx',]
        )

        # work mapper segment times
        self.mapper_seg_times_g = prom.Gauge('wepy_mapper_segment_times_seconds',
                                             "",
                                             ['tag', 'worker_idx', 'seg_idx'])

    def _cleanup_metrics(self):

        logging.info(f"SimMonitor ({self}): cleaning up metrics")

        ## progress tracking
        del self.cycle_counter

        # TODO unbinding events

        # TODO largest progresses

        ## object sizes
        del self.walker_size_g
        del self.ensemble_size_g

        del self.runner_size_g
        del self.resampler_size_g
        del self.bc_size_g
        del self.mapper_size_g

        del self.sim_manager_size_g

        del self.reporter_size_g

        ## timings

        # components

        del self.bc_time_g
        del self.resampling_time_g
        # runner splits
        del self.runner_precycle_time_g
        del self.runner_postcycle_time_g
        del self.sim_manager_segment_time_g
        del self.sim_manager_segment_overhead_time_g
        # runner segment splits
        del self.runner_segment_gen_sim_time_g

        del self.runner_segment_steps_time_g

        del self.runner_segment_get_state_time_g
        del self.runner_segment_run_segment_time_g
        # work mapper segment times
        del self.mapper_seg_times_g

    def init(self,
             port=None,
    ):

        if port is not None:
            port = port

        elif self.port is not None:
            port = self.port

        else:
            port = self.DEFAULT_PORT

        logging.info(f"SimMonitor ({self}): starting prometheus client http server at port {port}")
        prom.start_http_server(port)

        # initialize all the metrics. These need to be done at init
        # time since they can't be pickled
        self._init_metrics()

    def cleanup(self):

        # remove all the metrics
        logging.info(f"SimMonitor ({self}): cleaning up")
        self._cleanup_metrics()



    def cycle_monitor(self, sim_manager, walkers):

        logging.info(f"SimMonitor ({self}): running the cycle monitoring")

        last_report = sim_manager._last_report

        ## Simulation Progress

        # increment the cycle counter
        self.cycle_counter.labels(tag=self.tag).inc()

        ## object sizes

        # get the sizes of the objects
        walker_size = asizeof(walkers[0])
        ensemble_size = asizeof(walkers)

        runner_size = asizeof(sim_manager.runner)
        resampler_size = asizeof(sim_manager.resampler)
        bc_size = asizeof(sim_manager.boundary_conditions)
        mapper_size = asizeof(sim_manager.work_mapper)
        sim_manager_size = asizeof(sim_manager)

        reporter_sizes = {}
        for idx, reporter_name in enumerate(self.reporter_order):
            reporter_sizes[reporter_name] = asizeof(sim_manager.reporters[idx])

        # then update the gauges
        self.walker_size_g.labels(tag=self.tag).set(walker_size)
        self.ensemble_size_g.labels(tag=self.tag).set(ensemble_size)

        self.runner_size_g.labels(tag=self.tag).set(runner_size)
        self.resampler_size_g.labels(tag=self.tag).set(resampler_size)
        self.bc_size_g.labels(tag=self.tag).set(bc_size)
        self.mapper_size_g.labels(tag=self.tag).set(mapper_size)

        self.sim_manager_size_g.labels(tag=self.tag).set(sim_manager_size)

        for reporter_name in self.reporter_order:
            self.reporter_size_g.labels(
                tag=self.tag,
                name=reporter_name,
            ).set(reporter_sizes[reporter_name])


        ## Timings

        self.bc_time_g.labels(tag=self.tag).set(last_report['cycle_bc_time'])
        self.resampling_time_g.labels(tag=self.tag).set(last_report['cycle_resampling_time'])

        # runner splits for each segment
        for seg_idx, segment in enumerate(last_report['runner_splits_time']):

            self.runner_segment_gen_sim_time_g.labels(
                tag=self.tag,
                segment_idx=seg_idx,
            ).set(segment['gen_sim_time'])

            self.runner_segment_steps_time_g.labels(
                tag=self.tag,
                segment_idx=seg_idx,
            ).set(segment['steps_time'])

            self.runner_segment_get_state_time_g.labels(
                tag=self.tag,
                segment_idx=seg_idx,
            ).set(segment['get_state_time'])

            self.runner_segment_run_segment_time_g.labels(
                tag=self.tag,
                segment_idx=seg_idx,
            ).set(segment['run_segment_time'])

        # components
        self.runner_precycle_time_g.labels(
            tag=self.tag
        ).set(
            last_report['runner_precycle_time'])

        self.runner_postcycle_time_g.labels(
            tag=self.tag
        ).set(
            last_report['runner_postcycle_time'])

        self.sim_manager_segment_time_g.labels(
            tag=self.tag
        ).set(
            last_report['cycle_sim_manager_segment_time'])

        self.sim_manager_segment_overhead_time_g.labels(
            tag=self.tag
        ).set(
            last_report['sim_manager_segment_overhead_time'])

        # work mapper workers timings
        for worker_idx, segments in last_report['worker_segment_times'].items():

            for seg_idx, segment in enumerate(segments):
                self.mapper_seg_times_g.labels(
                    tag=self.tag,
                    worker_idx=worker_idx,
                    seg_idx=seg_idx,
                ).set(segment)


        logging.info(f"SimMonitor ({self}): done with cycle monitoring")
