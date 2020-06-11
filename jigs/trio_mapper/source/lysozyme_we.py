from pympler.asizeof import asizeof
import prometheus_client as prom

class SimMonitor:

    def __init__(self,
                 tag="",
                 reporter_order=(
                     'hdf5_reporter',
                     'dashboard_reporter',
                 ),
    ):

        self.tag = tag

        self.reporter_order = reporter_order

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


    def cycle_monitor(self, sim_manager, walkers):

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


def get_size(obj):
    """get the size in units of Mb"""

    return asizeof(obj) / 1000000



if __name__ == "__main__":

    prom.start_http_server(9001)

    import os
    import shutil
    import sys
    import logging
    from pathlib import Path

    # from multiprocessing_logging import install_mp_handler

    from wepy_tools.sim_makers.openmm.lysozyme import LysozymeImplicitOpenMMSimMaker

    logging.getLogger().setLevel(logging.DEBUG)

    # install_mp_handler()

    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("arguments: n_cycles, n_steps, n_walkers, n_workers, platform, resampler, work_mapper, tag")
        exit()
    else:
        n_cycles = int(sys.argv[1])
        n_steps = int(sys.argv[2])
        n_walkers = int(sys.argv[3])
        n_workers = int(sys.argv[4])
        platform = sys.argv[5]
        resampler = sys.argv[6]
        work_mapper = sys.argv[7]
        tag = sys.argv[8]

        print("Number of steps: {}".format(n_steps))
        print("Number of cycles: {}".format(n_cycles))


    output_dir = Path('_output')
    result_dir = output_dir / 'we_lysozyme'

    # make the results directory if not already made
    try:
        shutil.rmtree(result_dir)
    except FileNotFoundError:
        pass

    os.makedirs(result_dir, exist_ok=True)

    sim_maker = LysozymeImplicitOpenMMSimMaker()

    apparatus = sim_maker.make_apparatus(
        integrator='LangevinIntegrator',
        resampler=resampler,
        bc='UnbindingBC',
        platform=platform,
    )

    # work_mapper_params = {
    #     'num_workers' : n_workers,
    # }

    work_mapper_spec = None
    work_mapper_class = None
    if work_mapper == "TrioMapper":

        work_mapper_spec = None
        work_mapper_class = TrioMapper

    else:
        work_mapper_spec = work_mapper
        work_mapper_class = None

    config = sim_maker.make_configuration(apparatus,
                                          work_mapper_class=work_mapper_class,
                                          work_mapper_spec=work_mapper_spec,
                                          platform=platform,
                                          work_dir=str(result_dir))

    ## set up profiling and initial stats

    print("Orchestration objects")
    print("----------------------------------------")
    print(f"Sim maker size: {get_size(sim_maker)} Mb")
    print(f"Apparatus size: {get_size(apparatus)} Mb")
    print(f"Configuration size: {get_size(config)} Mb")
    print("----------------------------------------\n")

    sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)

    def print_sim_objs(sim_manager, reporter_order):

        print(f"init_walkers: {get_size(sim_manager.init_walkers)} Mb")
        print(f"runner: {get_size(sim_manager.runner)} Mb")
        print(f"resampler: {get_size(sim_manager.resampler)} Mb")
        print(f"bc: {get_size(sim_manager.boundary_conditions)} Mb")
        print(f"mapper: {get_size(sim_manager.work_mapper)} Mb")

        for name, reporter in zip(reporter_order, sim_manager.reporters):
            print(f"reporter {name}: {get_size(reporter)} Mb")

        print(f"sim_manager: {get_size(sim_manager)} Mb")

    reporter_order = (
        'hdf5_reporter',
        'dashboard_reporter',
        # 'restree_reporter',
        'walker_reporter',
    )

    sim_monitor = SimMonitor(
        tag=tag,
        reporter_order=reporter_order)


    print("Starting run")
    print("----------------------------------------")
    print_sim_objs(sim_manager, reporter_order)
    print("----------------------------------------\n")

    sim_manager.run_simulation(n_cycles, n_steps,
                               num_workers=n_workers,
                               sim_monitor=sim_monitor)

    print("Finished run")
    print("----------------------------------------")
    print_sim_objs(sim_manager, reporter_order)
    print("----------------------------------------\n")
