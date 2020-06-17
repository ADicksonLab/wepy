from pympler.asizeof import asizeof

def get_size(obj):
    """get the size in units of Mb"""

    return asizeof(obj) / 1000000


if __name__ == "__main__":

    # prom.start_http_server(9001)

    import os
    import shutil
    import sys
    import logging
    from pathlib import Path

    # from multiprocessing_logging import install_mp_handler

    from wepy_tools.monitoring.prometheus import SimMonitor
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

    work_mapper_spec = work_mapper
    work_mapper_class = None

    work_mapper_params = {
        'platform' : platform,
        'device_ids' : [str(i) for i in range(n_workers)],
    }


    monitor_class = SimMonitor
    monitor_params = {
        'tag' : tag,
        'port' : 9001,
    }

    config = sim_maker.make_configuration(apparatus,
                                          work_mapper_class=work_mapper_class,
                                          work_mapper_spec=work_mapper_spec,
                                          work_mapper_params=work_mapper_params,
                                          platform=platform,
                                          work_dir=str(result_dir),
                                          monitor_class=monitor_class,
                                          monitor_params=monitor_params,
    )

    breakpoint()

    ## set up profiling and initial stats

    print("Orchestration objects")
    print("----------------------------------------")
    print(f"Sim maker size: {get_size(sim_maker)} Mb")
    print(f"Apparatus size: {get_size(apparatus)} Mb")
    print(f"Configuration size: {get_size(config)} Mb")
    print("----------------------------------------\n")

    sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)

    print("Starting run")
    print("----------------------------------------")

    sim_manager.run_simulation(n_cycles, n_steps,
                               num_workers=n_workers)

    print("----------------------------------------")
    print("Finished run")

