
if __name__ == "__main__":

    import os
    import shutil
    import sys
    import logging
    from pathlib import Path

    from multiprocessing_logging import install_mp_handler

    from wepy_tools.sim_makers.openmm.lysozyme import LysozymeImplicitOpenMMSimMaker

    logging.getLogger().setLevel(logging.DEBUG)
    install_mp_handler()

    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("arguments: n_cycles, n_steps, n_walkers, n_workers, platform, resampler")
        exit()
    else:
        n_cycles = int(sys.argv[1])
        n_steps = int(sys.argv[2])
        n_walkers = int(sys.argv[3])
        n_workers = int(sys.argv[4])
        platform = sys.argv[5]
        resampler = sys.argv[6]

        print("Number of steps: {}".format(n_steps))
        print("Number of cycles: {}".format(n_cycles))


    output_dir = Path('_output')

    # make the results directory if not already made
    try:
        shutil.rmtree(output_dir / 'we')
    except FileNotFoundError:
        pass

    os.makedirs(output_dir / 'we', exist_ok=True)

    sim_maker = LysozymeImplicitOpenMMSimMaker()

    apparatus = sim_maker.make_apparatus(
        integrator='LangevinIntegrator',
        resampler=resampler,
        bc='UnbindingBC',
        platform=platform,
    )
    config = sim_maker.make_configuration(apparatus,
                                          work_mapper_spec='TaskMapper',
                                          platform=platform,
                                          work_dir=str(output_dir / 'we'))

    sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)

    sim_manager.run_simulation(n_cycles, n_steps, num_workers=n_workers)
