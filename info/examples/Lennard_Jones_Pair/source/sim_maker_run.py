
if __name__ == "__main__":

    import os
    import shutil
    import sys
    import logging
    from multiprocessing_logging import install_mp_handler

    from wepy_tools.sim_makers.openmm.lennard_jones import LennardJonesPairOpenMMSimMaker

    OUTPUT_DIR = "_output/sim_maker_run"

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

    # make the results directory if not already made
    try:
        shutil.rmtree(OUTPUT_DIR)
    except FileNotFoundError:
        pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sim_maker = LennardJonesPairOpenMMSimMaker()

    apparatus = sim_maker.make_apparatus(
        integrator='LangevinIntegrator',
        resampler=resampler,
        bc='UnbindingBC',
        platform=platform,
    )
    config = sim_maker.make_configuration(apparatus,
                                          work_mapper_spec='Mapper',
                                          platform=platform,
                                          work_dir=OUTPUT_DIR)

    sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)

    sim_manager.run_simulation(n_cycles, n_steps, num_workers=n_workers)
