
if __name__ == "__main__":

    import sys
    import logging
    from multiprocessing_logging import install_mp_handler

    from wepy_tools.sim_makers.openmm.lysozyme import LysozymeImplicitOpenMMSimMaker

    logging.getLogger().setLevel(logging.DEBUG)
    install_mp_handler()

    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("arguments: n_cycles, n_steps, n_walkers, n_workers")
    else:
        n_cycles = int(sys.argv[1])
        n_steps = int(sys.argv[2])
        n_walkers = int(sys.argv[3])
        n_workers = int(sys.argv[4])

        print("Number of steps: {}".format(n_steps))
        print("Number of cycles: {}".format(n_cycles))

    sim_maker = LysozymeImplicitOpenMMSimMaker()

    apparatus = sim_maker.make_apparatus(platform='OpenCL')
    config = sim_maker.make_configuration(work_mapper='TaskMapper', platform='OpenCL')

    sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)

    sim_manager.run_simulation(n_cycles, n_steps, num_workers=n_workers)

