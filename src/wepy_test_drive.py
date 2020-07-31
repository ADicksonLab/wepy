import click
import logging
from multiprocessing_logging import install_mp_handler

import simtk.unit as unit

# import all of the sim makers we have available
from wepy_tools.sim_makers.openmm import *

SYSTEM_SIM_MAKERS = {
    'LennardJonesPair' : LennardJonesPairOpenMMSimMaker,
    'LysozymeImplicit' : LysozymeImplicitOpenMMSimMaker,
}

def parse_system_spec(spec):

    sys_spec, runner_platform = spec.split("/")

    runner, platform = runner_platform.split('-')

    return sys_spec, runner, platform

@click.option('-v', '--verbose', is_flag=True)
@click.option('-W', '--work-mapper',
              default='WorkerMapper',
              help="Work mapper for doing work.")
@click.option('-R', '--resampler',
              default='WExplore',
              help="Resampling algorithm.")
@click.argument('n_workers', type=int)
@click.argument('tau', type=float)
@click.argument('n_cycles', type=int)
@click.argument('n_walkers', type=int)
@click.argument('system')
@click.command()
def cli(
        verbose,
        work_mapper,
        resampler,
        n_workers,
        tau,
        n_cycles,
        n_walkers,
        system,
):
    """Run a pre-parametrized wepy simulation.

    \b
    Parameters
    ----------

    \b
    SYSTEM : str
        Which pre-parametrized simulation to run should have the format: System/Runner-Platform

    \b
    N_WALKERS : int
        Number of parallel trajectories to run

    \b
    N_CYCLES : int
        How many cycles to run the simulation for

    \b
    TAU : float
        Cycle simulation time in picoseconds

    \b
    N_WORKERS : int
        Number of worker processes to run on

    \b
    Available Systems
    -----------------

    LennardJonesPair : A pair of Lennard-Jones particles

    LysozymeImplicit : Lysozyme-xylene receptor ligand in implicit solvent (2621 atoms)

    \b
    Available Runners/Platforms
    ---------------------------

    \b
    OpenMM-
      Reference
      CPU
      OpenCL (GPU)
      CUDA (GPU)


    \b
    Available Work Mappers
    ----------------------

    WorkerMapper (default) : parallel python multiprocessing based
                             worker-consumer concurrency model

    WIP not available in test drive yet:

    TaskMapper : parallel python multiprocessing based task-process
                 based concurrency model

    Mapper : non-parallel single-process implementation


    \b
    Available Resamplers
    --------------------

    No : Doesn't do any resampling. Simply runs an ensemble of walkers.

    WExplore : Hierarchical History Dependent Voronoi Binning

    REVO : Stateless and Binless algorithm that rewards in-ensemble novelty.

    \b
    Examples
    --------

    python -m wepy_test_drive LennardJonesPair/OpenMM-CPU 20 10 2 4

    \b
    Notes
    -----

    When using a GPU platform your number of workers should be the
    number of GPUs you want to use.

    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        install_mp_handler()
        logging.debug("Starting the test")


    resampler_fullname = resampler + 'Resampler'

    sys_spec, runner, platform = parse_system_spec(system)

    # choose which sim_maker to use
    sim_maker = SYSTEM_SIM_MAKERS[sys_spec]()

    apparatus = sim_maker.make_apparatus(
        platform = platform,
        resampler = resampler_fullname,
    )

    # compute the number of steps to take from tau
    tau = tau * unit.picosecond
    n_steps = round(tau / apparatus.filters[0].integrator.getStepSize())

    config = sim_maker.make_configuration(apparatus,
                                          work_mapper_spec=work_mapper,
                                          platform=platform)

    sim_manager = sim_maker.make_sim_manager(n_walkers, apparatus, config)

    # run the simulation
    sim_manager.run_simulation(n_cycles, n_steps, num_workers=n_workers)


if __name__ == "__main__":

    cli()
