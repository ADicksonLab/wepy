from pint import UnitRegistry


from wepy.reporter.dashboard import RunnerDashboardSection

# initialize the unit registry
units = UnitRegistry()

class OpenMMRunnerDashboardSection(RunnerDashboardSection):

    RUNNER_SECTION_TEMPLATE = \
"""

Runner: {{ name }}

Integration Step Size: {{ step_time }}

Single Walker Sampling Time: {{ walker_total_sampling_time }}

Total Sampling Time: {{ total_sampling_time }}
"""

    def __init__(self, runner=None,
                 step_time=None,
                 **kwargs
    ):

        if 'name' not in kwargs:
            kwargs['name'] = 'OpenMMRunner'

        super().__init__(runner=runner,
                         step_time=step_time,
                         **kwargs)

        if runner is None:
            assert step_time is not None, "If no complete runner is given must give parameters: step_time"

            # assume it has units
            self.step_time = step_time

        else:

            simtk_step_time = runner.integrator.getStepSize()
            simtk_val = simtk_step_time.value_in_unit(simtk_step_time.unit)

            # convert to a more general purpose pint unit, which will be
            # used for the dashboards so we don't have the simtk
            # dependency
            self.step_time = simtk_val * units(simtk_step_time.unit.get_name())


        # TODO

        # integrator and params


        # FF and params

        # updatables
        self.walker_total_sampling_time = 0.0 * units('microsecond')
        self.total_sampling_time = 0.0 * units('microsecond')

    def update_values(self, **kwargs):

        super().update_values(**kwargs)

        # amount of new sampling time for each walker
        new_walker_sampling_time = self.step_time * kwargs['n_segment_steps']

        # accumulated sampling time for a single walker
        self.walker_total_sampling_time += new_walker_sampling_time

        # amount of sampling time for all walkers
        new_sampling_time = new_walker_sampling_time * len(kwargs['new_walkers'])

        # accumulated sampling time for the ensemble
        self.total_sampling_time += new_sampling_time


    def gen_fields(self, **kwargs):

        fields = super().gen_fields(**kwargs)

        ## formatting

        # units for the different time scales
        step_size_unit_spec = 'femtosecond'
        sampling_time_unit_spec = 'microsecond'

        # get the unit objects for them
        step_size_unit = units(step_size_unit_spec)
        sampling_time_unit = units(sampling_time_unit_spec)

        # step size string
        step_size_str = str(self.step_time.to(step_size_unit))

        # single walker sampling
        walker_samp_str = str(self.walker_total_sampling_time.to(sampling_time_unit))

        # total sampling
        total_samp_str = str(self.total_sampling_time.to(sampling_time_unit))

        new_fields = {
            'step_time' : step_size_str,
            'walker_total_sampling_time' : walker_samp_str,
            'total_sampling_time' : total_samp_str,
        }

        fields.update(new_fields)

        return fields
