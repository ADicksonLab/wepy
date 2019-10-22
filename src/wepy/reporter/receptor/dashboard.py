from pint import UnitRegistry
from jinja2 import Template

from wepy.reporter.dashboard import BCDashboardSection

# initialize the unit registry
units = UnitRegistry()


class ReceptorBCDashboardSection(BCDashboardSection):


    BC_SECTION_TEMPLATE = \
"""

Boundary Condition: {{ name }}

Total Number of Warps: {{ n_exit_points }}

Cumulative Boundary Crossed Weight {{ total_unbound_weight }}

Parameters:

{{ parameters }}


** Warping Log

{{ warping_log }}

"""


    def __init__(self, bc):

        super().__init__(bc)

        # updatables
        self.warp_records = []
        self.exit_point_weights = []
        self.exit_point_times = []
        self.n_exit_points = 0
        self.total_unbound_weight = 0.0


    def update_values(self, **kwargs):

        super().update_values(**kwargs)

        for warp_record in kwargs['warp_data']:

            weight = warp_record['weight'][0]
            walker_idx = warp_record['walker_idx'][0]

            # make a record for a table that includes the time in
            # walkers sampling time
            record = (walker_idx, weight, kwargs['cycle_idx'])
            self.warp_records.append(record)

            # also add them to the individual records
            self.exit_point_weights.append(weight)

            # STUB
            # this is flawed since this is not the individual walker
            # time, we aren't really that interested in it anyways.
            # self.exit_point_times.append(self.walker_total_sampling_time)

            # increase the number of exit points by 1
            self.n_exit_points += 1


            # total accumulated unbound probability
            self.total_unbound_weight += weight

        # STUB: see note above we don't have proper walker sampling
        # times, and it isn't that interesting anyways. The exit rate
        # is more useful

        # # calculate the expected value of unbinding times
        # self.expected_unbinding_time = np.sum([self.exit_point_weights[i] * self.exit_point_times[i]
        #                                        for i in range(self.n_exit_points)])

        # # expected rate of reactive trajectories
        # self.reactive_traj_rate = 1 / self.expected_unbinding_time


    def gen_fields(self, **kwargs):

        fields = super().gen_fields(**kwargs)

        # STUB: because the total sampling time is from a different
        # section we have to remove this for now until we figure that
        # out

        # calculate the new rate using the Hill relation after taking
        # into account all of these warps
        # self.exit_rate = self.total_unbound_weight / self.total_sampling_time

        new_fields = {
            'parameters' : '',
            'cycle_n_exit_points' : self.n_exit_points,
            'total_unbound_weight' : self.total_unbound_weight,
        }


        # combine the superclass fields with the fields here,
        # overwriting them from the superclass if they were redefined
        # explicitly
        fields.update(new_fields)

        return fields



class UnbindingBCDashboardSection(ReceptorBCDashboardSection):

    RECEPTOR_PARAMETERS = \
"""
Cutoff Distance: {{ cutoff_distance }}
"""

    def __init__(self, bc):

        super().__init__(bc)

        # bc params
        self.cutoff_distance = bc.cutoff_distance

    def gen_fields(self, **kwargs):

        fields = super().gen_fields(**kwargs)

        parameters_str = Template(self.RECEPTOR_PARAMETERS).render(
            cutoff_distance=self.cutoff_distance,
        )

        new_fields = {
            'parameters' : parameters_str,
        }

        return fields

class RebindingBCDashboardSection(ReceptorBCDashboardSection):

    RECEPTOR_PARAMETERS = \
"""
Cutoff RMSD: {{ cutoff_rmsd }}
"""

    def __init__(self, bc):

        super().__init__(bc)

        self.cutoff_rmsd = bc.cutoff_rmsd

    def gen_fields(self, **kwargs):

        fields = super().gen_fields(**kwargs)

        parameters_str = Template(self.RECEPTOR_PARAMETERS).render(
            cutoff_rmsd=self.cutoff_rmsd,
        )

        new_fields = {
            'parameters' : parameters_str,
        }

        return fields



