
from wepy.reporter.dashboard import BCDashboardSection

class ReceptorBCDashboardSection(BCDashboardSection):


        BC_SECTION_TEMPLATE = \
"""

Boundary Condition: Receptor

Cutoff Distance: {{ cutoff_distance }}

Number of Exit Points this Cycle: {{ cycle_n_exit_points }}

Total Number of Exit Points: {{ n_exit_points }}

Cumulative Unbound Weight {{ total_unbound_weight }}

Expected Reactive Traj. Time: {{ expected_unbinding_time }} seconds
Expected Reactive Traj. Rate: {{ reactive_traj_rate }} 1/seconds

Rate: {{ exit_rate }} 1/seconds

** Warping Log
{{ warping_log }}

"""


    def __init__(self, bc):

        super().__init__(bc)


        # updatables

    def update_values(self, **kwargs):

        super().update_values(**kwargs)

    def gen_fields(self, **kwargs):

        fields = super().gen_fields(**kwargs)


        new_fields = {

            

        }


        # combine the superclass fields with the fields here,
        # overwriting them from the superclass if they were redefined
        # explicitly
        fields.update(new_fields)

        return fields

