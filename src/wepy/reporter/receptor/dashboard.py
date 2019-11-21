from pint import UnitRegistry
from jinja2 import Template

from wepy.reporter.dashboard import BCDashboardSection

# initialize the unit registry
units = UnitRegistry()


class ReceptorBCDashboardSection(BCDashboardSection):


    BC_SECTION_TEMPLATE = \
"""

Boundary Condition: {{ name }}

Total Number of Dynamics segments: {{ total_n_walker_segments }}

Total Number of Warps: {{ total_crossings }}

Cumulative Boundary Crossed Weight: {{ total_unbound_weight }}

Rate (1/τ): {{ rate }}

Parameters:

{{ parameters }}


** Warping Log

{{ warping_log }}

"""


    def __init__(self, **kwargs):

        super().__init__(**kwargs)


    def gen_fields(self, **kwargs):

        fields = super().gen_fields(**kwargs)

        # since there is only one boundary to cross here we don't
        # really have to do any special reporting for meaningful
        # boundaries. So we just use the standard one.


        # we calculate the non-dimensional rate in terms of the cycle
        # numbers, then you would just have to multiply that number by
        # 1/time_per_cycle. In order to handle variable number of
        # walkers we just keep track of all of the segments that were
        # run total by keeping a running tally of how many
        # 'new_walkers' are received in the 'total_n_walkers'
        # attribute

        rate = self.total_crossed_weight / self.total_n_walker_segments

        new_fields = {
            'parameters' : '',
            'rate' : rate,
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

    def __init__(self, bc=None,
                 cutoff_distance=None,
                 **kwargs):

        if 'name' not in kwargs:
            kwargs['name'] = 'UnbindingBC'

        super().__init__(bc=bc,
                         cutoff_distance=cutoff_distance,
                         **kwargs)


        if bc is not None:
            self.cutoff_distance = bc.cutoff_distance
        else:
            assert cutoff_distance is not None, \
                "If no bc is given must give parameters: cutoff_distance"
            self.cutoff_distance = cutoff_distance

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

    def __init__(self, bc=None,
                 cutoff_rmsd=None,
                 **kwargs):

        if 'name' not in kwargs:
            kwargs['name'] = 'RebindingBC'

        super().__init__(bc=bc,
                         cutoff_rmsd=cutoff_rmsd,
                         **kwargs)

        if bc is not None:
            self.cutoff_rmsd = bc.cutoff_rmsd
        else:
            assert cutoff_rmsd is not None, \
                "If no bc is given must give parameters: cutoff_rmsd"
            self.cutoff_rmsd = cutoff_rmsd

    def gen_fields(self, **kwargs):

        fields = super().gen_fields(**kwargs)

        parameters_str = Template(self.RECEPTOR_PARAMETERS).render(
            cutoff_rmsd=self.cutoff_rmsd,
        )

        new_fields = {
            'parameters' : parameters_str,
        }

        return fields



