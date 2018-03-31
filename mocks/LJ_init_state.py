import os
import os.path as osp
import pickle
from copy import copy

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from openmmtools.testsystems import LennardJonesPair

from wepy.runners.openmm import OpenMMRunner, OpenMMState
from wepy.runners.openmm import UNIT_NAMES, GET_STATE_KWARG_DEFAULTS

# make the test system from openmmtools
test_sys = LennardJonesPair()

TEMPERATURE= 300.0*unit.kelvin
FRICTION_COEFFICIENT = 1/unit.picosecond
STEP_SIZE = 0.002*unit.picoseconds
integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

# make a context and set the positions
context = omm.Context(test_sys.system, copy(integrator))
context.setPositions(test_sys.positions)

# get the data from this context so we have a state to start the
# simulation with
get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
init_sim_state = context.getState(**get_state_kwargs)
init_state = OpenMMState(init_sim_state)

with open("LJ_init_state.pkl", 'wb') as wf:
    pickle.dump(init_state, wf)
