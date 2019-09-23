"""NAMD runner

NAMD is an external program to run molecular dynamics
simulations.  It is called using a subprocess command.

The NAMD runner uses normal WalkerState objects for the walkers
with entries for the positions, box_vectors, and velocities (optional).

NAMDWalker objects are defined below, which can carry the current cycle 
and the walker index.
  
"""
from copy import copy
import random as rand
from warnings import warn
import logging
import time
import os.path as osp
import os
import shutil
import subprocess

import numpy as np

from wepy.walker import Walker, WalkerState
from wepy.runners.runner import Runner
from wepy.work_mapper.worker import Worker

def generate_state(work_dir, output_pref, get_velocities=False):
    """Method for generating a wepy compliant state from a NAMD
    simulation and obtaining data about the last segment of dynamics.

    Parameters
    ----------

    output_pref : str
        A prefix to make filenames for reading coor, vel and xsc files.

    get_velocities : bool
        Whether or not to get read the velocities and add to the state.

    Returns
    -------

    new_state : wepy.runners.openmm.NAMDState object
        A new state from the simulation state.

    """

    # make an empty state dict
    state = {}

    # make filenames
    coor_name = output_pref + '.coor'
    box_name = output_pref + '.xsc'
    coor_path = osp.join(work_dir,coor_name)
    box_path = osp.join(work_dir,box_name)

    with open(coor_path,'rb') as f:
        natoms = np.fromfile(f,dtype=np.dtype('i'),count=1)[0]
        pos = np.fromfile(f,dtype=np.dtype('d'),count=3*natoms)

        # multiply by 0.1 to convert from angstroms to nanometers
        state['positions'] = 0.1*pos.reshape(natoms,3)

    # get box vectors
    with open(box_path,'r') as f:
        # skip the first two lines
        tmp = f.readline()
        tmp = f.readline()
        # parse the third to get the box vectors
        tmp = f.readline()
        vals = tmp.split(' ')
        state['box_vectors'] = np.array(vals[1:10],dtype=np.dtype('d')).reshape(3,3)

    if get_velocities:
        # make filenames
        vel_name = output_pref + '.vel'
        vel_path = osp.join(work_dir,vel_name)

        with open(vel_path,'rb') as f:
            natoms = np.fromfile(f,dtype=np.dtype('i'),count=1)[0]
            vel = np.fromfile(f,dtype=np.dtype('d'),count=3*natoms)

            # multiply by 0.1 to convert from angstroms/time to nanometers/time
            state['velocities'] = 0.1*vel.reshape(natoms,3)

    # make a WalkerState wrapper with this
    new_state = WalkerState(**state)

    return new_state

    
class NAMDRunner(Runner):
    """Runner for NAMD simulations."""

    def __init__(self, runcmd, common_dir_path, conf_file_path, work_dir_path, get_velocities=False):
        """Constructor for NAMDRunner.

        Parameters
        ----------
        runcmd : str
            Command to run NAMD, should include either an absolute filename
            or one that is in the PATH at runtime.  Parallelization options 
            such as +p4 should be included here, if desired.

        common_dir_path : str
            Location of a folder that contains any common files (e.g. psf,
            forcefield, etc.) that are required by NAMD.

        conf_file_path : str
            Location of a generic NAMD configuration containing wildcard
            entries for 'coordinates', 'outputName' and 'nsteps'

        work_dir_path : str
            Location of working directory 
        
        """

        self.runcmd = runcmd

        with open(conf_file_path,"r") as f:
            self.conf_text = f.read()

        self.work_dir = work_dir_path
        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)

        # copy all files from common dir to work dir
        src_files = os.listdir(common_dir_path)
        for file_name in src_files:
            full_file_name = osp.join(common_dir_path, file_name)
            if osp.isfile(full_file_name):
                shutil.copy(full_file_name, self.work_dir)

        self.get_vel = get_velocities

    def run_segment(self, walker, segment_length, walker_idx=-1, DeviceIndex=0):
        """Run dynamics for the walker.

        Parameters
        ----------
        walker : object implementing the Walker interface
            The walker for which dynamics will be propagated.

        segment_length : int or float
            The numerical value that specifies how many dynamics steps are to be run.

        Returns
        -------
        new_walker : object implementing the Walker interface
            Walker after dynamics was run, only the state should be modified.

        """

        assert isinstance(walker,NAMDWalker), "Error! walker must be a NAMDWalker"
        
        run_segment_start = time.time()

        # build the conf file
        # grab information from walker
        nextinput = walker.nextinput
        thiscycle = walker.cycle + 1
        output_pref = 'walker{0}_{1}'.format(walker_idx,thiscycle)

        # write new conf file
        tmp1 = self.conf_text.replace('TMP_INPUT_NAME',nextinput)
        tmp2 = tmp1.replace('TMP_NSTEPS',str(segment_length))
        new_conf = tmp2.replace('TMP_OUTPUT_NAME',output_pref)
        new_conf_file_name = osp.join(self.work_dir,'seg_{0}_{1}.conf'.format(walker_idx,thiscycle))
        with open(new_conf_file_name,"w") as f:
            f.write(new_conf)

        # actually run the simulation
        steps_start = time.time()
        cmd = self.runcmd.split(' ') + [new_conf_file_name]

        # use current environment, but change CUDA_VISIBLE_DEVICES
        new_env = os.environ.copy()
        new_env['CUDA_VISIBLE_DEVICES'] = DeviceIndex
        out_file = osp.join(self.work_dir,'seg_{0}_{1}.log'.format(walker_idx,thiscycle))
        err_file = osp.join(self.work_dir,'seg_{0}_{1}.err'.format(walker_idx,thiscycle))

        f_out = open(out_file,'w')
        f_err = open(err_file,'w')
        completed_process = subprocess.run(cmd, cwd=self.work_dir, env=new_env, stdout=f_out, stderr=f_err)
        f_out.close()
        f_err.close()

        steps_end = time.time()
        steps_time = steps_end - steps_start
        logging.info("Time to run {} sim steps: {}".format(segment_length, steps_time))

        get_state_start = time.time()
        
        
        get_state_end = time.time()
        get_state_time = get_state_end - get_state_start
        logging.info("Getting context state time: {}".format(get_state_time))

        # generate the new state/walker
        new_state = generate_state(self.work_dir,output_pref,get_velocities=self.get_vel)

        # create a new walker for this
        new_walker = NAMDWalker(new_state, walker.weight, cycle=thiscycle, next_input_pref=output_pref)

        run_segment_end = time.time()
        run_segment_time = run_segment_end - run_segment_start
        logging.info("Total internal run_segment time: {}".format(run_segment_time))

        return new_walker

class NAMDWalker(Walker):
    """Walker for NAMDRunner simulations.
    Accepts standard WalkerState objects for state.  Carries the current cycle
    and the walker index.
    """

    def __init__(self, state, weight, cycle=0, next_input_pref=None):
        # documented in superclass

        super().__init__(state, weight)

        self.cycle = cycle
        self.nextinput = next_input_pref
