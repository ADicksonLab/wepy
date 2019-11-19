import simtk.unit as unit

from wepy.util.util import box_vectors_to_lengths_angles
from wepy.util.mdtraj import json_to_mdtraj_topology, mdtraj_to_json_topology

import mdtraj as mdj
import numpy as np


def binding_site_idxs(json_topology,
                      ligand_idxs, receptor_idxs,
                      coords,
                      box_vectors,
                      cutoff,
                      periodic=True):
    """

    Parameters
    ----------

    json_topology : str

    ligand_idxs : arraylike (1,)

    receptor_idxs : arraylike (1,)

    coords : simtk.Quantity

    box_vectors : simtk.Quantity

    cutoff : float

    Returns
    -------

    binding_site_idxs : arraylike (1,)

    """

    # convert quantities to numbers in nanometers
    cutoff = cutoff.value_in_unit(unit.nanometer)
    coords = coords.value_in_unit(unit.nanometer)

    box_lengths, box_angles = box_vectors_to_lengths_angles(box_vectors.value_in_unit(unit.nanometer))

    # make a trajectory to compute the neighbors from
    traj = mdj.Trajectory(np.array([coords]),
                          unitcell_lengths=[box_lengths],
                          unitcell_angles=[box_angles],
                          topology=json_to_mdtraj_topology(json_topology))

    neighbors_idxs = mdj.compute_neighbors(traj, cutoff, ligand_idxs,
                                           periodic=periodic)[0]

    # selects protein atoms from neighbors list
    binding_selection_idxs = np.intersect1d(neighbors_idxs, receptor_idxs)

    return binding_selection_idxs
