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
    """Parameters
    ----------

    json_topology : str

    ligand_idxs : arraylike (1,)

    receptor_idxs : arraylike (1,)

    coords : N x 3 arraylike of float or simtk.Quantity
        If not a quantity will implicitly be treated as being in
        nanometers.

    box_vectors : simtk.Quantity
        If not a quantity will implicitly be treated as being in
        nanometers.

    cutoff : float or simtk.Quantity
        If not a quantity will implicitly be treated as being in
        nanometers.

    Returns
    -------

    binding_site_idxs : arraylike (1,)

    """


    # if they are simtk.units convert quantities to numbers in
    # nanometers
    if unit.is_quantity(cutoff):
        cutoff = cutoff.value_in_unit(unit.nanometer)

    if unit.is_quantity(coords):
        coords = coords.value_in_unit(unit.nanometer)

    if unit.is_quantity(box_vectors):
        box_vectors = box_vectors.value_in_unit(unit.nanometer)


    box_lengths, box_angles = box_vectors_to_lengths_angles(box_vectors)

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
