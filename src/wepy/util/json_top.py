import json

import pandas as pd

CHAIN_KEY = 'index'
RESIDUE_KEY = 'index'
ATOM_KEY = 'index'
CHAIN_ATTR_KEYS = (CHAIN_KEY,)
RESIDUE_ATTR_KEYS = (RESIDUE_KEY, 'name', 'resSeq', 'segmentID')
ATOM_ATTR_KEYS = (ATOM_KEY, 'name', 'element')

def json_top_chain_fields(json_topology):
    """Creates a fields dictionary for all of the chains in a topology.

    Parameters
    ----------

    json_topology : str
        JSON format topology

    Returns
    -------

    chain_cols : dict of str: list

    """


    top = json.loads(json_topology)

    chain_cols = {key : [] for key in CHAIN_ATTR_KEYS}

    for chain in top['chains']:
        for key in CHAIN_ATTR_KEYS:
            chain_cols[key].append(chain[key])

    return chain_cols

def json_top_chain_df(json_topology):
    """Make a pandas dataframe for the chains in the JSON topology.

    Parameters
    ----------

    json_topology : str
        JSON format topology

    Returns
    -------

    chain_df : pandas.DataFrame

    """

    return pd.DataFrame(json_top_chain_fields(json_topology))

def json_top_residue_fields(json_topology):
    """Creates a fields dictionary for all of the residues in a topology.

    Parameters
    ----------

    json_topology : str
        JSON format topology

    Returns
    -------

    residue_cols : dict of str: list

    """


    top = json.loads(json_topology)

    residue_cols = {key : [] for key in RESIDUE_ATTR_KEYS}
    residue_cols['chain_key'] = []

    for chain in top['chains']:
        for residue in chain['residues']:
            residue_cols['chain_key'].append(chain[CHAIN_KEY])
            for key in RESIDUE_ATTR_KEYS:
                residue_cols[key].append(residue[key])

    return residue_cols

def json_top_residue_df(json_topology):
    """Make a pandas dataframe for the residues in the JSON topology.

    Parameters
    ----------

    json_topology : str
        JSON format topology

    Returns
    -------

    residue_df : pandas.DataFrame

    """

    return pd.DataFrame(json_top_residue_fields(json_topology))


def json_top_atom_fields(json_topology):
    """Creates a fields dictionary for all of the atoms in a topology.

    Parameters
    ----------

    json_topology : str
        JSON format topology

    Returns
    -------

    atom_cols : dict of str: list

    """

    top = json.loads(json_topology)

    atom_cols = {key : [] for key in ATOM_ATTR_KEYS}
    atom_cols['chain_key'] = []
    atom_cols['residue_key'] = []

    for chain in top['chains']:
        for residue in chain['residues']:
            for atom in residue['atoms']:
                atom_cols['chain_key'].append(chain[CHAIN_KEY])
                atom_cols['residue_key'].append(residue[RESIDUE_KEY])
                for key in ATOM_ATTR_KEYS:
                    atom_cols[key].append(atom[key])

    return atom_cols

def json_top_atom_df(json_topology):
    """Make a pandas dataframe for the atoms in the JSON topology.

    Parameters
    ----------

    json_topology : str
        JSON format topology

    Returns
    -------

    atoms_df : pandas.DataFrame

    """

    return pd.DataFrame(json_top_atom_fields(json_topology))


def json_top_atom_count(json_str):
    """Count the number of atoms in the JSON topology used by wepy HDF5.

    Parameters
    ----------
    json_str : str
        A string of valid JSON in the format of JSON used in WepyHDF5
        and mdtraj HDF5 format.

    Returns
    -------
    n_atoms : int
        Number of atoms in the topology.

    """

    top_d = json.loads(json_str)
    atom_count = 0
    atom_count = 0
    for chain in top_d['chains']:
        for residue in chain['residues']:
            atom_count += len(residue['atoms'])

    return atom_count

def json_top_subset(json_str, atom_idxs):
    """Given a JSON topology and atom indices from that topology returns
    another JSON topology which is a subset of the first, preserving
    the topology between remaining atoms. The atoms will be ordered in
    the order in which the indices are given.

    Parameters
    ----------

    json_str : str
        A string of valid JSON in the format of JSON used in WepyHDF5
        and mdtraj HDF5 format.

    atom_idxs : list of int
        The atoms for which you want to make a subset of.

    Returns
    -------

    subset_json_str : str
        JSON string of the subset of atoms. Ordering preserved.

    """

    # cast so we can use the list index method
    atom_idxs = list(atom_idxs)

    # do checks on the atom idxs

    # no duplicates
    assert len(set(atom_idxs)) == len(atom_idxs), "duplicate atom indices"

    top = json.loads(json_str)

    # the dictionaries for each thing indexed by their old index
    atom_data_ds = {}
    residue_data_ds = {}

    # mapping of old atom indices to the residue data they belong to
    atom_res_idxs = {}
    res_chain_idxs = {}

    # go through and collect data on the atoms and convert indices to
    # the new ones
    for chain in top['chains']:

        for residue in chain['residues']:
            res_chain_idxs[residue['index']] = chain['index']

            # create the data dict for this residue
            residue_data_ds[residue['index']] = residue



            for atom in residue['atoms']:
                atom_res_idxs[atom['index']] = residue['index']

                # if the current atom's index is in the selection
                if atom['index'] in atom_idxs:

                    # we add this to the mapping by getting the index
                    # of the atom in subset
                    new_idx = atom_idxs.index(atom['index'])

                    # save the atom attributes
                    atom_data_ds[atom['index']] = atom

    # initialize the data structure for the topology subset
    top_subset = {'chains' : [],
                  'bonds' : []}

    residue_idx_map = {}
    chain_idx_map = {}

    old_to_new_atoms = {}

    # initialize the new indexing of the chains and residues
    new_res_idx_counter = 0
    new_chain_idx_counter = 0
    # now in the new order go through and create the topology
    for new_atom_idx, old_atom_idx in enumerate(atom_idxs):

        old_to_new_atoms[old_atom_idx] = new_atom_idx

        atom_data = atom_data_ds[old_atom_idx]

        # get the old residue index
        old_res_idx = atom_res_idxs[old_atom_idx]

        # since the datastrucutre is hierarchical starting with the
        # chains and residues we work our way back up craeting these
        # if neceessary for this atom, once this is taken care of we
        # will add the atom to the data structure
        if old_res_idx not in residue_idx_map:
            residue_idx_map[old_res_idx] = new_res_idx_counter
            new_res_idx_counter += 1

            # do the same but for the chain
            old_chain_idx = res_chain_idxs[old_res_idx]

            # make it if necessary
            if old_chain_idx not in chain_idx_map:
                chain_idx_map[old_chain_idx] = new_chain_idx_counter
                new_chain_idx_counter += 1

                # and add the chain to the topology
                new_chain_idx = chain_idx_map[old_chain_idx]
                top_subset['chains'].append({'index' : new_chain_idx,
                                             'residues' : []})

            # add the new index to the dats dict for the residue
            res_data = residue_data_ds[old_res_idx]
            res_data['index'] = residue_idx_map[old_res_idx]
            # clear the atoms
            res_data['atoms'] = []

            # add the reside to the chain idx
            new_chain_idx = chain_idx_map[old_chain_idx]
            top_subset['chains'][new_chain_idx]['residues'].append(res_data)

        # now that (if) we have made the necessary chains and residues
        # for this atom we replace the atom index with the new index
        # and add it to the residue
        new_res_idx = residue_idx_map[old_res_idx]
        new_chain_idx = chain_idx_map[res_chain_idxs[old_res_idx]]

        atom_data['index'] = new_atom_idx

        top_subset['chains'][new_chain_idx]['residues'][new_res_idx]['atoms'].append(atom_data)

    # then translate the atom indices in the bonds
    new_bonds = []
    for bond_atom_idxs in top['bonds']:

        if all([True if a_idx in old_to_new_atoms else False
                for a_idx in bond_atom_idxs]):
            new_bond_atom_idxs = [old_to_new_atoms[a_idx] for a_idx in bond_atom_idxs
                                  if a_idx in old_to_new_atoms]
            new_bonds.append(new_bond_atom_idxs)

    top_subset['bonds'] = new_bonds


    return json.dumps(top_subset)
