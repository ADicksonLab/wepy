import json
from warnings import warn
import operator

import numpy as np

import mdtraj as mdj
import mdtraj.core.element as elem

def mdtraj_to_json_topology(mdj_top):
    """ Copied in part from MDTraj.formats.hdf5.topology setter. """

    topology_dict = {
        'chains': [],
        'bonds': []
    }

    for chain in mdj_top.chains:
        chain_dict = {
            'residues': [],
            'index': int(chain.index)
        }
        for residue in chain.residues:
            residue_dict = {
                'index': int(residue.index),
                'name': str(residue.name),
                'atoms': [],
                "resSeq": int(residue.resSeq)
            }

            for atom in residue.atoms:

                try:
                    element_symbol_string = str(atom.element.symbol)
                except AttributeError:
                    element_symbol_string = ""

                residue_dict['atoms'].append({
                    'index': int(atom.index),
                    'name': str(atom.name),
                    'element': element_symbol_string
                })
            chain_dict['residues'].append(residue_dict)
        topology_dict['chains'].append(chain_dict)

    for atom1, atom2 in mdj_top.bonds:
        topology_dict['bonds'].append([
            int(atom1.index),
            int(atom2.index)
        ])

    top_json_str = json.dumps(topology_dict)

    return top_json_str

def json_to_mdtraj_topology(json_string):
    """ Copied in part from MDTraj.formats.hdf5 topology property."""

    topology_dict = json.loads(json_string)

    topology = mdj.Topology()

    for chain_dict in sorted(topology_dict['chains'], key=operator.itemgetter('index')):
        chain = topology.add_chain()
        for residue_dict in sorted(chain_dict['residues'], key=operator.itemgetter('index')):
            try:
                resSeq = residue_dict["resSeq"]
            except KeyError:
                resSeq = None
                warn('No resSeq information found in HDF file, defaulting to zero-based indices')
            try:
                segment_id = residue_dict["segmentID"]
            except KeyError:
                segment_id = ""
            residue = topology.add_residue(residue_dict['name'], chain,
                                           resSeq=resSeq, segment_id=segment_id)
            for atom_dict in sorted(residue_dict['atoms'], key=operator.itemgetter('index')):
                try:
                    element = elem.get_by_symbol(atom_dict['element'])
                except KeyError:
                    element = elem.virtual
                topology.add_atom(atom_dict['name'], element, residue)

    atoms = list(topology.atoms)
    for index1, index2 in topology_dict['bonds']:
        topology.add_bond(atoms[index1], atoms[index2])

    return topology
