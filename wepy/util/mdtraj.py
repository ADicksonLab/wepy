import json
from warnings import warn
import operator

import numpy as np

import mdtraj as mdj
import mdtraj.core.element as elem

# the following method contains portions of the software mdtraj which
# is distributed under the following license
##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2014 Stanford University and the Authors
#
# Authors: Peter Eastman, Robert McGibbon
# Contributors: Kyle A. Beauchamp, Matthew Harrigan, Carlos Xavier Hernandez
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
#
# Portions of this code originate from the OpenMM molecular simulation
# toolkit, copyright (c) 2012 Stanford University and Peter Eastman. Those
# portions are distributed under the following terms:
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
##############################################################################

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
