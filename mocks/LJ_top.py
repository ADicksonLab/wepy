from openmmtools.testsystems import LennardJonesPair
import mdtraj as mdj

from wepy.util.mdtraj import mdtraj_to_json_topology

sys = LennardJonesPair()
mdj_top = mdj.Topology.from_openmm(sys.topology)
top_json = mdtraj_to_json_topology(mdj_top)

json_path = 'LJ_pair.top.json'
with open(json_path, 'w') as wf:
    wf.write(top_json)
