import mdtraj as md

straj=md.load_dcd('traj.dcd',top='seh_tppu_mdtraj.pdb')

newtrj=straj.remove_solvent()

newtrj.save_dcd('traj_without_solvent.dcd')
