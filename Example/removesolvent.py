import mdtraj as md

In [2]: straj=md.load_dcd('traj.dcd',top='seh_tppu_mdtraj.pdb')

In [3]: newtrj=straj.remove_solvent?
Signature: straj.remove_solvent(exclude=None, inplace=False)
Docstring:
Create a new trajectory without solvent atoms

Parameters
----------
exclude : array-like, dtype=str, shape=(n_solvent_types)
    List of solvent residue names to retain in the new trajectory.
inplace : bool, default=False
    The return value is either ``self``, or the new trajectory,
    depending on the value of ``inplace``.

Returns
-------
traj : md.Trajectory
    The return value is either ``self``, or the new trajectory,
    depending on the value of ``inplace``.
File:      ~/anaconda3/lib/python3.6/site-packages/mdtraj/core/trajectory.py
Type:      method

In [4]: newtrj=straj.remove_solvent()

In [5]: newtrj.save_dcd('traj_without_solvent.dcd')
