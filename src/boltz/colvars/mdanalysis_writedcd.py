import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDWriter
import os

# Trajectory id and coordinate files
fpath = "trajs/"
trajid = 0
nsteps = 200

# Collect all pdb files
pdb_files = [f"diff_traj{trajid}_step{k}.pdb" for k in range(nsteps)]

# Establish the PDB topology file
u = mda.Universe(pdb_files[-1])

# Write to dcd
with DCDWriter(f"diff_traj{trajid}.dcd",n_atoms=u.atoms.n_atoms) as writer:
    for pdb in pdb_files:
        u = mda.Universe(pdb_files[-1], pdb)
        writer.write(u.atoms)
