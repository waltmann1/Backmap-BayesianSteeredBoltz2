import os
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from boltz.data.types import Coords, Interface, Record, Structure, StructureV2
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb

from boltz.colvars import colvars

def exportdiffpdb(atom_coords,step_id,
                  pad_masks=None,
                  steering_args=None,
                  out_dir="trajs"):

    traj_ids = colvars.fkpf_options(steering_args)["write_diff_traj_ids"]

    path_out_dir = Path(out_dir)
    path_out_dir.mkdir(parents=True, exist_ok=True)

    coords = atom_coords.unsqueeze(0)
    records = [Record.load(steering_args["path_record"])]

    # iterate over the records
    for record, coord, pad_mask in zip(records,coords,pad_masks):
        # load the structure
        structure: StructureV2 = StructureV2.load(steering_args["path_structure"])

        # Compute chain map with masked removed, to be used later
        chain_map = {}
        for i, mask in enumerate(structure.mask):
            if mask:
                chain_map[len(chain_map)] = i

        # Remove masked chains completely
        structure = structure.remove_invalid_chains()

        for model_idx in range(coord.shape[0]):
            if model_idx not in traj_ids:
                continue
            # Get model coord
            model_coord = coord[model_idx]

            # Unpad
            coord_unpad = model_coord[pad_mask.bool()]
            coord_unpad = coord_unpad.cpu().numpy()

            # Normalize atom coordinates to be centered and periodically wrapped
            mu = np.sum(coord_unpad,axis=0)/len(coord_unpad)
            coord_unpad = np.fmod(coord_unpad - mu.reshape(1,3),999)

            # New atom table
            atoms = structure.atoms
            atoms["coords"] = coord_unpad
            atoms["is_present"] = True
            structure: StructureV2
            coord_unpad = [(x,) for x in coord_unpad]
            coord_unpad = np.array(coord_unpad, dtype=Coords)

            # Mew residue table
            residues = structure.residues
            residues["is_present"] = True

            # Update the structure
            interfaces = np.array([], dtype=Interface)
            new_structure: StructureV2 = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                        coords=coord_unpad,
                    )

            # Update chain info
            chain_info = []
            for chain in new_structure.chains:
                old_chain_idx = chain_map[chain["asym_id"]]
                old_chain_info = record.chains[old_chain_idx]
                new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                chain_info.append(new_chain_info)

            # Save the structure
            #struct_dir = self.output_dir / record.id
            #struct_dir.mkdir(exist_ok=True)

            # Create path name
            #outname = f"{record.id}_traj{model_idx}_step{step_id}"
            outname = f"diff_traj{model_idx}_step{step_id}"

            # Save the structure
            #pdbtmp = visdiffto_pdb(new_structure)
            #with open(f"trajs/diff_traj{i}_step{loopk}.pdb","w") as f:
            #    f.write(pdbtmp)           
            path = path_out_dir / f"{outname}.pdb"
            with path.open("w") as f:
                f.write(
                        to_pdb(new_structure, plddts=None, boltz2=True)
                        )
    return None

#################### OLD ########################
import pandas as pd
#################################################
# Routines to visualize a diffusion trajectory
# over the batches
#################################################
#fkpf_opt = colvars.fkpf_options()
#flagvis = fkpf_opt["flagvis"]
#flagwritecsv = fkpf_opt["flagwritecolvarcsv"]
#ntrajs = fkpf_opt["nvistrajs"]
#flagrandtrajs = fkpf_opt["flagrandvistrajs"]

def mkcolvarcsv(atom_coords,bayespotential,
                ntpts,tval,pos,
                colvarsarray=None,flagwritecsvnow = False,
                flagwritecsv=False,
                flagdenoised=False,
                out_dir="trajs"):
    path_out_dir = Path(out_dir)
    path_out_dir.mkdir(parents=True, exist_ok=True)
    
    if flagwritecsv:
        zetavals = bayespotential.compute_variable(atom_coords,None)
        if colvarsarray is None:
            colvarsarray = torch.empty(ntpts,1+zetavals.flatten().shape[0])
        colvarsarray[pos,0] = tval
        colvarsarray[pos,1:] = zetavals.flatten()

        if flagwritecsvnow:
            cvnp = colvarsarray.cpu().numpy()
            df = pd.DataFrame(cvnp)
            fpath = path_out_dir / "cvzetas_denoised.csv" if flagdenoised else path_out_dir / "cvzetas.csv"
            df.to_csv(fpath)

        return colvarsarray
    else:

        return None
