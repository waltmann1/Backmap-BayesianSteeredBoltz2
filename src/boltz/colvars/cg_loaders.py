"""Loaders that turn external CG-map formats into a :class:`CGMap`.

Kept separate from ``CGPotential`` / ``cg_sample`` on purpose: those only see
an in-memory ``CGMap``; file formats live here.
"""

from __future__ import annotations

from typing import Optional

import gemmi
import yaml

from boltz.colvars.cg_potential import CGBead, CGMap


def read_openmscg_yaml(path: str) -> list:
    """Parse an OpenMSCG/mstool map YAML into ``[(indices, x_weights), ...]``
    (one per site of ``system[0]``, 0-based indices into the all-atom file).

    Mirrors MSBack's reader: the first ``system`` entry, each site's shift
    added to the type's ``index``. ``x-weight`` (atomic masses) is kept as the
    bead weights; ``f-weight`` is ignored. ``repeat``/``offset`` other than a
    single copy are rejected rather than silently ignored.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    system = data["system"][0]
    if int(system.get("repeat", 1)) != 1:
        raise NotImplementedError(
            "OpenMSCG 'repeat' != 1 is not supported yet (multi-copy maps)."
        )
    sites = []
    for site_type, shift in system["sites"]:
        spec = data["site-types"][site_type]
        idx = [int(i) + int(shift) for i in spec["index"]]
        w = spec.get("x-weight")
        sites.append((site_type, idx, None if w is None else [float(x) for x in w]))
    return sites


def _aa_atom_table(pdb_path: str) -> list:
    """Flat list of ``(chain, residue position in chain (0-based), atom name)``
    in file order (hydrogens and altlocs kept, so indices match the file)."""
    st = gemmi.read_structure(pdb_path)
    table = []
    for chain in st[0]:
        for pos, residue in enumerate(chain):
            for atom in residue:
                table.append((chain.name, pos, atom.name.strip()))
    return table


def cg_map_from_openmscg(
    yaml_path: str,
    aa_pdb: str,
    chain_names: Optional[dict] = None,
    calpha_only: bool = False,
    use_weights: bool = True,
) -> CGMap:
    """Build a :class:`CGMap` from an OpenMSCG map YAML and the all-atom PDB it
    indexes.

    Args:
        yaml_path: map YAML (atom indices refer to ``aa_pdb`` order).
        aa_pdb: the all-atom PDB those indices refer to.
        chain_names: optional ``{pdb chain name: boltz chain name}``.
        calpha_only: keep only ``CA`` atoms in every bead (MSBack's
            ``alpha_c_only_map``).
        use_weights: use the YAML ``x-weight`` (masses); else equal weights.
    """
    table = _aa_atom_table(aa_pdb)
    beads = []
    for site_type, idx, weights in read_openmscg_yaml(yaml_path):
        atoms, w = [], []
        for n, i in enumerate(idx):
            if not 0 <= i < len(table):
                raise ValueError(
                    f"{site_type}: atom index {i} outside the PDB ({len(table)} atoms)"
                )
            chain, pos, name = table[i]
            if calpha_only and name != "CA":
                continue
            chain = (chain_names or {}).get(chain, chain)
            atoms.append((chain, pos, name))
            if weights is not None:
                w.append(weights[n])
        if not atoms:
            raise ValueError(f"{site_type}: no atoms left (calpha_only?)")
        beads.append(
            CGBead(
                atoms=atoms,
                weights=w if (use_weights and weights is not None) else None,
                name=site_type,
            )
        )
    return CGMap(beads)


def boltz_input_from_pdb(pdb_path: str, msa_empty: bool = False) -> dict:
    """Boltz input dict (protein chains only) from the sequences in a PDB.

    Chains with identical sequences share one entity (``id: [A, B]``). Chain
    names are kept, so they match the names ``cg_map_from_openmscg`` produces.
    ``msa_empty=True`` sets ``msa: empty`` (single-sequence, no MSA server).
    """
    st = gemmi.read_structure(pdb_path)
    entities: dict = {}
    for chain in st[0]:
        seq = gemmi.one_letter_code([r.name for r in chain])
        if "X" in seq or not seq:
            raise ValueError(
                f"chain {chain.name}: non-standard or non-protein residues; "
                "write the Boltz input yourself"
            )
        entities.setdefault(seq, []).append(chain.name)
    sequences = []
    for seq, ids in entities.items():
        prot = {"id": ids if len(ids) > 1 else ids[0], "sequence": seq}
        if msa_empty:
            prot["msa"] = "empty"
        sequences.append({"protein": prot})
    return {"version": 1, "sequences": sequences}
