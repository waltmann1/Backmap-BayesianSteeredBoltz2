"""Fixed ("halo") atoms for CG-steered Boltz-2 sampling.

Some atoms are known and must not move (e.g. neighbouring chains in a domain
decomposition). :class:`FixedAtoms` holds their coordinates, found by identity
``(chain name, residue position in chain, atom name)``, and provides what the
sampler needs to keep them exactly where they are:

* an **x0 hook**: every x0 prediction of the fixed atoms is replaced by the known
  coordinates, rigidly superposed onto the current noisy fixed atoms (the sampler
  re-centres and randomly rotates the coordinates every step);
* **alignment weights**: the sampler's rigid re-alignment of the noisy structure
  onto x0 (``alignment_reverse_diff``, on by default) is fitted on the fixed atoms
  only. Fitted on all atoms, a mismatch of the free chain with its own prediction
  rotates the whole system and the free chain ends up mis-posed relative to the
  fixed atoms;
* an optional **template** (:meth:`write_template`, :meth:`template_entry`) so the
  trunk sees the fixed atoms' geometry (give it with ``force: false``);
* :meth:`transform_from_structure`: the rigid motion between a sampled structure
  and the known frame, to put the sampled free atoms back into that frame.

The fixed atoms end exactly at their known positions (up to that rigid motion)
when ``step_scale == 1``: the last step has no guidance and lands on x0.

Use through ``CGSampler.sample(..., fixed=FixedAtoms(...))``. The CG map should
also contain the beads of the fixed chains (they are then satisfied by the fixed
atoms) so that the free and the fixed chains share one rigid alignment in the
likelihood. Nothing here is imported by stock Boltz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from boltz.colvars.cg_potential import _decode_atom_names
from boltz.model.loss.diffusionv2 import weighted_rigid_align

AtomKey = tuple  # (chain name, residue position within chain (0-based), atom name)


def superpose(moving, target) -> dict:
    """Proper rotation + translations that best superpose ``moving`` onto ``target`` (both (n, 3)).

    Returns ``{"rotation", "center_from", "center_to", "rmsd"}``; apply with :func:`apply_transform`.
    """
    moving, target = np.asarray(moving, float), np.asarray(target, float)
    if moving.shape != target.shape or moving.ndim != 2 or moving.shape[1] != 3 or len(moving) < 3:
        raise ValueError("superpose needs two (n, 3) arrays with n >= 3")
    pc, qc = moving.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((moving - pc).T @ (target - qc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    rmsd = float(np.sqrt((((moving - pc) @ rot.T + qc - target) ** 2).sum(1).mean()))
    return {"rotation": rot, "center_from": pc, "center_to": qc, "rmsd": rmsd}


def apply_transform(coords, transform: dict) -> np.ndarray:
    """``(coords - center_from) @ rotation.T + center_to``."""
    return (np.asarray(coords, float) - transform["center_from"]) @ transform["rotation"].T + transform["center_to"]


@dataclass
class FixedAtoms:
    """Known atoms to hold fixed during sampling.

    Args:
        coords: ``{(chain, residue position, atom name): xyz}``, Angstrom, in the same frame as the CG targets.
        residue_names: ``{chain: [residue name, ...]}`` in residue order; only needed for the template.
        min_fraction: at least this fraction of the model's atoms in the fixed chains must have known
            coordinates (the rest, e.g. a missing OXT, stay free); otherwise resolution fails loudly.
    """

    coords: dict
    residue_names: Optional[dict] = None
    min_fraction: float = 0.95
    _bound: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not self.coords:
            raise ValueError("FixedAtoms needs at least one atom")
        clean = {}
        for key, xyz in self.coords.items():
            if not (isinstance(key, tuple) and len(key) == 3):
                raise ValueError(f"atom key must be (chain, residue position, atom name), got {key!r}")
            xyz = np.asarray(xyz, dtype=float)
            if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
                raise ValueError(f"atom {key}: coordinates must be 3 finite numbers")
            clean[(str(key[0]), int(key[1]), str(key[2]))] = xyz
        self.coords = clean
        if not 0 < self.min_fraction <= 1:
            raise ValueError("min_fraction must be in (0, 1]")

    @property
    def chains(self) -> set:
        return {k[0] for k in self.coords}

    @classmethod
    def from_structure(cls, path, chains, exclude_atoms=("OXT",), transform: Optional[dict] = None, **kwargs):
        """Fixed atoms from the heavy atoms of ``chains`` in a PDB/mmCIF file (hydrogens dropped).

        ``transform`` (from :func:`superpose`) moves them into another frame, e.g. a previous prediction
        into the CG target frame. ``exclude_atoms``: Boltz has no OXT atom.
        """
        import gemmi

        chains = {chains} if isinstance(chains, str) else set(chains)
        st = gemmi.read_structure(str(path))
        st.remove_hydrogens()
        coords, names = {}, {}
        for ch in st[0]:
            if ch.name not in chains:
                continue
            names[ch.name] = [r.name for r in ch]
            for pos, res in enumerate(ch):
                for at in res:
                    nm = at.name.strip()
                    if nm in exclude_atoms:
                        continue
                    xyz = np.array([at.pos.x, at.pos.y, at.pos.z])
                    coords[(ch.name, pos, nm)] = apply_transform(xyz, transform) if transform is not None else xyz
        missing = chains - set(names)
        if missing:
            raise ValueError(f"chains {sorted(missing)} not found in {path}")
        return cls(coords, residue_names=names, **kwargs)

    # ------------------------------------------------------------ resolution
    def _resolve(self, feats: dict, chain_ids: dict):
        """(idx, known): Boltz atom indices of the fixed atoms and their known positions (cached per feats)."""
        key = (id(feats), tuple(sorted(chain_ids.items())), feats["atom_pad_mask"].shape)
        hit = self._bound.get("key")
        if hit == key:
            return self._bound["idx"], self._bound["known"]
        device = feats["atom_pad_mask"].device
        unknown = self.chains - set(chain_ids)
        if unknown:
            raise ValueError(f"fixed atoms in chains {sorted(unknown)} that are not in the Boltz input")
        asym_to_name = {int(v): k for k, v in chain_ids.items()}
        pad = feats["atom_pad_mask"][0].bool().cpu().numpy()
        tok = feats["atom_to_token"][0].argmax(dim=-1).cpu()
        asym = feats["asym_id"][0][tok].cpu().numpy()
        res = feats["residue_index"][0][tok].cpu().numpy()
        names = _decode_atom_names(feats["ref_atom_name_chars"][0])
        chains = self.chains
        idx, known, n_in_chains = [], [], 0
        for i in range(len(pad)):
            if not pad[i]:
                continue
            ch = asym_to_name.get(int(asym[i]))
            if ch not in chains:
                continue
            n_in_chains += 1
            k = (ch, int(res[i]), names[i])
            if k in self.coords:
                idx.append(i)
                known.append(self.coords[k])
        if not idx or len(idx) < self.min_fraction * n_in_chains:
            raise ValueError(
                f"fixed atoms: only {len(idx)} of the {n_in_chains} atoms of chains {sorted(chains)} have known "
                f"coordinates (need {self.min_fraction:.0%}); check chain names and residue numbering"
            )
        idx_t = torch.tensor(idx, dtype=torch.long, device=device)
        known_t = torch.tensor(np.array(known), dtype=torch.float32, device=device)
        self._bound.clear()
        self._bound.update(key=key, idx=idx_t, known=known_t, n_atoms=len(pad))
        return idx_t, known_t

    # ----------------------------------------------------------------- hooks
    def make_x0_hook(self, chain_ids: dict):
        """Hook for ``AtomDiffusion.x0_hook``: fixed atoms' x0 := known coordinates, superposed on the noisy input."""

        def hook(denoised, noised, network_condition_kwargs):
            idx, known = self._resolve(network_condition_kwargs["feats"], chain_ids)
            with torch.no_grad(), torch.autocast(device_type=denoised.device.type, enabled=False):
                b = noised.shape[0]
                w = torch.ones(b, idx.numel(), device=denoised.device)
                aligned = weighted_rigid_align(
                    known.unsqueeze(0).expand(b, -1, -1), noised[:, idx].detach().float(), w, w
                )
            out = denoised.clone()
            out[:, idx] = aligned.to(out.dtype)
            return out

        return hook

    def make_alignment_weights(self, chain_ids: dict):
        """Hook for ``AtomDiffusion.alignment_weights``: 1 on the fixed atoms, 0 elsewhere."""

        def fn(feats):
            idx, _ = self._resolve(feats, chain_ids)
            w = torch.zeros(feats["atom_pad_mask"].shape[-1], device=idx.device)
            w[idx] = 1.0
            return w

        return fn

    # -------------------------------------------------------------- template
    def write_template(self, path) -> Path:
        """Write the fixed atoms as a PDB with SEQRES records (what Boltz's PDB template parser needs)."""
        import gemmi

        if not self.residue_names:
            raise ValueError("write_template needs residue_names (build with FixedAtoms.from_structure)")
        by_res: dict = {}
        for (chain, pos, atom), xyz in self.coords.items():
            by_res.setdefault((chain, pos), []).append((atom, xyz))
        st = gemmi.Structure()
        model = gemmi.Model("1")
        for chain_name in sorted(self.chains):
            chain = gemmi.Chain(chain_name)
            for pos, rname in enumerate(self.residue_names[chain_name]):
                res = gemmi.Residue()
                res.name = rname
                res.seqid = gemmi.SeqId(pos + 1, " ")
                for atom_name, xyz in by_res.get((chain_name, pos), []):
                    at = gemmi.Atom()
                    at.name = atom_name
                    at.element = gemmi.Element(atom_name[0])
                    at.pos = gemmi.Position(*[float(x) for x in xyz])
                    at.occ, at.b_iso = 1.0, 20.0
                    res.add_atom(at)
                chain.add_residue(res)
            model.add_chain(chain)
        st.add_model(model)
        st.setup_entities()
        for ent in st.entities:
            if not ent.full_sequence:
                src = next((c for c in st[0] if c.name == ent.name), None) or next(iter(st[0]))
                ent.full_sequence = [r.name for r in src]
        opt = gemmi.PdbWriteOptions()
        opt.seqres_records = True
        path = Path(path)
        st.write_pdb(str(path), opt)
        return path

    def template_entry(self, pdb_path, force: bool = False, threshold: Optional[float] = None) -> dict:
        """The ``templates:`` entry for a Boltz input YAML. Boltz renames chains of a PDB template to ``<name>1``."""
        if force and threshold is None:
            raise ValueError("force=True needs a threshold (A)")
        chains = sorted(self.chains)
        entry = {
            "pdb": str(Path(pdb_path).resolve()),
            "chain_id": chains,
            "template_id": [f"{c}1" for c in chains],
            "force": bool(force),
        }
        if threshold is not None:
            entry["threshold"] = float(threshold)
        return entry

    # ------------------------------------------------------------- transform
    def transform_from_structure(self, structure_path) -> dict:
        """Rigid motion mapping a sampled structure's fixed atoms onto their known positions
        (``apply_transform(x_sampled, tf)`` puts sampled coordinates into the known frame)."""
        import gemmi

        st = gemmi.read_structure(str(structure_path))
        found = {}
        for ch in st[0]:
            if ch.name in self.chains:
                for pos, res in enumerate(ch):
                    for at in res:
                        k = (ch.name, pos, at.name.strip())
                        if k in self.coords:
                            found[k] = np.array([at.pos.x, at.pos.y, at.pos.z])
        if len(found) < self.min_fraction * len(self.coords):
            raise ValueError(f"only {len(found)} of {len(self.coords)} fixed atoms found in {structure_path}")
        keys = sorted(found)
        return superpose(np.array([found[k] for k in keys]), np.array([self.coords[k] for k in keys]))
