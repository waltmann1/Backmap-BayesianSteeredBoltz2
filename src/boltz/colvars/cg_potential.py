"""Coarse-grained (CG) bead likelihood for Bayesian-steered Boltz-2 sampling.

A CG model is described by a :class:`CGMap` (beads = weighted atom sets). The
likelihood of the target bead positions given an all-atom sample ``x`` is a
Gaussian on each bead's position (after an optimal rigid-body alignment of the
target onto the sample):

    -ln p = sum_i  0.5 * ( max(0, |m_i(x) - R_i| - allowed) / sigma_i )**2

with ``m_i(x)`` the weighted average of bead i's atoms and ``R_i`` the target.
For ``allowed = 0`` this is a plain harmonic restraint with k = 1 / sigma**2.
There is no separate weight: the energy is exactly -ln p, and the potential
divides by ``fk_lambda`` (inherited from ``BayesianPotential``) so Boltz's
``fk_lambda`` scaling cancels.

Nothing in this module is imported by stock Boltz, so it cannot affect normal
behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch

from boltz.colvars.colvars import BayesianPotential
from boltz.model.loss.diffusionv2 import weighted_rigid_align

# (chain, residue position within chain (0-based, Boltz ``res_idx``), atom name)
# ``chain`` is a chain name (needs ``chain_ids``) or an integer Boltz asym_id.
AtomRef = tuple


@dataclass
class CGBead:
    """One CG bead: a weighted set of atoms.

    Give either ``atoms`` (identities, resolved against the Boltz input) or
    ``indices`` (raw Boltz atom indices). ``weights`` default to equal; the
    bead position is the weighted average of the atoms.
    """

    atoms: Optional[list] = None
    indices: Optional[list] = None
    weights: Optional[list] = None
    name: str = ""

    def __post_init__(self):
        if (self.atoms is None) == (self.indices is None):
            raise ValueError(f"bead {self.name!r}: give exactly one of atoms/indices")
        members = self.atoms if self.atoms is not None else self.indices
        if len(members) == 0:
            raise ValueError(f"bead {self.name!r} has no atoms")
        if self.weights is not None:
            if len(self.weights) != len(members):
                raise ValueError(
                    f"bead {self.name!r}: {len(self.weights)} weights for "
                    f"{len(members)} atoms"
                )
            w = np.asarray(self.weights, dtype=float)
            if not np.all(np.isfinite(w)) or np.any(w <= 0) or w.sum() <= 0:
                raise ValueError(f"bead {self.name!r}: weights must be positive")


@dataclass
class CGMap:
    """A list of beads. Independent of Boltz's atom ordering (unless raw
    ``indices`` are used)."""

    beads: list

    def __len__(self):
        return len(self.beads)

    @classmethod
    def from_indices(cls, index_lists: Sequence, weights: Optional[Sequence] = None):
        """Beads from raw Boltz atom indices (escape hatch)."""
        weights = weights if weights is not None else [None] * len(index_lists)
        return cls(
            [
                CGBead(indices=list(ix), weights=None if w is None else list(w))
                for ix, w in zip(index_lists, weights)
            ]
        )

    @classmethod
    def from_calpha(
        cls,
        residue_lists: Sequence,
        chain: Union[str, int] = "A",
        one_based: bool = True,
        atom_name: str = "CA",
    ):
        """Cα preset: ``[[1, 2, 9], [3, 4, 5, 6], ...]`` -> one bead per list,
        equal weights, atoms ``atom_name`` of those residues in ``chain``
        (Chroma-style residue lists)."""
        shift = 1 if one_based else 0
        return cls(
            [
                CGBead(
                    atoms=[(chain, int(r) - shift, atom_name) for r in res],
                    name=f"CG{i + 1}",
                )
                for i, res in enumerate(residue_lists)
            ]
        )


def _decode_atom_names(onehot_chars: torch.Tensor) -> list:
    """Invert ``featurizerv2.convert_atom_name`` (chr(code + 32), 4 slots)."""
    codes = (onehot_chars.argmax(dim=-1) + 32).cpu().numpy()
    return ["".join(chr(c) for c in row).strip() for row in codes]


def resolve_cg_map(
    cg_map: CGMap,
    feats: dict,
    chain_ids: Optional[dict] = None,
) -> torch.Tensor:
    """Resolve a CGMap to a dense ``(n_beads, n_atoms)`` row-normalised weight
    matrix in Boltz's (padded) atom order, using the model ``feats``.

    Fails loudly on unknown / ambiguous / padding atoms.
    """
    pad_mask = feats["atom_pad_mask"][0].bool().cpu().numpy()
    n_atoms = pad_mask.shape[0]
    needs_identity = any(b.atoms is not None for b in cg_map.beads)

    table = {}
    if needs_identity:
        tok = feats["atom_to_token"][0].argmax(dim=-1).cpu()
        asym = feats["asym_id"][0][tok].cpu().numpy()
        res = feats["residue_index"][0][tok].cpu().numpy()
        names = _decode_atom_names(feats["ref_atom_name_chars"][0])
        for i in range(n_atoms):
            if not pad_mask[i]:
                continue
            key = (int(asym[i]), int(res[i]), names[i])
            table[key] = -1 if key in table else i  # -1 marks ambiguous

    W = torch.zeros(len(cg_map), n_atoms, dtype=torch.float64)
    problems = []
    for k, bead in enumerate(cg_map.beads):
        label = bead.name or f"bead {k}"
        n_problems_before = len(problems)
        if bead.indices is not None:
            idx = [int(i) for i in bead.indices]
            for i in idx:
                if not (0 <= i < n_atoms) or not pad_mask[i]:
                    problems.append(f"{label}: atom index {i} is out of range/padding")
        else:
            idx = []
            for chain, res_idx, name in bead.atoms:
                if isinstance(chain, str):
                    if chain_ids is None or chain not in chain_ids:
                        problems.append(f"{label}: unknown chain {chain!r}")
                        continue
                    asym_id = int(chain_ids[chain])
                else:
                    asym_id = int(chain)
                found = table.get((asym_id, int(res_idx), name))
                if found is None:
                    problems.append(f"{label}: no atom {chain}/{res_idx}/{name}")
                elif found == -1:
                    problems.append(f"{label}: ambiguous atom {chain}/{res_idx}/{name}")
                else:
                    idx.append(found)
        if len(problems) > n_problems_before:
            continue  # this bead is invalid; problems already recorded
        w = (
            np.ones(len(idx))
            if bead.weights is None
            else np.asarray(bead.weights, dtype=float)
        )
        for i, wi in zip(idx, w):
            W[k, i] += float(wi)
    if problems:
        shown = "\n  ".join(problems[:15])
        more = f"\n  ... and {len(problems) - 15} more" if len(problems) > 15 else ""
        raise ValueError(f"Could not resolve the CG map:\n  {shown}{more}")
    return (W / W.sum(dim=1, keepdim=True)).float()


class CGPotential(BayesianPotential):
    """Gaussian / harmonic CG bead likelihood. One instance per Boltz input.

    Args:
        cg_map: :class:`CGMap` of beads.
        target: ``(n_beads, 3)`` target bead coordinates (Angstrom).
        sigma: Gaussian width in Angstrom *per Cartesian component*; a float
            or one value per bead. Default 1.0.
        allowed: flat-bottom tolerance in Angstrom (0 = plain harmonic).
        chain_ids: ``{chain name: Boltz asym_id}`` for beads that name chains.
        align_subset: optional bool mask over beads used for the rigid-body fit
            (default: all beads).
        parameters: the usual potential parameters (schedules, ``fk_lambda``,
            ...). ``cg_sample`` builds these; see ``cg_potential_parameters``.
    """

    def __init__(
        self,
        cg_map: CGMap,
        target,
        sigma: Union[float, Sequence[float]] = 1.0,
        allowed: float = 0.0,
        chain_ids: Optional[dict] = None,
        align_subset=None,
        parameters: Optional[dict] = None,
    ):
        super().__init__(parameters)
        n = len(cg_map)
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (n, 3):
            raise ValueError(f"target must be ({n}, 3), got {target.shape}")
        sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=np.float32), (n,)).copy()
        if not np.all(sigma_arr > 0):
            raise ValueError("sigma must be > 0")
        if allowed < 0:
            raise ValueError("allowed must be >= 0")
        if align_subset is None:
            align_subset = np.ones(n, dtype=bool)
        align_subset = np.asarray(align_subset, dtype=bool)
        if align_subset.shape != (n,) or align_subset.sum() < 3:
            raise ValueError("align_subset must be a bool mask over beads (>=3 True)")

        self.cg_map = cg_map
        self.chain_ids = chain_ids
        self.allowed = float(allowed)
        self._target = torch.from_numpy(target)
        self._sigma = torch.from_numpy(sigma_arr)
        self._align = torch.from_numpy(align_subset)
        self._W = None  # resolved lazily from feats on first use
        self._n_atoms = None
        self._dev_cache = {}

    def compute_args(self, feats, *args, **kwargs):
        if self._W is None:
            self._W = resolve_cg_map(self.cg_map, feats, self.chain_ids)
            self._n_atoms = self._W.shape[1]
        elif feats["atom_pad_mask"].shape[-1] != self._n_atoms:
            raise ValueError(
                "CGPotential was resolved for a different input; use one "
                "CGPotential per Boltz input."
            )
        return (self._W,), None, None, None, None

    def _on(self, device):
        key = str(device)
        if key not in self._dev_cache:
            self._dev_cache[key] = (
                self._W.to(device),
                self._target.to(device),
                self._sigma.to(device),
                self._align.to(device),
            )
        return self._dev_cache[key]

    def compute_variable(self, coords, index, compute_gradient=False):
        """Bead residuals ``(batch, n_beads, 3)``: sample beads minus the
        target rigidly aligned onto them (rotation held constant)."""
        W, target, _, align = self._on(coords.device)
        with torch.autocast(device_type=coords.device.type, enabled=False):
            pred = torch.einsum("kn,bnd->bkd", W, coords.float())
            batch = pred.shape[0]
            with torch.no_grad():
                t = target.unsqueeze(0).expand(batch, -1, -1)
                mask = align.float().unsqueeze(0).expand(batch, -1)
                aligned = weighted_rigid_align(t, pred.detach(), torch.ones_like(mask), mask)
            return pred - aligned

    def compute_function(self, value, *args, negation_mask=None, compute_derivative=False):
        sigma = self._sigma.to(value.device)
        d2 = (value**2).sum(dim=-1)  # (batch, n_beads)
        if self.allowed > 0:
            d = torch.sqrt(d2 + 1e-12)
            excess = torch.clamp(d - self.allowed, min=0.0)
            e = 0.5 * (excess / sigma) ** 2
        else:
            e = 0.5 * d2 / sigma**2
        return e.sum(dim=-1)  # (batch,)

    def bead_rmsd(self, coords, feats) -> torch.Tensor:
        """Per-sample bead RMSD (Angstrom) after alignment; for reporting."""
        index = self.compute_args(feats)[0]
        with torch.no_grad():
            r = self.compute_variable(coords, index)
        return torch.sqrt((r**2).sum(dim=-1).mean(dim=-1))


def cg_potential_parameters(steering_args: dict, **schedule_overrides) -> dict:
    """Default ``parameters`` for a CG potential: the fork's FKPF schedules
    merged with ``steering_args`` (which supplies ``fk_lambda``). Pass
    ``guidance_weight=...``, ``resampling_weight=...``, ``guidance_interval=...``
    to override the schedules."""
    from boltz.colvars import colvars

    params = colvars.fkpf_options(steering_args) | dict(steering_args)
    params.update(schedule_overrides)
    return params
