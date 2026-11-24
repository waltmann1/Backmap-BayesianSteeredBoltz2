"""``cg_sample``: Bayesian-steered Boltz-2 sampling conditioned on CG beads.

Pure Python entry point (no CLI, no map files). It mirrors ``boltz.main.predict``
for the Boltz-2 path, with two differences: the potential list is composed here
(``CGPotential`` plus any extras you pass), and the loaded model is kept between
calls so repeated sampling does not reload the checkpoint.

Defaults follow the CLI's Boltz-2 behaviour in this fork (``step_scale=1.0``,
``gamma_0=0.05``), *not* the dataclass defaults.

Usage::

    sampler = CGSampler(cache="/path/to/cache")
    res = sampler.sample("input.yaml", "out", cg_map, target, sigma=1.0)
    res.structure_paths, res.bead_rmsd

Importing this module does not alter stock Boltz behaviour.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import gemmi
import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from rdkit import Chem

from boltz.colvars.cg_fixed import FixedAtoms, apply_transform  # noqa: F401  (apply_transform re-exported)
from boltz.colvars.cg_potential import CGMap, CGPotential, cg_potential_parameters
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.main import (
    BoltzProcessedInput,
    BoltzSteeringParams,
    Boltz2DiffusionParams,
    MSAModuleArgs,
    PairformerArgsV2,
    check_inputs,
    download_boltz2,
    filter_inputs_structure,
    process_inputs,
)
from boltz.model.models.boltz2 import Boltz2

# AtomDiffusion attributes that are read at sampling time and may be changed
# per call without reloading the model.
DIFFUSION_KNOBS = (
    "gamma_0",
    "gamma_min",
    "noise_scale",
    "step_scale",
    "rho",
    "sigma_min",
    "sigma_max",
    "sigma_data",
)
SCHEDULE_KNOBS = ("guidance_weight", "resampling_weight", "guidance_interval")


@dataclass
class CGSampleResult:
    """Outputs of one ``sample`` call (one entry per diffusion sample)."""

    out_dir: Path
    structure_paths: list = field(default_factory=list)
    confidence: list = field(default_factory=list)
    bead_rmsd: list = field(default_factory=list)  # Angstrom, after alignment
    # with ``fixed``: per structure, the rigid motion taking the structure's fixed atoms onto their known
    # positions ({"rotation", "center_from", "center_to", "rmsd"}); ``apply_transform(xyz, tf)`` puts sampled
    # coordinates into the known frame
    fixed_transform: list = field(default_factory=list)


def _kabsch_rmsd(fixed: np.ndarray, moving: np.ndarray) -> float:
    """RMSD of ``moving`` rigidly (proper rotation) fitted onto ``fixed``."""
    fc, mc = fixed - fixed.mean(0), moving - moving.mean(0)
    u, _, vt = np.linalg.svd(mc.T @ fc)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((fc - mc @ r) ** 2).sum(1).mean()))


def bead_rmsd_from_structure(cg_map: CGMap, target, structure_path) -> Optional[float]:
    """Bead RMSD (Angstrom, rigidly aligned) of a written structure against the
    target, resolving beads by (chain name, residue position, atom name).

    Returns ``None`` if any bead uses raw atom indices (identity unknown).
    Raises ``ValueError`` if an atom is missing from the structure.
    """
    if any(b.atoms is None for b in cg_map.beads):
        return None
    st = gemmi.read_structure(str(structure_path))
    pos = {}
    for chain in st[0]:
        for i, res in enumerate(chain):
            for atom in res:
                pos[(chain.name, i, atom.name.strip())] = np.array(
                    [atom.pos.x, atom.pos.y, atom.pos.z]
                )
    beads = []
    for k, b in enumerate(cg_map.beads):
        missing = [a for a in b.atoms if a not in pos]
        if missing:
            raise ValueError(f"bead {b.name or k}: atoms not in structure: {missing[:3]}")
        w = np.ones(len(b.atoms)) if b.weights is None else np.asarray(b.weights, float)
        beads.append((w[:, None] * np.array([pos[a] for a in b.atoms])).sum(0) / w.sum())
    return _kabsch_rmsd(np.asarray(beads), np.asarray(target, dtype=float))


def _check_chains(cg_map: CGMap, chain_ids: dict) -> None:
    """Fail before any GPU work if the map names chains the input lacks."""
    used = {c for b in cg_map.beads if b.atoms for (c, _, _) in b.atoms if isinstance(c, str)}
    unknown = sorted(used - set(chain_ids))
    if unknown:
        raise ValueError(
            f"CG map uses chains {unknown} that are not in the Boltz input "
            f"(input chains: {sorted(chain_ids)})"
        )


def _check_fixed_settings(fixed: FixedAtoms, step_scale: float) -> None:
    """Warn about settings under which the fixed atoms would not end exactly on their known positions."""
    if abs(float(step_scale) - 1.0) > 1e-9:
        warnings.warn(
            f"fixed atoms with step_scale={step_scale}: the last step lands on x0 only for step_scale=1, so the "
            "fixed atoms will not end exactly at their known positions",
            UserWarning,
            stacklevel=3,
        )


def _check_fixed_chains(fixed: FixedAtoms, cg_map: CGMap, chain_ids: dict) -> None:
    """Fail on fixed chains the input lacks; warn if the CG map has no bead in them."""
    unknown = sorted(fixed.chains - set(chain_ids))
    if unknown:
        raise ValueError(
            f"fixed atoms in chains {unknown} that are not in the Boltz input (input chains: {sorted(chain_ids)})"
        )
    if not any(a[0] in fixed.chains for b in cg_map.beads if b.atoms for a in b.atoms):
        warnings.warn(
            "no CG bead lies in the fixed chains: the free chains' placement against them is then left to the "
            "model's prior. Add the fixed chains' beads (and targets) to the CG map.",
            UserWarning,
            stacklevel=3,
        )


class CGSampler:
    """Holds the Boltz-2 model between calls. One ``sample`` call = one input."""

    def __init__(
        self,
        cache: str = "~/.boltz",
        accelerator: str = "gpu",
        devices: int = 1,
        checkpoint: Optional[str] = None,
        no_kernels: bool = False,
        subsample_msa: bool = True,
        num_subsampled_msa: int = 1024,
        diffusion: Optional[dict] = None,
    ):
        # absolute: sampling runs inside the results directory (see sample())
        self.cache = Path(cache).expanduser().resolve()
        self.accelerator = accelerator
        self.devices = devices
        self.checkpoint = (
            None if checkpoint is None else str(Path(checkpoint).expanduser().resolve())
        )
        self.no_kernels = no_kernels
        self.subsample_msa = subsample_msa
        self.num_subsampled_msa = num_subsampled_msa
        # sampler-level diffusion defaults; CLI Boltz-2 defaults for these two
        self.diffusion_defaults = {"step_scale": 1.0, "gamma_0": 0.05}
        self.diffusion_defaults.update(self._check_diffusion(diffusion))
        self._model = None
        self._base_diffusion = None

    @staticmethod
    def _check_diffusion(overrides: Optional[dict]) -> dict:
        overrides = dict(overrides or {})
        unknown = set(overrides) - set(DIFFUSION_KNOBS)
        if unknown:
            raise ValueError(
                f"unknown diffusion settings {sorted(unknown)}; allowed: {DIFFUSION_KNOBS}"
            )
        return overrides

    def _load_model(self):
        diffusion_params = Boltz2DiffusionParams()
        for k, v in self.diffusion_defaults.items():
            setattr(diffusion_params, k, v)
        checkpoint = self.checkpoint or (self.cache / "boltz2_conf.ckpt")
        model = Boltz2.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args={
                "recycling_steps": 3,
                "sampling_steps": 200,
                "diffusion_samples": 1,
                "max_parallel_samples": None,
                "write_confidence_summary": True,
                "write_full_pae": False,
                "write_full_pde": False,
            },
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not self.no_kernels,
            pairformer_args=asdict(PairformerArgsV2()),
            msa_args=asdict(
                MSAModuleArgs(
                    subsample_msa=self.subsample_msa,
                    num_subsampled_msa=self.num_subsampled_msa,
                    use_paired_feature=True,
                )
            ),
            steering_args=asdict(BoltzSteeringParams()),
        )
        model.eval()
        self._model = model
        sm = model.structure_module
        self._base_diffusion = {k: getattr(sm, k) for k in DIFFUSION_KNOBS}

    def sample(
        self,
        inputs: Union[str, Path, dict],
        out_dir: Union[str, Path],
        cg_map: CGMap,
        target,
        *,
        name: str = "cg_input",
        sigma: Union[float, Sequence[float]] = 1.0,
        allowed: float = 0.0,
        align_subset=None,
        extra_potentials: Sequence = (),
        # fixed ("halo") atoms
        fixed: Optional[FixedAtoms] = None,
        fixed_template: bool = True,
        # sampling
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        # particle steering
        num_particles: int = 3,
        fk_lambda: float = 4.0,
        fk_resampling_interval: int = 1,
        num_gd_steps: int = 1,
        schedules: Optional[dict] = None,
        # diffusion process (per call; unspecified keys use the sampler defaults)
        diffusion: Optional[dict] = None,
        # input processing / output
        use_msa_server: bool = False,
        msa_server_url: str = "https://api.colabfold.com",
        msa_pairing_strategy: str = "greedy",
        output_format: str = "pdb",
        num_workers: int = 2,
        seed: Optional[int] = None,
        override: bool = True,
    ) -> CGSampleResult:
        """Sample all-atom structures restrained to CG bead targets.

        ``inputs`` is a Boltz input YAML path, or a dict that is written to
        ``<out_dir>/<name>.yaml``. ``use_msa_server`` defaults to False so no
        sequence leaves the machine unless you ask for it.

        ``extra_potentials`` are additional ``Potential`` objects (e.g. upstream
        Boltz potentials you construct yourself); ``CGPotential`` is always last.
        ``schedules`` may override ``guidance_weight``, ``resampling_weight``
        and ``guidance_interval`` (a schedule object or a number).

        ``fixed`` (a :class:`~boltz.colvars.cg_fixed.FixedAtoms`) holds those atoms exactly at their known
        positions while the other chains are sampled; the CG map should include the fixed chains' beads. With
        ``fixed_template=True`` (default) the fixed atoms are also given to the trunk as a non-forced template,
        which needs ``inputs`` as a dict. See ``boltz.colvars.cg_fixed``. Physical potentials (e.g. Boltz's
        VDWOverlap) are not used unless passed in ``extra_potentials``; see the README for what we learned.
        """
        if schedules:
            bad = set(schedules) - set(SCHEDULE_KNOBS)
            if bad:
                raise ValueError(f"unknown schedules {sorted(bad)}; allowed: {SCHEDULE_KNOBS}")
        diffusion = self._check_diffusion(diffusion)
        if fixed is not None:
            if fixed_template and not isinstance(inputs, dict):
                raise ValueError(
                    "fixed_template=True needs `inputs` as a dict (the template is added to it); "
                    "pass a dict or set fixed_template=False"
                )
            _check_fixed_settings(fixed, {**self.diffusion_defaults, **diffusion}.get("step_scale", 1.0))
        n_target = np.asarray(target).shape
        if n_target != (len(cg_map), 3):
            raise ValueError(f"target must be ({len(cg_map)}, 3), got {n_target}")

        # --- global state, as in boltz.main.predict -------------------------
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("highest")
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        if seed is not None:
            seed_everything(seed)
        for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
            os.environ[key] = os.environ.get(key, "1")

        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if fixed is not None and fixed_template:
            tpl = fixed.write_template(out_dir / f"{name}_fixed_template.pdb")
            inputs = {**inputs, "templates": [*inputs.get("templates", []), fixed.template_entry(tpl)]}
        if isinstance(inputs, dict):
            data_path = out_dir / f"{name}.yaml"
            with open(data_path, "w") as f:
                yaml.safe_dump(inputs, f, sort_keys=False)
        else:
            data_path = Path(inputs).expanduser().resolve()
        results_dir = out_dir / f"boltz_results_{data_path.stem}"
        results_dir.mkdir(parents=True, exist_ok=True)

        self.cache.mkdir(parents=True, exist_ok=True)
        download_boltz2(self.cache)
        data = check_inputs(data_path)

        mol_dir = self.cache / "mols"
        process_inputs(
            data=data,
            out_dir=results_dir,
            ccd_path=self.cache / "ccd.pkl",
            mol_dir=mol_dir,
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            msa_server_username=os.environ.get("BOLTZ_MSA_USERNAME"),
            msa_server_password=os.environ.get("BOLTZ_MSA_PASSWORD"),
            api_key_header=None,
            api_key_value=os.environ.get("MSA_API_KEY_VALUE"),
            boltz2=True,
            preprocessing_threads=1,
            max_msa_seqs=8192,
        )
        manifest = Manifest.load(results_dir / "processed" / "manifest.json")
        filtered = filter_inputs_structure(
            manifest=manifest, outdir=results_dir, override=override
        )
        if len(filtered.records) != 1:
            raise ValueError(
                f"cg_sample handles exactly one input per call; got {len(filtered.records)} "
                "(existing predictions? pass override=True)"
            )
        record = filtered.records[0]
        chain_ids = {c.chain_name: int(c.chain_id) for c in record.chains}
        _check_chains(cg_map, chain_ids)
        if fixed is not None:
            _check_fixed_chains(fixed, cg_map, chain_ids)

        processed_dir = results_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=filtered,
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            constraints_dir=(processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None,
            template_dir=(processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None,
            extra_mols_dir=(processed_dir / "mols")
            if (processed_dir / "mols").exists()
            else None,
        )

        # --- model (cached) and per-call settings -----------------------------
        if self._model is None:
            self._load_model()
        model = self._model
        sm = model.structure_module
        # base = values the model was loaded with (sampler defaults); per-call
        # overrides never persist into the next call
        for k, v in {**self._base_diffusion, **diffusion}.items():
            setattr(sm, k, v)
        model.predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        steering = asdict(BoltzSteeringParams())
        steering.update(
            fk_steering=True,
            colvars_guidance_update=True,
            physical_guidance_update=False,
            contact_guidance_update=False,
            num_diffsamples=diffusion_samples,
            num_particles=num_particles,
            fk_lambda=fk_lambda,
            fk_resampling_interval=fk_resampling_interval,
            num_gd_steps=num_gd_steps,
            no_diffusion=False,
            no_confidence=False,
            path_sequence=data_path,
            path_record=processed_dir / "records" / f"{data_path.stem}.json",
            path_structure=processed_dir / "structures" / f"{data_path.stem}.npz",
        )
        cg_pot = CGPotential(
            cg_map,
            target,
            sigma=sigma,
            allowed=allowed,
            chain_ids=chain_ids,
            align_subset=align_subset,
            parameters=cg_potential_parameters(steering, **(schedules or {})),
        )
        # CGPotential last (visdiff's write_colvar expects the Bayesian one last)
        steering["potentials"] = [*extra_potentials, cg_pot]
        model.steering_args = steering

        # --- run ----------------------------------------------------------------
        writer = BoltzWriter(
            data_dir=processed.targets_dir,
            output_dir=results_dir / "predictions",
            output_format=output_format,
            boltz2=True,
            write_embeddings=False,
        )
        trainer = Trainer(
            default_root_dir=results_dir,
            strategy="auto",
            callbacks=[writer],
            accelerator=self.accelerator,
            devices=self.devices,
            precision="bf16-mixed",
            inference_mode=False,
        )
        data_module = Boltz2InferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method=None,
        )
        if fixed is not None:
            sm.x0_hook = fixed.make_x0_hook(chain_ids)
            sm.alignment_weights = fixed.make_alignment_weights(chain_ids)
        try:
            # The sampler writes scratch tensors (atom_coords.pt, argsort.pt,
            # resample_weights.pt) into the *current directory*; keep them with
            # the results instead of cluttering wherever the caller ran from.
            # (All paths used above are already absolute.)
            with contextlib.chdir(results_dir):
                trainer.predict(model, datamodule=data_module, return_predictions=False)
        finally:
            # do not leak this call's potentials or fixed-atom hooks into the cached model
            model.steering_args = asdict(BoltzSteeringParams())
            sm.x0_hook = None
            sm.alignment_weights = None

        return self._collect(results_dir, record.id, cg_map, target, output_format, fixed)

    @staticmethod
    def _collect(results_dir, rec_id, cg_map, target, output_format, fixed=None) -> CGSampleResult:
        pred_dir = results_dir / "predictions" / rec_id
        ext = "pdb" if output_format == "pdb" else "cif"

        def rank(p):
            m = re.search(r"_model_(\d+)\.", p.name)
            return int(m.group(1)) if m else 0

        paths = sorted(pred_dir.glob(f"{rec_id}_model_*.{ext}"), key=rank)
        res = CGSampleResult(out_dir=results_dir)
        for p in paths:
            res.structure_paths.append(p)
            conf = pred_dir / f"confidence_{p.stem}.json"
            res.confidence.append(json.load(open(conf)) if conf.exists() else {})
            res.bead_rmsd.append(bead_rmsd_from_structure(cg_map, target, p))
            if fixed is not None:
                res.fixed_transform.append(fixed.transform_from_structure(p))
        return res


def cg_sample(inputs, out_dir, cg_map, target, *, cache="~/.boltz", sampler_kwargs=None, **kwargs):
    """One-shot convenience wrapper: build a ``CGSampler`` and sample once.
    For repeated calls create a ``CGSampler`` yourself so the model is reused."""
    return CGSampler(cache=cache, **(sampler_kwargs or {})).sample(
        inputs, out_dir, cg_map, target, **kwargs
    )
