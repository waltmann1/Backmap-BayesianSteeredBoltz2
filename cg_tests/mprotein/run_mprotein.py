"""M-protein (homodimer) CG-restrained Boltz-2 test. Pure Python, no map-file logic in cg_sample.

  python run_mprotein.py --out out_test                 # sample (needs a GPU node)
  python run_mprotein.py --score some_structure.pdb     # bead RMSD of any structure (CPU ok)

Inputs next to this file (copied from MSBack's examples/Mprotein):
  M_long_AA.pdb   all-atom reference the YAML indexes (also gives chains/sequence)
  M_mapping.yaml  OpenMSCG map, 40 beads (CA/CB atoms, masses as weights)
  M_CG.pdb        target bead coordinates (chain X, bead k = k-th atom)
"""
import os
import argparse
import json
from pathlib import Path

import gemmi
import numpy as np

from boltz.colvars.cg_loaders import boltz_input_from_pdb, cg_map_from_openmscg
from boltz.colvars.cg_sample import CGSampler, bead_rmsd_from_structure


def geometry_report(path):
    """One-line sanity check: coordinate extent and CA-CA distances (expect ~3.8 A)."""
    st = gemmi.read_structure(str(path))
    xyz, bad, tot, med = [], 0, 0, []
    for chain in st[0]:
        ca = [(a.pos.x, a.pos.y, a.pos.z) for r in chain for a in r if a.name.strip() == "CA"]
        xyz += [(a.pos.x, a.pos.y, a.pos.z) for r in chain for a in r]
        d = np.linalg.norm(np.diff(np.array(ca), axis=0), axis=1)
        bad += int((np.abs(d - 3.8) > 0.5).sum()); tot += len(d); med.append(float(np.median(d)))
    x = np.array(xyz)
    flag = "OK" if (np.abs(x).max() < 500 and bad / tot < 0.05) else "*** BROKEN STRUCTURE ***"
    return (f"geometry: max |coord| {np.abs(x).max():.0f} A, CA-CA median {np.median(med):.2f} A, "
            f"{100 * bad / tot:.1f}% of CA-CA off by >0.5 A  [{flag}]")

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out_mprotein")
ap.add_argument("--score", help="only score this structure against the CG target")
ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian width, A per component")
ap.add_argument("--allowed", type=float, default=0.0, help="flat-bottom tolerance, A")
ap.add_argument("--particles", type=int, default=3)
ap.add_argument("--samples", type=int, default=1)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--recycling", type=int, default=3)
ap.add_argument("--max-parallel", type=int, default=None,
                help="max trajectories through the network at once (lower it if the GPU runs out of memory)")
ap.add_argument("--fk-lambda", type=float, default=4.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--msa-server", action="store_true",
                help="use the public ColabFold MSA server (sends the sequence off-site); default: no MSA")
ap.add_argument("--cache", default=os.environ.get("BOLTZ_CACHE"))
a = ap.parse_args()

# ---- map + target (file formats live in the loaders, not in cg_sample) ----
cg_map = cg_map_from_openmscg(str(HERE / "M_mapping.yaml"), str(HERE / "M_long_AA.pdb"))
cg_atoms = [(r.name, a_.pos) for r in gemmi.read_structure(str(HERE / "M_CG.pdb"))[0][0] for a_ in r]
assert [n for n, _ in cg_atoms] == [b.name for b in cg_map.beads], "bead order != M_CG.pdb order"
target = np.array([[p.x, p.y, p.z] for _, p in cg_atoms])

ref = bead_rmsd_from_structure(cg_map, target, HERE / "M_long_AA.pdb")
print(f"reference: M_long_AA.pdb beads vs target = {ref:.2f} A (different conformation)")

if a.score:
    print(f"{a.score}: bead RMSD = {bead_rmsd_from_structure(cg_map, target, a.score):.2f} A")
    print(geometry_report(a.score))
    raise SystemExit

inputs = boltz_input_from_pdb(str(HERE / "M_long_AA.pdb"), msa_empty=not a.msa_server)
sampler = CGSampler(cache=a.cache)
res = sampler.sample(
    inputs, a.out, cg_map, target, name="M",
    sigma=a.sigma, allowed=a.allowed,
    num_particles=a.particles, diffusion_samples=a.samples, sampling_steps=a.steps,
    recycling_steps=a.recycling, fk_lambda=a.fk_lambda, max_parallel_samples=a.max_parallel,
    use_msa_server=a.msa_server, seed=a.seed,
)
for p, rmsd, conf in zip(res.structure_paths, res.bead_rmsd, res.confidence):
    print(f"{p.name}: bead RMSD {rmsd:.2f} A; confidence {conf.get('confidence_score')}")
    print("  " + geometry_report(p))
json.dump({"sigma": a.sigma, "allowed": a.allowed, "particles": a.particles,
           "paths": [str(p) for p in res.structure_paths], "bead_rmsd": res.bead_rmsd,
           "confidence": res.confidence}, open(Path(a.out) / "summary.json", "w"), indent=1)
