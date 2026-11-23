import numpy as np
import pytest
import torch

from boltz.colvars.cg_loaders import cg_map_from_openmscg
from boltz.colvars.cg_potential import CGBead, CGMap, CGPotential, resolve_cg_map
from boltz.data.feature.featurizerv2 import convert_atom_name

# 6 real atoms + 2 padding; tokens: A/0, A/1, B/0, B/1 (asym 0,0,1,1)
ATOMS = [(0, 0, "N"), (0, 0, "CA"), (0, 1, "CA"), (1, 0, "CA"), (1, 1, "CA"), (1, 1, "CB")]
ATOM_TOKEN = [0, 0, 1, 2, 3, 3]
N_ATOMS, N_TOK = 8, 4


def make_feats():
    names = torch.zeros(1, N_ATOMS, 4, 64)
    for i, (_, _, nm) in enumerate(ATOMS):
        for s, c in enumerate(convert_atom_name(nm)):
            names[0, i, s, c] = 1
    for i in range(len(ATOMS), N_ATOMS):
        names[0, i, :, 0] = 1
    a2t = torch.zeros(1, N_ATOMS, N_TOK)
    for i, t in enumerate(ATOM_TOKEN):
        a2t[0, i, t] = 1
    return {
        "atom_pad_mask": torch.tensor([[1.0] * len(ATOMS) + [0.0] * 2]),
        "atom_to_token": a2t,
        "asym_id": torch.tensor([[0, 0, 1, 1]]),
        "residue_index": torch.tensor([[0, 1, 0, 1]]),
        "ref_atom_name_chars": names,
    }


def cmap():
    return CGMap(
        [
            CGBead(atoms=[("A", 0, "CA"), ("A", 1, "CA")], name="b0"),
            CGBead(atoms=[("B", 0, "CA")], name="b1"),
            CGBead(atoms=[("B", 1, "CA")], name="b2"),
            CGBead(atoms=[("A", 0, "CA"), ("B", 1, "CB")], weights=[1, 3], name="b3"),
        ]
    )


CHAINS = {"A": 0, "B": 1}


def rot():
    q, _ = np.linalg.qr(np.random.RandomState(1).randn(3, 3))
    q = q * np.sign(np.linalg.det(q))
    return q.astype(np.float32)


def test_resolve_weights_and_indices():
    W = resolve_cg_map(cmap(), make_feats(), CHAINS)
    assert W.shape == (4, N_ATOMS)
    assert torch.allclose(W.sum(1), torch.ones(4))
    assert W[0, 1] == 0.5 and W[0, 2] == 0.5  # CA(A,0), CA(A,1)
    assert W[3, 1] == pytest.approx(0.25) and W[3, 5] == pytest.approx(0.75)


def test_calpha_preset_and_raw_indices():
    m = CGMap.from_calpha([[1, 2]], chain="A")
    assert resolve_cg_map(m, make_feats(), CHAINS)[0, [1, 2]].tolist() == [0.5, 0.5]
    m = CGMap.from_indices([[1, 2], [3]])
    assert resolve_cg_map(m, make_feats())[1, 3] == 1.0


@pytest.mark.parametrize(
    "bad",
    [
        CGBead(atoms=[("A", 0, "CZ")]),  # no such atom
        CGBead(atoms=[("Q", 0, "CA")]),  # unknown chain
        CGBead(indices=[7]),  # padding atom
        CGBead(indices=[99]),  # out of range
    ],
)
def test_resolve_fails_loudly(bad):
    with pytest.raises(ValueError):
        resolve_cg_map(CGMap([bad]), make_feats(), CHAINS)


def _pot(target, **kw):
    return CGPotential(cmap(), target, chain_ids=CHAINS, parameters={"fk_lambda": 4.0}, **kw)


def _target_from(coords, R, t):
    W = resolve_cg_map(cmap(), make_feats(), CHAINS)
    beads = (W @ coords).numpy()
    return beads @ R.T + t


def test_energy_zero_under_rigid_motion_and_nonzero_for_mirror():
    torch.manual_seed(0)
    x = torch.randn(N_ATOMS, 3) * 5
    feats = make_feats()
    R, t = rot(), np.array([3.0, -2.0, 7.0], dtype=np.float32)
    pot = _pot(_target_from(x, R, t))
    e = pot.compute(x.unsqueeze(0), feats, {"fk_lambda": 4.0})
    assert e.shape == (1,) and e.item() < 1e-6
    mirror = _target_from(x, R, t) * np.array([-1, 1, 1], dtype=np.float32)
    e_m = _pot(mirror).compute(x.unsqueeze(0), feats, {"fk_lambda": 4.0})
    assert e_m.item() > 1e-3  # proper rotations only: chirality is respected


def test_gaussian_value_and_sigma_scaling():
    torch.manual_seed(0)
    x = torch.randn(N_ATOMS, 3) * 5
    feats = make_feats()
    target = _target_from(x, np.eye(3, dtype=np.float32), np.zeros(3, np.float32))
    target = target + np.random.RandomState(2).randn(*target.shape).astype(np.float32)
    e1 = _pot(target, sigma=1.0).compute(x[None], feats, {"fk_lambda": 1.0})
    e2 = _pot(target, sigma=2.0).compute(x[None], feats, {"fk_lambda": 1.0})
    assert e1.item() == pytest.approx(4 * e2.item(), rel=1e-4)  # k = 1/sigma^2
    big = _pot(target, sigma=1.0, allowed=1e3).compute(x[None], feats, {"fk_lambda": 1.0})
    assert big.item() == 0.0  # inside the flat bottom


def test_gradient_finite_and_descends():
    torch.manual_seed(0)
    x = torch.randn(3, N_ATOMS, 3) * 5
    feats = make_feats()
    target = _target_from(x[0], rot(), np.zeros(3, np.float32))
    pot = _pot(target)
    p = {"fk_lambda": 4.0}
    g = pot.compute_gradient(x, feats, p)
    assert g.shape == x.shape and torch.isfinite(g).all()
    assert g[:, 6:].abs().max() == 0  # padding atoms never move
    e0 = pot.compute(x, feats, p)
    e1 = pot.compute(x - 0.05 * g, feats, p)
    assert (e1[1:] < e0[1:]).all()


def test_one_potential_per_input_and_bead_rmsd():
    x = torch.randn(1, N_ATOMS, 3) * 5
    feats = make_feats()
    target = _target_from(x[0], rot(), np.zeros(3, np.float32))
    pot = _pot(target)
    assert pot.bead_rmsd(x, feats).item() < 1e-3
    other = make_feats()
    other["atom_pad_mask"] = torch.ones(1, N_ATOMS + 2)
    with pytest.raises(ValueError):
        pot.compute_args(other)


def test_constructor_validation():
    with pytest.raises(ValueError):
        _pot(np.zeros((3, 3)))  # wrong bead count
    with pytest.raises(ValueError):
        _pot(np.zeros((4, 3)), sigma=0.0)
    with pytest.raises(ValueError):
        CGBead(atoms=[("A", 0, "CA")], weights=[-1.0])


PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  CA  GLY A   2       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  CA  ALA B   1       3.000   0.000   0.000  1.00  0.00           C
END
"""

YAML = """\
site-types:
  P0: {index: [1, 2], x-weight: [12.011, 12.011], f-weight: [1.0, 1.0]}
  P1: {index: [0], x-weight: [14.0], f-weight: [1.0]}
system:
  - anchor: 0
    repeat: 1
    offset: 4
    sites: [[P0, 0], [P1, 3]]
"""


def test_openmscg_loader(tmp_path):
    (tmp_path / "a.pdb").write_text(PDB)
    (tmp_path / "m.yaml").write_text(YAML)
    m = cg_map_from_openmscg(str(tmp_path / "m.yaml"), str(tmp_path / "a.pdb"))
    assert [b.atoms for b in m.beads] == [
        [("A", 0, "CA"), ("A", 1, "CA")],
        [("B", 0, "CA")],  # index 0 + site shift 3 -> atom 3
    ]
    assert m.beads[0].weights == [12.011, 12.011]
    ca = cg_map_from_openmscg(str(tmp_path / "m.yaml"), str(tmp_path / "a.pdb"), calpha_only=True)
    assert len(ca.beads) == 2
    (tmp_path / "r.yaml").write_text(YAML.replace("repeat: 1", "repeat: 5"))
    with pytest.raises(NotImplementedError):
        cg_map_from_openmscg(str(tmp_path / "r.yaml"), str(tmp_path / "a.pdb"))


# ---------------------------------------------------------------- cg_sample ---
from boltz.colvars.cg_sample import (  # noqa: E402
    CGSampler,
    _check_chains,
    bead_rmsd_from_structure,
)

PDB3 = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       5.000   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ALA B   1       0.000   7.000   0.000  1.00  0.00           C
ATOM      4  CA  ALA B   2       0.000   0.000   9.000  1.00  0.00           C
END
"""
BEADS3 = CGMap(
    [
        CGBead(atoms=[("A", 0, "CA")]),
        CGBead(atoms=[("A", 1, "CA")]),
        CGBead(atoms=[("B", 0, "CA")]),
        CGBead(atoms=[("B", 1, "CA")]),
    ]
)


def test_bead_rmsd_from_structure(tmp_path):
    p = tmp_path / "s.pdb"
    p.write_text(PDB3)
    pts = np.array([[0, 0, 0], [5, 0, 0], [0, 7, 0], [0, 0, 9]], dtype=float)
    moved = pts @ rot().T.astype(float) + np.array([10.0, -3.0, 4.0])
    assert bead_rmsd_from_structure(BEADS3, moved, p) < 1e-4  # rigid motion only
    noisy = pts + 1.0
    noisy[0] += [2.0, 0, 0]
    assert bead_rmsd_from_structure(BEADS3, noisy, p) > 0.1
    assert bead_rmsd_from_structure(CGMap.from_indices([[0], [1], [2], [3]]), pts, p) is None
    bad = CGMap([CGBead(atoms=[("A", 0, "CB")])] * 4)
    with pytest.raises(ValueError):
        bead_rmsd_from_structure(bad, pts, p)


def test_check_chains_fails_early():
    _check_chains(BEADS3, {"A": 0, "B": 1})
    with pytest.raises(ValueError, match="not in the Boltz input"):
        _check_chains(BEADS3, {"A": 0})


def test_sampler_validates_before_any_heavy_work(tmp_path):
    s = CGSampler(cache=str(tmp_path))
    assert s.diffusion_defaults["step_scale"] == 1.0 and s.diffusion_defaults["gamma_0"] == 0.05
    with pytest.raises(ValueError):
        CGSampler(cache=str(tmp_path), diffusion={"not_a_knob": 1})
    with pytest.raises(ValueError, match="schedules"):
        s.sample("x.yaml", tmp_path, BEADS3, np.zeros((4, 3)), schedules={"bogus": 1})
    with pytest.raises(ValueError, match="diffusion"):
        s.sample("x.yaml", tmp_path, BEADS3, np.zeros((4, 3)), diffusion={"bogus": 1})
    with pytest.raises(ValueError, match="target must be"):
        s.sample("x.yaml", tmp_path, BEADS3, np.zeros((3, 3)))
    assert s._model is None  # nothing was loaded
