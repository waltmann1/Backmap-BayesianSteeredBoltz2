import numpy as np
import pytest
import torch

from boltz.colvars.cg_fixed import FixedAtoms, apply_transform, superpose
from boltz.colvars.cg_potential import CGBead, CGMap
from boltz.colvars.cg_sample import CGSampler, _check_fixed_chains, _check_fixed_settings
from boltz.model.loss.diffusionv2 import weighted_rigid_align
from boltz.model.modules.diffusionv2 import AtomDiffusion
from test_cg_potential import CHAINS, N_ATOMS, make_feats

# chain B owns Boltz atoms 3, 4, 5 in make_feats: (B,0,CA) (B,1,CA) (B,1,CB)
HALO = np.array([3, 4, 5])


def rng_xyz(seed, n=3, scale=8.0):
    return np.random.RandomState(seed).randn(n, 3) * scale


def fixed_b(min_fraction=0.95):
    xyz = rng_xyz(0)
    coords = {("B", 0, "CA"): xyz[0], ("B", 1, "CA"): xyz[1], ("B", 1, "CB"): xyz[2]}
    return FixedAtoms(coords, residue_names={"B": ["ALA", "GLY"]}, min_fraction=min_fraction)


def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# ------------------------------------------------------------------ resolve
def test_resolve_finds_atoms_by_identity():
    fx = fixed_b()
    idx, known = fx._resolve(make_feats(), CHAINS)
    assert idx.tolist() == [3, 4, 5]
    assert np.allclose(known.numpy(), rng_xyz(0), atol=1e-6)


def test_resolve_fails_loudly():
    xyz = rng_xyz(0)
    partial = FixedAtoms({("B", 0, "CA"): xyz[0], ("B", 1, "CA"): xyz[1]})  # CB missing: 2 of 3 atoms known
    with pytest.raises(ValueError, match="have known"):
        partial._resolve(make_feats(), CHAINS)
    ok = FixedAtoms({("B", 0, "CA"): xyz[0], ("B", 1, "CA"): xyz[1]}, min_fraction=0.6)
    assert ok._resolve(make_feats(), CHAINS)[0].tolist() == [3, 4]
    with pytest.raises(ValueError, match="not in the Boltz input"):
        FixedAtoms({("Q", 0, "CA"): xyz[0]})._resolve(make_feats(), CHAINS)
    with pytest.raises(ValueError):
        FixedAtoms({})
    with pytest.raises(ValueError):
        FixedAtoms({("B", 0, "CA"): [np.nan, 0, 0]})


# --------------------------------------------------------------------- hooks
def test_x0_hook_puts_fixed_atoms_on_aligned_known_and_leaves_the_rest():
    fx, feats = fixed_b(), make_feats()
    hook = fx.make_x0_hook(CHAINS)
    torch.manual_seed(0)
    denoised = torch.randn(2, N_ATOMS, 3) * 5
    noised = torch.randn(2, N_ATOMS, 3) * 5
    known = rng_xyz(0)
    for b in range(2):  # noisy halo = rigidly moved known + noise
        noised[b, HALO] = torch.tensor(known @ rotz(0.7 * (b + 1)).T + [5.0, -3.0, 2.0] + rng_xyz(10 + b, scale=0.3)).float()
    out = hook(denoised, noised, {"feats": feats})
    other = np.setdiff1d(np.arange(N_ATOMS), HALO)
    assert torch.equal(out[:, other], denoised[:, other])  # nothing else touched
    w = torch.ones(2, 3)
    k = torch.tensor(known).float().unsqueeze(0).expand(2, -1, -1)
    fitted = weighted_rigid_align(k, out[:, HALO], w, w)
    assert (fitted - out[:, HALO]).abs().max() < 1e-4  # exactly the known shape, rigidly placed


def test_x0_hook_follows_rigid_motion_of_the_noisy_input():
    fx, feats = fixed_b(), make_feats()
    hook = fx.make_x0_hook(CHAINS)
    base = torch.zeros(1, N_ATOMS, 3)
    noised = torch.randn(1, N_ATOMS, 3) * 4
    R, t = rotz(1.1), np.array([7.0, 1.0, -4.0])
    moved = noised.clone()
    moved[0] = torch.tensor(noised[0].numpy() @ R.T + t).float()
    a = hook(base.clone(), noised, {"feats": feats})[0, HALO].numpy()
    b = hook(base.clone(), moved, {"feats": feats})[0, HALO].numpy()
    assert np.allclose(b, a @ R.T + t, atol=1e-4)


def test_alignment_weights_fit_on_fixed_atoms_only():
    """A mismatch of the free chain with its prediction must not rotate the fixed atoms."""
    fx, feats = fixed_b(), make_feats()
    wfn = fx.make_alignment_weights(CHAINS)
    wv = wfn(feats)
    assert wv.shape == (N_ATOMS,) and wv.tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    torch.manual_seed(1)
    pred = torch.randn(1, N_ATOMS, 3) * 6                      # the prediction (x0)
    noisy = pred.clone()
    noisy[0, :3] = torch.tensor(pred[0, :3].numpy() @ rotz(0.7).T).float()  # free chain 40 deg off, fixed atoms agree
    mask = feats["atom_pad_mask"].float()
    full = weighted_rigid_align(noisy, pred, mask, mask)
    w = mask * wv
    halo_only = weighted_rigid_align(noisy, pred, w, w)
    assert (halo_only[0, HALO] - pred[0, HALO]).abs().max() < 1e-4   # fixed atoms stay on their prediction
    assert (full[0, HALO] - pred[0, HALO]).abs().max() > 0.5         # the all-atom fit drags them away


def make_stub_diffusion():
    m = AtomDiffusion.__new__(AtomDiffusion)
    torch.nn.Module.__init__(m)
    m.c_in = lambda s: torch.ones_like(s)
    m.c_skip = lambda s: 0.5 * torch.ones_like(s)
    m.c_out = lambda s: 2.0 * torch.ones_like(s)
    m.c_noise = lambda s: s
    m.score_model = lambda r_noisy, times, **kw: 3 * r_noisy
    m.x0_hook = None
    m.alignment_weights = None
    return m


def test_forward_is_stock_without_hook_and_calls_hook_with_it():
    m = make_stub_diffusion()
    x = torch.randn(2, 4, 3)
    kwargs = {"feats": "F"}
    out = m.preconditioned_network_forward(x, 5.0, kwargs)
    assert torch.allclose(out, 0.5 * x + 2.0 * (3 * x))  # c_skip*x + c_out*r_update, untouched
    seen = {}

    def hook(denoised, noised, kw):
        seen.update(kw=kw, same_noised=noised is x)
        return denoised + 1.0

    m.x0_hook = hook
    out2 = m.preconditioned_network_forward(x, 5.0, kwargs)
    assert torch.allclose(out2, out + 1.0) and seen["kw"] is kwargs and seen["same_noised"]


# ------------------------------------------------------------- template etc.
def test_template_writer_and_entry(tmp_path):
    import gemmi

    fx = fixed_b()
    p = fx.write_template(tmp_path / "t.pdb")
    text = p.read_text()
    assert "SEQRES" in text and text.count("\nATOM") == 3
    st = gemmi.read_structure(str(p))
    assert [c.name for c in st[0]] == ["B"] and [r.name for r in st[0]["B"]] == ["ALA", "GLY"]
    e = fx.template_entry(p)
    assert e["chain_id"] == ["B"] and e["template_id"] == ["B1"] and e["force"] is False and "threshold" not in e
    assert fx.template_entry(p, force=True, threshold=0.5)["threshold"] == 0.5
    with pytest.raises(ValueError):
        fx.template_entry(p, force=True)
    with pytest.raises(ValueError):
        FixedAtoms(fx.coords).write_template(tmp_path / "x.pdb")  # no residue names


def test_transform_from_structure_roundtrip(tmp_path):
    fx = fixed_b()
    R, t = rotz(0.9), np.array([12.0, -5.0, 3.0])
    moved = FixedAtoms({k: v @ R.T + t for k, v in fx.coords.items()}, residue_names=fx.residue_names)
    p = moved.write_template(tmp_path / "moved.pdb")
    tf = fx.transform_from_structure(p)
    assert tf["rmsd"] < 1e-2  # PDB writes 3 decimals
    keys = sorted(moved.coords)
    back = apply_transform(np.array([moved.coords[k] for k in keys]), tf)
    assert np.allclose(back, np.array([fx.coords[k] for k in keys]), atol=1e-2)


def test_superpose_and_from_structure(tmp_path):
    a = rng_xyz(3, 6)
    R, t = rotz(0.4), np.array([1.0, 2.0, 3.0])
    tf = superpose(a, a @ R.T + t)
    assert tf["rmsd"] < 1e-9 and np.allclose(apply_transform(a, tf), a @ R.T + t)
    pdb = tmp_path / "s.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  H   ALA A   1       0.000   1.000   0.000  1.00  0.00           H\n"
        "ATOM      4  CA  GLY A   2       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      5  OXT GLY A   2       2.000   1.000   0.000  1.00  0.00           O\n"
        "ATOM      6  CA  ALA B   1       3.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    fx = FixedAtoms.from_structure(pdb, "A", min_fraction=0.5)
    assert sorted(fx.coords) == [("A", 0, "CA"), ("A", 0, "N"), ("A", 1, "CA")]  # no H, no OXT, no chain B
    assert fx.residue_names == {"A": ["ALA", "GLY"]}
    with pytest.raises(ValueError, match="not found"):
        FixedAtoms.from_structure(pdb, "Z")
    moved = FixedAtoms.from_structure(pdb, "A", transform={"rotation": np.eye(3), "center_from": np.zeros(3), "center_to": np.array([10.0, 0, 0])})
    assert np.allclose(moved.coords[("A", 0, "CA")], [11.0, 0, 0])


# ------------------------------------------------------- sampler-level checks
BEADS_A_ONLY = CGMap([CGBead(atoms=[("A", 0, "CA")]), CGBead(atoms=[("A", 1, "CA")])])
BEADS_AB = CGMap([CGBead(atoms=[("A", 0, "CA")]), CGBead(atoms=[("B", 0, "CA")])])


def test_sampler_fixed_validation(tmp_path):
    s = CGSampler(cache=str(tmp_path))
    fx = fixed_b()
    # path input + default fixed_template=True cannot take the template: fails before any heavy work
    with pytest.raises(ValueError, match="needs `inputs` as a dict"):
        s.sample("x.yaml", tmp_path, BEADS_AB, np.zeros((2, 3)), fixed=fx)
    assert s._model is None
    with pytest.warns(UserWarning, match="step_scale"):
        _check_fixed_settings(fx, 1.5)
    _check_fixed_settings(fx, 1.0)  # no warning
    with pytest.raises(ValueError, match="not in the Boltz input"):
        _check_fixed_chains(fx, BEADS_AB, {"A": 0})
    with pytest.warns(UserWarning, match="no CG bead"):
        _check_fixed_chains(fx, BEADS_A_ONLY, CHAINS)
    _check_fixed_chains(fx, BEADS_AB, CHAINS)  # beads in the fixed chain: fine
