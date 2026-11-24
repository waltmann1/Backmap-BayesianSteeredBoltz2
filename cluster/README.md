# Setting up on Beagle3 (any account)

Tested from a fresh clone: install takes ~15 min, uses ~11 GB and ~57,000 files
(env 8 GB, caches 2 GB), plus ~8 GB for the model weights on the first GPU run.
If `pip` fails with `No space left on device`, check `quota -s` (also the file
count) and contact RCC instead of retrying.

Everything (env, caches, weights, outputs) goes under `BOLTZ_ROOT` on `/beagle3`,
never `$HOME` (it has almost no space).

```bash
cd /beagle3/<lab>/<you>            # persistent project dir with ~20 GB free (not scratch: it is purged)
mkdir boltz2 && cd boltz2
git clone -b cg-potential https://github.com/waltmann1/Backmap-BayesianSteeredBoltz2.git
bash Backmap-BayesianSteeredBoltz2/cluster/setup.sh   # login node, ~10-20 min
```
`setup.sh` makes `envs/boltz2-bayes` (Python 3.11, `cluster/env.yml`), installs torch
2.10.0+cu126, then the fork editable (`pip install -e`) pinned by `cluster/constraints.txt`.
If `conda` is not found, load Anaconda first (`module avail python`).

Use it:
```bash
source Backmap-BayesianSteeredBoltz2/cluster/env.sh      # always, also in job scripts
python -m pytest Backmap-BayesianSteeredBoltz2/tests/test_cg_potential.py   # no GPU needed
cd Backmap-BayesianSteeredBoltz2
sbatch --account=<your_pi_account> cluster/run_mprotein.sbatch --out run1   # GPU test, M-protein
```
First GPU run downloads ~8 GB of weights into `$BOLTZ_ROOT/cache/boltz`. To skip that,
copy an existing `cache/boltz` (`boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols/`).

Rules: GPU jobs only on GPU nodes; torch must stay a cu12x build (driver 535);
use a unique `--out` per run (outputs are overwritten).
