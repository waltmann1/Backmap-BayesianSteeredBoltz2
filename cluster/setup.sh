#!/bin/bash
# One-time install. Usage (on the login node):
#   git clone -b cg-potential https://github.com/waltmann1/Backmap-BayesianSteeredBoltz2.git
#   bash Backmap-BayesianSteeredBoltz2/cluster/setup.sh
# Creates envs/ and cache/ next to the clone (or under $BOLTZ_ROOT if you set it).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
command -v conda >/dev/null || { echo "conda not found: load the Anaconda module (module avail python) and retry"; exit 1; }
BOLTZ_NO_ACTIVATE=1 source "$HERE/env.sh"
echo "BOLTZ_ROOT=$BOLTZ_ROOT"
case "$BOLTZ_ROOT" in "$HOME"*) echo "BOLTZ_ROOT is under \$HOME, which has little space. Set BOLTZ_ROOT to a /beagle3 path."; exit 1;; esac
ENV=$BOLTZ_ROOT/envs/boltz2-bayes
[ -d "$ENV" ] || conda env create -p "$ENV" -f "$HERE/env.yml"
PY=$ENV/bin/python
# torch must be a cu12x build: Beagle3 GPU drivers (535) do not run CUDA 13 wheels.
$PY -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu126
# Fork (editable) with the versions known to work; constraints pin everything else.
$PY -m pip install -e "$REPO[cuda]" mdanalysis -c "$HERE/constraints.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu126
$PY -m pip install pytest
echo "Done. Next: source $HERE/env.sh && python -m pytest $REPO/tests/test_cg_potential.py"
echo "Model weights download on first GPU run into $BOLTZ_CACHE (~8 GB)."
