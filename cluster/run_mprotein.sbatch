#!/bin/bash
# usage (from anywhere):  sbatch --account=<your_pi_account> cluster/run_mprotein.sbatch [--out NAME] [--sigma 1.0] ...
#SBATCH --job-name=cg-mprotein
#SBATCH --partition=beagle3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
REPO=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "$REPO/cluster/env.sh" ] || { echo "submit from the repo root"; exit 1; }
source "$REPO/cluster/env.sh"
cd "$REPO/cg_tests/mprotein"
nvidia-smi
python run_mprotein.py --out "${OUT:-m_test}" --sigma 1.0 "$@"
