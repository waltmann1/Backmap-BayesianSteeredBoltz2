# Source this before using Boltz:  source cluster/env.sh
# BOLTZ_ROOT = directory that contains the repo clone, the envs/ and the cache/
# (default: the parent of the repo). Override with: export BOLTZ_ROOT=/beagle3/<lab>/<you>/boltz2
_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BOLTZ_ROOT=${BOLTZ_ROOT:-$(dirname "$(dirname "$_HERE")")}
export CONDA_PKGS_DIRS=$BOLTZ_ROOT/cache/conda_pkgs
export PIP_CACHE_DIR=$BOLTZ_ROOT/cache/pip
export XDG_CACHE_HOME=$BOLTZ_ROOT/cache/xdg
export TMPDIR=$BOLTZ_ROOT/cache/tmp; mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR=$BOLTZ_ROOT/cache/triton
export CUDA_CACHE_PATH=$BOLTZ_ROOT/cache/nv
export TORCH_HOME=$BOLTZ_ROOT/cache/torch
export TORCHINDUCTOR_CACHE_DIR=$BOLTZ_ROOT/cache/inductor
export NUMBA_CACHE_DIR=$BOLTZ_ROOT/cache/numba
export MPLCONFIGDIR=$BOLTZ_ROOT/cache/mpl
export WANDB_DIR=$BOLTZ_ROOT/cache/wandb WANDB_CONFIG_DIR=$BOLTZ_ROOT/cache/wandb WANDB_CACHE_DIR=$BOLTZ_ROOT/cache/wandb
export HF_HOME=$BOLTZ_ROOT/cache/hf
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
         "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$WANDB_DIR" "$HF_HOME" "$BOLTZ_ROOT/cache/boltz"
export BOLTZ_CACHE=$BOLTZ_ROOT/cache/boltz   # pass to boltz via --cache
if [ -z "${BOLTZ_NO_ACTIVATE:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$BOLTZ_ROOT/envs/boltz2-bayes"
fi
