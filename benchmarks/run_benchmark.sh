#!/bin/bash
# System benchmark for lerobot VLA training/eval.
# Run directly on a GPU node (NOT via sbatch):
#   bash benchmarks/run_benchmark.sh
#
# Optional: skip slow sections
#   bash benchmarks/run_benchmark.sh --skip inference libero_env
#
# GPU selection: change CUDA_VISIBLE_DEVICES below if needed.

set -euo pipefail
export PS1=${PS1:-}

# ── GPU selection ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# ── Environment setup (mirrors the slurm scripts) ─────────────────────────────
module purge
module load anaconda3/2024.2
module load cmake/3.30.8

export SCR=/scratch/gpfs/EYSENBACH/ij9461
export TMPDIR=$SCR/tmp
export PIP_CACHE_DIR=$SCR/.cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$SCR/logs"

eval "$(conda shell.bash hook)"
conda activate "$SCR/.conda/envs/lerobot2"

# ── WandB (disabled for benchmark) ───────────────────────────────────────────
export WANDB_MODE=disabled

# ── LIBERO config ─────────────────────────────────────────────────────────────
export LIBERO_CONFIG_PATH="$SCR/.libero"
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<'PY'
import os, yaml, libero.libero as L
cfg = L.get_default_path_dict()
# Expand ~ so LIBERO finds cached assets on first check (avoids repeated "downloading" messages)
cfg = {k: os.path.expanduser(v) if isinstance(v, str) else v for k, v in cfg.items()}
cfg_path = os.path.join(os.environ["LIBERO_CONFIG_PATH"], "config.yaml")
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f)
print(f"[libero] config written: {cfg_path}")
PY

# ── HuggingFace + rendering ───────────────────────────────────────────────────
export HF_LEROBOT_HOME=$SCR/huggingface/lerobot
export HF_HOME=$SCR/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=osmesa
mkdir -p "$TMPDIR/osmesa_fix"
ln -sf /lib64/libOSMesa.so.8 "$TMPDIR/osmesa_fix/libOSMesa.so" 2>/dev/null || true
export LD_LIBRARY_PATH="$TMPDIR/osmesa_fix:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── robosuite quiet logging ────────────────────────────────────────────────────
ROBOSUITE_MACROS=$(python -c "
import robosuite, os
print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))
" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS" ] && [ ! -f "$ROBOSUITE_MACROS" ]; then
    echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS"
fi

# ── Dataset / policy paths (passed to benchmark script via env vars) ──────────
export TRAIN_DATASET_ROOT=$SCR/hf_cache_user/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/cc29b569e0c32cd8d492757c7f2e076de90c7ba5
export TRAIN_DATASET_REPO_ID=HuggingFaceVLA/libero
export POLICY_PATH=$SCR/huggingface/smolvla_base

# ── Python path ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ── Print context ─────────────────────────────────────────────────────────────
echo "========================================================================"
echo "  lerobot benchmark runner"
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "  HOSTNAME = $(hostname)"
echo "  DATE     = $(date)"
echo "========================================================================"
echo ""
nvidia-smi
echo ""
echo "CPU cores: $(nproc)"
echo "RAM:"
grep -E "MemTotal|MemAvailable" /proc/meminfo
echo ""

# ── Run benchmark ─────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
python benchmarks/benchmark_system.py "$@" 2>&1 | tee "$SCR/logs/benchmark_$(date +%Y%m%d_%H%M%S).log"
