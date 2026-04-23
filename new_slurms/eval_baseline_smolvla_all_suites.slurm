#!/bin/bash
#SBATCH --job-name=eval-smolvla-base
#SBATCH --account=scalingcrl
#SBATCH --partition=pli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=4G
#SBATCH --time=20:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=ij9461@princeton.edu
#SBATCH --output=/scratch/gpfs/EYSENBACH/ij9461/logs/eval_baseline_smolvla_all_suites_%j.out
#SBATCH --error=/scratch/gpfs/EYSENBACH/ij9461/logs/eval_baseline_smolvla_all_suites_%j.err
#
# Zero-shot baseline: evaluate pretrained smolvla_base (no finetuning) across
# all four LIBERO suites: libero_90, libero_10, libero_object, libero_goal.

set -euo pipefail
export PS1=${PS1:-}

module purge
module load anaconda3/2024.2
module load cmake/3.30.8

export SCR=/scratch/gpfs/EYSENBACH/ij9461
export TMPDIR=$SCR/tmp
export PIP_CACHE_DIR=$SCR/.cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$SCR/logs"

eval "$(conda shell.bash hook)"
conda activate "$SCR/.conda/envs/lerobot2"

# Force LIBERO to use env-local paths instead of stale ~/.libero config.
export LIBERO_CONFIG_PATH="$SCR/.libero"
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<'PY'
import os
import yaml
import libero.libero as L

cfg = L.get_default_path_dict()
cfg = {k: os.path.expanduser(v) if isinstance(v, str) else v for k, v in cfg.items()}
cfg_path = os.path.join(os.environ["LIBERO_CONFIG_PATH"], "config.yaml")
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f)
print(f"[libero] wrote config: {cfg_path}")
PY

export HF_LEROBOT_HOME=$SCR/huggingface/lerobot
export HF_HOME=$SCR/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=osmesa
mkdir -p "$TMPDIR/osmesa_fix" && ln -sf /lib64/libOSMesa.so.8 "$TMPDIR/osmesa_fix/libOSMesa.so"
export LD_LIBRARY_PATH="$TMPDIR/osmesa_fix:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

RUN_ID="eval_baseline_smolvla_all_suites_$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.lerobot_eval \
  --policy.path=$SCR/huggingface/smolvla_base \
  --policy.use_amp=true \
  --env.type=libero \
  --env.task=libero_90,libero_10,libero_object,libero_goal \
  --env.max_parallel_tasks=1 \
  --eval.tasks_per_batch=10 \
  --eval.n_episodes_per_task=5 \
  --eval.use_async_envs=true \
  --output_dir=$SCR/outputs/${RUN_ID} \
  --job_name=${RUN_ID} \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
