#!/bin/bash
#SBATCH --job-name=ft-bl-orig-l90obj
#SBATCH --account=scalingcrl
#SBATCH --partition=pli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=30:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=ij9461@princeton.edu
#SBATCH --output=/scratch/gpfs/EYSENBACH/ij9461/logs/ft_baseline_original_l90_objdisjoint_eval_object_%j.out
#SBATCH --error=/scratch/gpfs/EYSENBACH/ij9461/logs/ft_baseline_original_l90_objdisjoint_eval_object_%j.err

# Baseline (no MoE), keep original action expert (no reinitialization).
# Train dataset should be a prefiltered LIBERO-90 subset with object vocabulary disjoint from LIBERO-Object.
# Eval at eval_freq on all LIBERO suites: libero_10, libero_goal, libero_object, libero_spatial.

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

export WANDB_MODE=offline
export WANDB_DIR=$SCR/wandb
export WANDB_CACHE_DIR=$SCR/.cache/wandb
export WANDB_DATA_DIR=$SCR/.local/share/wandb
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

# Force LIBERO to use env-local paths instead of stale ~/.libero config.
export LIBERO_CONFIG_PATH="$SCR/.libero"
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<'PY'
import os
import yaml
import libero.libero as L

cfg = L.get_default_path_dict()
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

export TRAIN_DATASET_ROOT=/scratch/gpfs/EYSENBACH/ij9461/hf_cache_user/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/cc29b569e0c32cd8d492757c7f2e076de90c7ba5
export TRAIN_DATASET_REPO_ID=HuggingFaceVLA/libero
export KEEP_EPISODES_JSON=/scratch/gpfs/EYSENBACH/ij9461/datasets/libero90_objdisjoint_keep_episodes.json
KEEP_EPISODES=$(python - <<'PY'
import json, os
path = os.environ["KEEP_EPISODES_JSON"]
print(",".join(str(x) for x in json.load(open(path, "r"))))
PY
)

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

rm -rf $SCR/outputs/ft_baseline_original_l90_objdisjoint_eval_object

cd /home/ij9461/Documents/vla-moe-diversity

python -m lerobot.scripts.lerobot_train \
  --policy.path=$SCR/huggingface/smolvla_base \
  --policy.push_to_hub=false \
  --policy.reinit_expert_mlps=false \
  --dataset.repo_id=$TRAIN_DATASET_REPO_ID \
  --dataset.root=$TRAIN_DATASET_ROOT \
  --dataset.episodes="[$KEEP_EPISODES]" \
  --env.type=libero \
  --env.task=libero_10,libero_goal,libero_object,libero_spatial \
  --batch_size=64 \
  --num_workers=7 \
  --steps=50000 \
  --eval_freq=2500 \
  --eval.batch_size=5 \
  --eval.n_episodes=50 \
  --log_freq=100 \
  --save_freq=5000 \
  --output_dir=$SCR/outputs/ft_baseline_original_l90_objdisjoint_eval_object \
  --job_name=ft_baseline_original_l90_objdisjoint_eval_object \
  --seed=1000 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.disable_artifact=true \
  --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'
