#!/bin/bash

# Residual full-expert MoE with learned gate + orthogonality diversity loss + noisy routing:
# output = alpha * orig_expert(x) + zeroconv(routed_full_experts(x))
# Router chooses top-k among N full copies of the action expert.
# Noisy routing: N(0,1) noise added to router logits before softmax (training only).
# Diversity loss: penalises pairwise squared cosine similarity of mean expert outputs.
# moe_init_from_pretrained=true: expert copies initialised from the pretrained action expert
# weights (sparse upcycling) rather than from scratch.

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

export LIBERO_CONFIG_PATH="$SCR/.libero"
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<'PY'
import os, yaml, libero.libero as L
cfg = L.get_default_path_dict()
cfg = {k: os.path.expanduser(v) if isinstance(v, str) else v for k, v in cfg.items()}
with open(os.path.join(os.environ["LIBERO_CONFIG_PATH"], "config.yaml"), "w") as f:
    yaml.safe_dump(cfg, f)
PY

export HF_LEROBOT_HOME=$SCR/huggingface/lerobot
export HF_HOME=$SCR/huggingface
export HF_DATASETS_CACHE=$TMPDIR/hf_datasets_${SLURM_JOB_ID:-$$}
mkdir -p "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=osmesa
mkdir -p "$TMPDIR/osmesa_fix" && ln -sf /lib64/libOSMesa.so.8 "$TMPDIR/osmesa_fix/libOSMesa.so"
export LD_LIBRARY_PATH="$TMPDIR/osmesa_fix:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TRAIN_DATASET_ROOT=/scratch/gpfs/EYSENBACH/ij9461/hf_cache_user/lerobot/hub/HuggingFaceVLA_libero_fresh
export TRAIN_DATASET_REPO_ID=HuggingFaceVLA/libero

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

RUN_ID="ft_residual_sep_experts_lg_diversity_noisy_init_pretrained_$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=1 python -m lerobot.scripts.lerobot_train \
  --policy.path=$SCR/huggingface/smolvla_base \
  --policy.push_to_hub=false \
  --policy.use_amp=true \
  --policy.use_moe=true \
  --policy.separate_experts=true \
  --policy.moe_residual_mode=learned_gate \
  --policy.moe_num_experts=8 \
  --policy.moe_top_k=3 \
  --policy.moe_load_balance_weight=0.2 \
  --policy.moe_init_from_pretrained=true \
  --policy.moe_residual_freeze_original=true \
  --policy.use_diversity_loss=true \
  --policy.moe_lambda_orth=0.05 \
  --policy.moe_noisy_routing=true \
  --dataset.repo_id=$TRAIN_DATASET_REPO_ID \
  --dataset.root=$TRAIN_DATASET_ROOT \
  --train_dataset=libero_all \
  --env.type=libero \
  --env.task=libero_10,libero_goal,libero_object,libero_spatial \
  --batch_size=32 \
  --num_workers=20 \
  --steps=50000 \
  --eval_freq=1000 \
  --eval.tasks_per_batch=10 \
  --eval.n_episodes_per_task=3 \
  --eval.use_async_envs=true \
  --log_freq=100 \
  --save_freq=5000 \
  --output_dir=$SCR/outputs/${RUN_ID} \
  --job_name=${RUN_ID} \
  --seed=1000 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.disable_artifact=true \
  --wandb.project=vla-moe-diversity \
  --wandb.group=separate_experts_lg_diversity_noisy_init_pretrained \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2", "observation.images.image2": "observation.images.camera2"}'
