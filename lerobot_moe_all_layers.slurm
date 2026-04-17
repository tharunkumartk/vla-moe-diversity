#!/bin/bash
#SBATCH --job-name=vla_moe_all_layers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=35:59:59
#SBATCH --partition=pli
#SBATCH --account=scalingcrl
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=ij9461@princeton.edu
#SBATCH --output=slurm_logs/slurm-%j.out

# LeRobot SmolVLA MoE (LIBERO-10) — vla-moe-diversity
#
# One-time setup (login node):
#   cd /path/to/vla-moe-diversity
#   uv venv .venv --python 3.12
#   source .venv/bin/activate
#   uv pip install -e ".[smolvla]"
#   python -c 'from huggingface_hub import snapshot_download; snapshot_download("lerobot/smolvla_base", local_dir="checkpoints/smolvla_base")'
#
# Optional: clone/checkout with broken LFS test assets:
#   GIT_LFS_SKIP_SMUDGE=1 git clone ... && git checkout moe-training
#
# Offline compute nodes: set HF_LEROBOT_HOME and HF_HOME to caches populated on
# a login node, then enable the offline exports below.

set -eo pipefail

cd "/home/ij9461/Documents/vla-moe-diversity"

mkdir -p slurm_logs outputs

# --- Python env (conda) ---
module purge
module load anaconda3/2024.2
conda activate vla-moe

# --- Hugging Face / LeRobot caches (all on scratch) ---
export HF_HOME=/scratch/gpfs/EYSENBACH/ij9461/huggingface

# --- Weights & Biases ---
WANDB_ENABLE=false
WANDB_PROJECT="lerobot_MoE"

echo "Starting MoE-all-layers training: PROJECT_ROOT=${PROJECT_ROOT}"
echo "WANDB_ENABLE=${WANDB_ENABLE} WANDB_PROJECT=${WANDB_PROJECT}"

python -m lerobot.scripts.lerobot_train \
  --policy.path=checkpoints/smolvla_base \
  --policy.push_to_hub=false \
  --policy.use_moe=true \
  --policy.moe_all_layers=true \
  --policy.moe_num_experts=8 \
  --policy.moe_top_k=3 \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=lerobot/libero_10 \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.n_episodes=50 \
  --eval.batch_size=50 \
  --batch_size=32 \
  --steps=50000 \
  --eval_freq=5000 \
  --save_freq=10000 \
  --output_dir=/scratch/gpfs/EYSENBACH/ij9461/vla-moe-outputs/moe_all_layers \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${WANDB_PROJECT}" \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}'

echo "Training finished."
