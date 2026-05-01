#!/bin/bash
#SBATCH --job-name=crl_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --constraint=gpu80
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=ij9461@princeton.edu
#SBATCH --output=/scratch/gpfs/EYSENBACH/ij9461/vla-moe-diversity/slurm_logs/eval_expert_masking_sep_experts_noisy_035000_mask012-%j.out
#SBATCH --error=/scratch/gpfs/EYSENBACH/ij9461/vla-moe-diversity/slurm_logs/eval_expert_masking_sep_experts_noisy_035000_mask012-%j.err

set -euo pipefail
export PS1=${PS1:-}

module purge
module load anaconda3/2024.2
module load cmake/3.30.8

export SCR=/scratch/gpfs/EYSENBACH/ij9461
export REPO_ROOT=$SCR/vla-moe-diversity
export TMPDIR=$SCR/tmp
export PIP_CACHE_DIR=$SCR/.cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$REPO_ROOT/slurm_logs"

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=osmesa
mkdir -p "$TMPDIR/osmesa_fix" && ln -sf /lib64/libOSMesa.so.8 "$TMPDIR/osmesa_fix/libOSMesa.so"
export LD_LIBRARY_PATH="$TMPDIR/osmesa_fix:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [[ -n "$ROBOSUITE_MACROS_PRIVATE" && ! -f "$ROBOSUITE_MACROS_PRIVATE" ]]; then
    echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK:-3}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/expert_masking_eval/ft_residual_separate_experts_lg_diversity_noisy_step035000}
POLICY_PATH=$SCR/outputs/ft_residual_separate_experts_lg_diversity_noisy_20260425_212717/checkpoints/035000/pretrained_model

cd "$REPO_ROOT"

python -m lerobot.scripts.eval_expert_masking \
  --policy.path "$POLICY_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --subsets "libero_10,libero_goal,libero_object,libero_spatial" \
  --rollouts_per_task "$ROLLOUTS_PER_TASK" \
  --masked_experts 0 1 2 \
  --skip_original \
  --seed 1000 \
  --policy-device cuda \
  --policy-use-amp \
  --results_filename "results_masked_012.json" \
  --rename_map '{"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2", "observation.images.image2": "observation.images.camera2"}'
