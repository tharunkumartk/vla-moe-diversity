#!/bin/bash
#SBATCH --job-name=ft-gen-moe-div
#SBATCH --account=eladgroup
#SBATCH --partition=pli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/EHAZAN/tharuntk/logs/ft_gen_moe_diversity_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/tharuntk/logs/ft_gen_moe_diversity_%j.err

mkdir -p /scratch/gpfs/EHAZAN/tharuntk/logs

export HF_LEROBOT_HOME=/scratch/gpfs/EHAZAN/tharuntk/lerobot_data
export HF_HOME=/scratch/gpfs/EHAZAN/tharuntk/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=osmesa
mkdir -p /tmp/osmesa_fix && ln -sf /lib64/libOSMesa.so.8 /tmp/osmesa_fix/libOSMesa.so
export LD_LIBRARY_PATH=/tmp/osmesa_fix:${LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/tt6444/lerobot/.venv/bin/activate

# Fix robosuite log permission issue on shared nodes
ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "$ROBOSUITE_MACROS_PRIVATE" ] && [ ! -f "$ROBOSUITE_MACROS_PRIVATE" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "$ROBOSUITE_MACROS_PRIVATE"
fi

rm -rf /scratch/gpfs/EHAZAN/tharuntk/outputs/finetune_gen/moe_diversity_goal

python -m lerobot.scripts.lerobot_train \
  --policy.path=/scratch/gpfs/EHAZAN/tharuntk/outputs/moe_diversity/checkpoints/last/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_goal \
  --batch_size=64 \
  --num_workers=7 \
  --steps=2000 \
  --eval_freq=200 \
  --eval.batch_size=5 \
  --eval.n_episodes=50 \
  --log_freq=50 \
  --save_freq=500 \
  --output_dir=/scratch/gpfs/EHAZAN/tharuntk/outputs/finetune_gen/moe_diversity_goal \
  --job_name=ft_gen_moe_diversity_goal \
  --seed=1000 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
