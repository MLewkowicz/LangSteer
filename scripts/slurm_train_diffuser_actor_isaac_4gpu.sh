#!/bin/bash
#SBATCH --job-name=da_isaac_4gpu
#SBATCH --output=logs/%j_da_isaac_4gpu.out
#SBATCH --error=logs/%j_da_isaac_4gpu.err
#SBATCH --partition=clear-l40s
#SBATCH --account=clear
#SBATCH --qos=clear-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4        # one task per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8          # 8 dataloader workers per GPU
#SBATCH --mem=128G                 # 4× the single-GPU run
#SBATCH --time=8:00:00

# =============================================================================
# Diffuser Actor (Isaac Sim primitive+object) training on SLURM — 4 GPUs DDP
# Submit from repo root: sbatch scripts/slurm_train_diffuser_actor_isaac_4gpu.sh
#
# DDP plumbing: trainer reads SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS to
# init the NCCL process group.  With ntasks-per-node=4 and gres=gpu:4, each
# task pins to one GPU; effective global batch = batch_size * 4 = 32.
# =============================================================================

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# IsaacDataset resolves train_path / val_path against ISAAC_DEMO_DIR.
export ISAAC_DEMO_DIR=/data/scratch/mlewkowicz/isaac_sim_demos

export WANDB_MODE=online
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN              # set INFO if NCCL hangs

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | World size: $SLURM_NTASKS"
echo "Dataset: $ISAAC_DEMO_DIR"

# To resume from a single-GPU checkpoint (or any prior run), uncomment and
# point at the .pth file.  The loader handles the .module. prefix mismatch
# between single-GPU and DDP checkpoints automatically.
#
#   training.resume=true \
#   training.resume_checkpoint_path=/abs/path/to/last.pth
#
# Note: switching world size mid-cosine-schedule causes a small LR transient
# (effective batch quadruples). Loss may briefly tick up before settling.

srun python scripts/train_diffuser_actor.py \
    training=diffuser_actor_isaac \
    training.experiment_name="da_isaac_4gpu_${SLURM_JOB_ID}"
