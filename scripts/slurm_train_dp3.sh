#!/bin/bash
#SBATCH --job-name=dp3_calvin
#SBATCH --output=logs/%j_dp3.out
#SBATCH --error=logs/%j_dp3.err
#SBATCH --partition=clear-l40s
#SBATCH --account=clear
#SBATCH --qos=clear-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# =============================================================================
# DP3 DDP training on SLURM (4 GPUs, 1 node)
# Submit from repo root: sbatch scripts/slurm_train_dp3.sh
# =============================================================================

# Environment setup (uv-managed venv)
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# Environment variables
export CALVIN_ZARR_PATH="${CALVIN_ZARR_PATH:?Set CALVIN_ZARR_PATH before submitting}"
export WANDB_MODE=online
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1

# DDP master address and port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

# Create logs directory
mkdir -p logs

# Job info
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | GPUs: $SLURM_NTASKS"
echo "Master: $MASTER_ADDR:$MASTER_PORT | Dataset: $CALVIN_ZARR_PATH"

# Launch DDP training
srun python scripts/train_dp3.py training=dp3_calvin \
    training.experiment_name="dp3_calvin_slurm_${SLURM_JOB_ID}"
