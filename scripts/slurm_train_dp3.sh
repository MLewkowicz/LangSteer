#!/bin/bash
#SBATCH --job-name=dp3_calvin
#SBATCH --output=logs/dp3_%j.out
#SBATCH --error=logs/dp3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # Number of GPUs per node
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4  # Request 4 GPUs
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --partition=gpu

# SLURM Training Script for DP3 on CALVIN
# Launches distributed training across multiple GPUs

# Environment setup
module load cuda/11.8  # Adjust based on your cluster
module load python/3.10  # Adjust based on your cluster

# Activate virtual environment
source ~/.venv/langsteer/bin/activate  # Adjust to your venv path

# Set environment variables
export CALVIN_ZARR_PATH="/path/to/your/calvin_training.zarr"  # UPDATE THIS
export MASTER_PORT=29500
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create logs directory
mkdir -p logs

# Print job info
echo "Starting DP3 training on SLURM"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Dataset: $CALVIN_ZARR_PATH"

# Launch training with srun (SLURM distributed launcher)
srun python scripts/train_dp3.py training=dp3_calvin \
    training.experiment_name="dp3_calvin_slurm_${SLURM_JOB_ID}"

echo "Training complete!"
