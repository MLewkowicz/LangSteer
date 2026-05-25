#!/bin/bash
#SBATCH --job-name=diffuser_actor_isaac
#SBATCH --output=logs/%j_da_isaac.out
#SBATCH --error=logs/%j_da_isaac.err
#SBATCH --partition=clear-l40s
#SBATCH --account=clear
#SBATCH --qos=clear-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# =============================================================================
# Diffuser Actor (Isaac Sim primitive+object) training on SLURM — single GPU
# Submit from repo root: sbatch scripts/slurm_train_diffuser_actor_isaac.sh
# =============================================================================

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# IsaacDataset resolves train_path / val_path against ISAAC_DEMO_DIR.
# Expects $ISAAC_DEMO_DIR/training and $ISAAC_DEMO_DIR/validation to contain
# episode_*.h5 files written by IsaacSimRecorder.
export ISAAC_DEMO_DIR=/data/scratch/mlewkowicz/isaac_sim_demos

# Prefer `wandb login` (writes ~/.netrc) over committing keys.  Fallback: set
# WANDB_API_KEY in your shell rc on the cluster, not here.
export WANDB_MODE=online
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Dataset: $ISAAC_DEMO_DIR"

srun python scripts/train_diffuser_actor.py \
    training=diffuser_actor_isaac \
    training.experiment_name="da_isaac_${SLURM_JOB_ID}"
