#!/usr/bin/env python3
"""Training script for DP3 on CALVIN dataset.

Supports both single-GPU and multi-GPU distributed training on SLURM clusters.

Usage:
    # Single GPU (debug)
    python scripts/train_dp3.py training=dp3_debug

    # Single GPU (full training)
    python scripts/train_dp3.py training=dp3_calvin

    # Multi-GPU with SLURM
    sbatch scripts/slurm_train_dp3.sh

    # Override specific parameters
    python scripts/train_dp3.py training=dp3_calvin training.batch_size=64 training.num_epochs=1000
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from training.policies.dp3.trainer import DP3TrainingWorkspace

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training entry point."""

    # Ensure training config is provided
    if cfg.training is None:
        raise ValueError(
            "Training configuration not specified. "
            "Use: python scripts/train_dp3.py training=dp3_calvin"
        )

    # Log configuration (only on main process)
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        logger.info("=" * 80)
        logger.info("DP3 Training Configuration:")
        logger.info("=" * 80)
        logger.info(OmegaConf.to_yaml(cfg.training))
        logger.info("=" * 80)

    # Initialize and run training workspace
    workspace = DP3TrainingWorkspace(cfg.training)
    workspace.train()


if __name__ == "__main__":
    main()
