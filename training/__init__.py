"""Training infrastructure for LangSteer policies.

This module contains training-specific code extracted from 3D-Diffusion-Policy
and adapted for LangSteer's architecture. It provides dataset loaders, training
loops, checkpoint management, and other utilities needed for training diffusion
policies on manipulation datasets.

Key components:
- dp3_trainer: Main training workspace for DP3
- calvin_dataset: PyTorch Dataset for CALVIN manipulation data (Zarr format)
- replay_buffer: Zarr-based replay buffer for efficient data loading
- sampler: Sequence sampling utilities for temporal batching
- checkpoint_util: TopK checkpoint management
- ema_model: Exponential Moving Average for stable training
"""

__all__ = [
    'DP3TrainingWorkspace',
    'CalvinDataset',
    'ReplayBuffer',
    'SequenceSampler',
    'TopKCheckpointManager',
    'EMAModel',
]
