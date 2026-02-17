"""CALVIN Dataset for DP3 training.

Loads CALVIN manipulation data from Zarr format with point clouds,
robot proprioception, and actions.
"""

from typing import Dict
import torch
import numpy as np
import copy
from training.common.pytorch_util import dict_apply
from training.policies.dp3.replay_buffer import ReplayBuffer
from training.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from policies.dp3_components.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from termcolor import cprint


class CalvinDataset(torch.utils.data.Dataset):
    """CALVIN dataset loader for DP3 training.

    Loads pre-processed CALVIN data from Zarr format including:
    - Point clouds (2048 points, XYZRGB format, but only XYZ used)
    - Robot proprioception (15D state)
    - Actions (7D relative actions)

    Args:
        zarr_path: Path to Zarr dataset directory
        horizon: Sequence length for temporal batching
        pad_before: Number of frames to pad before sequence
        pad_after: Number of frames to pad after sequence
        seed: Random seed for train/val split
        val_ratio: Fraction of episodes to use for validation
        max_train_episodes: Maximum number of training episodes (None = use all)
    """

    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'img', 'depth'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        """Create a validation dataset from the same replay buffer."""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """Compute normalizer statistics from training data.

        Args:
            mode: Normalization mode ('limits' or 'gaussian')
            **kwargs: Additional arguments for normalizer

        Returns:
            LinearNormalizer fitted to the training data
        """
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            # Point clouds use identity normalization (no normalization)
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """Convert sampled data to model input format.

        Args:
            sample: Dictionary from sampler with keys: state, action, point_cloud

        Returns:
            Dictionary with 'obs' and 'action' keys
        """
        agent_pos = sample['state'][:,].astype(np.float32)  # (T, 15) - robot_obs from CALVIN
        point_cloud = sample['point_cloud'][:,].astype(np.float32)  # (T, 2048, 6) - XYZRGB

        # Extract only XYZ coordinates for point cloud (first 3 channels)
        point_cloud_xyz = point_cloud[..., :3]  # (T, 2048, 3)

        data = {
            'obs': {
                'point_cloud': point_cloud_xyz,  # T, 2048, 3
                'agent_pos': agent_pos,  # T, 15
            },
            'action': sample['action'].astype(np.float32)  # T, 7 - rel_actions from CALVIN
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'obs' and 'action' tensors
        """
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
