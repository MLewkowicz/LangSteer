"""
Trajectory data collection utilities.

Provides standardized data structures and collection methods
for recording policy rollout data.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from core.types import Observation, Action

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """Single timestep of trajectory data."""

    timestep: int                    # Step number in episode
    observation: Observation         # Observation at this step
    action: Action                   # Action taken
    reward: float                    # Reward received
    done: bool                       # Whether episode ended
    info: Dict                       # Additional info from environment

    # Commonly used derived fields
    ee_position: Optional[np.ndarray] = None   # (3,) End-effector x,y,z
    ee_pose: Optional[np.ndarray] = None       # (7,) Full EE pose

    def __post_init__(self):
        """Extract commonly used fields from observation."""
        if self.ee_position is None and hasattr(self.observation, 'ee_pose'):
            self.ee_position = self.observation.ee_pose[:3].copy()
        if self.ee_pose is None and hasattr(self.observation, 'ee_pose'):
            self.ee_pose = self.observation.ee_pose.copy()


class TrajectoryDataCollector:
    """Collects and manages trajectory data during rollouts."""

    def __init__(self):
        """Initialize empty trajectory collector."""
        self.trajectory: List[TrajectoryPoint] = []
        self.episode_reward: float = 0.0
        self.episode_length: int = 0
        self.success: bool = False

    def reset(self):
        """Reset for new episode."""
        self.trajectory = []
        self.episode_reward = 0.0
        self.episode_length = 0
        self.success = False

    def add_step(
        self,
        timestep: int,
        observation: Observation,
        action: Action,
        reward: float,
        done: bool,
        info: Dict
    ):
        """
        Add a single timestep to the trajectory.

        Args:
            timestep: Current step number
            observation: Observation before action
            action: Action taken
            reward: Reward received
            done: Whether episode ended
            info: Additional info from environment
        """
        point = TrajectoryPoint(
            timestep=timestep,
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            info=info
        )

        self.trajectory.append(point)
        self.episode_reward += reward
        self.episode_length = timestep + 1

        # Check for success
        if info.get('success', False):
            self.success = True

    def get_trajectory_array(self, field: str = 'ee_position') -> np.ndarray:
        """
        Extract trajectory as numpy array.

        Args:
            field: Which field to extract ('ee_position', 'ee_pose', 'action')

        Returns:
            Numpy array of shape (T, D) where T is trajectory length
        """
        if field == 'ee_position':
            return np.array([p.ee_position for p in self.trajectory])
        elif field == 'ee_pose':
            return np.array([p.ee_pose for p in self.trajectory])
        elif field == 'action':
            return np.array([p.action.trajectory[0] for p in self.trajectory])
        elif field == 'reward':
            return np.array([p.reward for p in self.trajectory])
        else:
            raise ValueError(f"Unknown field: {field}")

    def save_to_npz(self, path: str):
        """
        Save trajectory to .npz file.

        Args:
            path: Output file path
        """
        data = {
            'ee_positions': self.get_trajectory_array('ee_position'),
            'ee_poses': self.get_trajectory_array('ee_pose'),
            'actions': self.get_trajectory_array('action'),
            'rewards': self.get_trajectory_array('reward'),
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'success': self.success,
        }

        np.savez(path, **data)
        logger.info(f"Saved trajectory to {path}")

    def get_statistics(self) -> Dict:
        """
        Compute trajectory statistics.

        Returns:
            Dictionary with statistics
        """
        ee_positions = self.get_trajectory_array('ee_position')

        return {
            'episode_length': self.episode_length,
            'episode_reward': self.episode_reward,
            'success': self.success,
            'spatial_extent': {
                'x_min': float(ee_positions[:, 0].min()),
                'x_max': float(ee_positions[:, 0].max()),
                'y_min': float(ee_positions[:, 1].min()),
                'y_max': float(ee_positions[:, 1].max()),
                'z_min': float(ee_positions[:, 2].min()),
                'z_max': float(ee_positions[:, 2].max()),
            },
            'path_length': float(np.sum(np.linalg.norm(np.diff(ee_positions, axis=0), axis=1))),
        }
