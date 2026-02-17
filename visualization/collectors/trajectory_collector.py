"""
Trajectory collection utilities for multi-rollout analysis.

Runs multiple environment rollouts from the same initial state and
collects end-effector trajectory data for visualization and analysis.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
import random
import torch

from core.env import BaseEnvironment
from core.policy import BasePolicy
from core.steering import BaseSteering
from utils.state_management.env_snapshots import EnvSnapshot, EnvSnapshotManager
from utils.rollout import EpisodeRunner

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryData:
    """Storage for a single timestep's trajectory data."""

    rollout_id: int              # Which rollout (0 to N-1)
    timestep: int                # Environment timestep within rollout
    ee_position: np.ndarray      # (3,) - end-effector x, y, z coordinates
    action_taken: np.ndarray     # (7,) - action executed [pos, orient, gripper]
    reward: float                # Reward received at this timestep
    done: bool                   # Whether episode ended


class TrajectoryCollector:
    """Runs N rollouts and collects end-effector trajectories using shared EpisodeRunner."""

    def __init__(
        self,
        env: BaseEnvironment,
        policy: BasePolicy,
        snapshot: EnvSnapshot,
        num_rollouts: int,
        steering: Optional[BaseSteering] = None
    ):
        """
        Initialize trajectory collector.

        Args:
            env: Environment instance
            policy: Policy instance
            snapshot: Initial environment state to start from
            num_rollouts: Number of rollouts to execute
            steering: Optional steering module for guided sampling
        """
        self.env = env
        self.policy = policy
        self.snapshot = snapshot
        self.num_rollouts = num_rollouts
        self.steering = steering
        self.snapshot_manager = EnvSnapshotManager()

        # Storage for collected data
        self.trajectories: List[TrajectoryData] = []
        self.success_count: int = 0

        logger.info(f"TrajectoryCollector initialized")
        logger.info(f"  Rollouts: {num_rollouts}")
        logger.info(f"  Task: {snapshot.task}")
        logger.info(f"  Steering: {steering.__class__.__name__ if steering else 'None'}")

    def collect(self, max_steps: int = 50, show_progress: bool = True) -> List[TrajectoryData]:
        """
        Run N rollouts from same initial state and collect trajectories.

        Args:
            max_steps: Maximum steps per rollout
            show_progress: Whether to show progress bar

        Returns:
            List of TrajectoryData objects
        """
        logger.info(f"Starting trajectory collection: {self.num_rollouts} rollouts, {max_steps} max steps")

        # Clear previous data
        self.trajectories = []
        self.success_count = 0

        # Create episode runner
        runner = EpisodeRunner(
            env=self.env,
            policy=self.policy,
            steering=self.steering,
            max_steps=max_steps,
            collect_data=True  # We'll extract data ourselves via step_callback
        )

        # Setup progress tracking
        if show_progress:
            pbar = tqdm(total=self.num_rollouts, desc="Collecting rollouts")

        # Run each rollout
        for rollout_id in range(self.num_rollouts):
            logger.debug(f"Starting rollout {rollout_id + 1}/{self.num_rollouts}")

            # Set different random seed for each rollout to get diverse diffusion samples
            rollout_seed = hash(f"{self.snapshot.timestamp}_{rollout_id}") % (2**32)
            random.seed(rollout_seed)
            np.random.seed(rollout_seed)
            torch.manual_seed(rollout_seed)
            logger.debug(f"Rollout {rollout_id} using seed: {rollout_seed}")

            # Restore environment to initial state
            initial_obs = self.snapshot_manager.restore_state(self.env, self.snapshot)

            # Define step callback to collect trajectory data
            def step_callback(timestep, obs, action, reward, done, info):
                traj_data = TrajectoryData(
                    rollout_id=rollout_id,
                    timestep=timestep,
                    ee_position=obs.ee_pose[:3].copy(),
                    action_taken=action.trajectory[0].copy(),
                    reward=reward,
                    done=done
                )
                self.trajectories.append(traj_data)

            # Run episode using shared runner
            result = runner.run_episode(
                initial_obs=initial_obs,
                reset_env=False,  # Already restored from snapshot
                reset_policy=True,
                step_callback=step_callback,
                render=False
            )

            # Track success
            if result.success:
                self.success_count += 1

            # Update progress
            if show_progress:
                pbar.update(1)
                pbar.set_postfix({
                    'steps': result.episode_length,
                    'success': 'Yes' if result.success else 'No',
                    'success_rate': f'{self.success_count}/{rollout_id+1}'
                })

            logger.debug(f"Rollout {rollout_id + 1} complete: "
                        f"{result.episode_length} steps, success={result.success}")

        if show_progress:
            pbar.close()

        # Log summary
        success_rate = self.success_count / self.num_rollouts
        logger.info(f"Collection complete: {len(self.trajectories)} trajectory points collected")
        logger.info(f"  Success rate: {success_rate:.1%} ({self.success_count}/{self.num_rollouts})")

        return self.trajectories

    def get_rollout_trajectories(self, rollout_id: int) -> List[TrajectoryData]:
        """Get trajectory data for a specific rollout."""
        return [t for t in self.trajectories if t.rollout_id == rollout_id]

    def get_trajectories_by_rollout(self) -> List[List[TrajectoryData]]:
        """Group trajectories by rollout ID."""
        grouped = [[] for _ in range(self.num_rollouts)]
        for traj in self.trajectories:
            grouped[traj.rollout_id].append(traj)
        return grouped

    def get_summary_statistics(self) -> dict:
        """Compute summary statistics from collected trajectories."""
        grouped = self.get_trajectories_by_rollout()
        steps_per_rollout = [len(trajs) for trajs in grouped]
        all_positions = np.array([t.ee_position for t in self.trajectories])

        return {
            'num_rollouts': self.num_rollouts,
            'total_points': len(self.trajectories),
            'success_rate': self.success_count / self.num_rollouts,
            'avg_steps': np.mean(steps_per_rollout),
            'std_steps': np.std(steps_per_rollout),
            'spatial_extent': {
                'x_min': float(all_positions[:, 0].min()),
                'x_max': float(all_positions[:, 0].max()),
                'y_min': float(all_positions[:, 1].min()),
                'y_max': float(all_positions[:, 1].max()),
                'z_min': float(all_positions[:, 2].min()),
                'z_max': float(all_positions[:, 2].max()),
            }
        }

    def verify_same_initial_state(self, tolerance: float = 1e-6) -> bool:
        """Verify that all rollouts started from the same initial state."""
        grouped = self.get_trajectories_by_rollout()
        if len(grouped) < 2:
            return True

        first_positions = [trajs[0].ee_position for trajs in grouped if len(trajs) > 0]
        reference = first_positions[0]

        for pos in first_positions[1:]:
            if not np.allclose(pos, reference, atol=tolerance):
                logger.warning(f"Initial positions differ: {reference} vs {pos}")
                return False

        logger.debug("All rollouts started from same initial position")
        return True
