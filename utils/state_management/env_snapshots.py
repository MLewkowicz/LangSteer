"""
Environment state snapshot management for reproducible rollouts.

Provides utilities to save and restore CALVIN environment states,
enabling multiple rollouts from identical initial configurations.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class EnvSnapshot:
    """Represents a saved environment state."""

    robot_obs: np.ndarray      # (15,) - joint positions, velocities, gripper
    scene_obs: np.ndarray      # Variable size - object poses, drawer/slider states
    task: str                  # Task name (e.g., "open_drawer")
    instruction: str           # Language instruction
    timestamp: float           # When snapshot was created

    def save(self, path: str) -> None:
        """
        Save snapshot to .npz file.

        Args:
            path: Output file path (.npz)
        """
        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save arrays and metadata
        np.savez_compressed(
            path,
            robot_obs=self.robot_obs,
            scene_obs=self.scene_obs,
            task=self.task,
            instruction=self.instruction,
            timestamp=self.timestamp
        )
        logger.info(f"Saved environment snapshot to {path}")

    @staticmethod
    def load(path: str) -> 'EnvSnapshot':
        """
        Load snapshot from .npz file.

        Args:
            path: Input file path (.npz)

        Returns:
            EnvSnapshot object with loaded state
        """
        data = np.load(path, allow_pickle=True)

        snapshot = EnvSnapshot(
            robot_obs=data['robot_obs'],
            scene_obs=data['scene_obs'],
            task=str(data['task']),
            instruction=str(data['instruction']),
            timestamp=float(data['timestamp'])
        )

        logger.info(f"Loaded environment snapshot from {path}")
        logger.info(f"  Task: {snapshot.task}")
        logger.info(f"  Instruction: {snapshot.instruction}")

        return snapshot


class EnvSnapshotManager:
    """Manages environment state capture and restoration."""

    def capture_state(self, env) -> EnvSnapshot:
        """
        Capture current environment state from CalvinEnvironment.

        Args:
            env: CalvinEnvironment instance

        Returns:
            EnvSnapshot with current state

        Raises:
            AttributeError: If environment doesn't support state capture
        """
        try:
            # Access underlying CALVIN environment
            # CalvinEnvironment -> CalvinGymWrapper -> CALVIN env
            calvin_env = env._gym_env._env

            # Get current observation which contains robot state
            current_obs = calvin_env.get_obs()

            # Extract robot state (proprioception)
            # CALVIN robot_obs is typically shape (15,) containing:
            # - Joint positions (7)
            # - Joint velocities (7)
            # - Gripper state (1)
            robot_obs = current_obs['robot_obs'].copy()

            # Get scene state (object poses, articulated states)
            # CALVIN scene_obs contains object positions, orientations, etc.
            try:
                # Try to get scene state from info
                info = calvin_env.get_info()
                scene_obs = info.get('scene_obs', np.array([]))

                # If scene_obs not in info, try to extract from state
                if scene_obs.size == 0:
                    # Fallback: get state directly from simulation
                    if hasattr(calvin_env, 'get_state'):
                        state = calvin_env.get_state()
                        scene_obs = state.get('scene_obs', np.array([]))
                    else:
                        # If no scene state available, use empty array
                        logger.warning("Could not capture scene state, using empty array")
                        scene_obs = np.array([])

            except Exception as e:
                logger.warning(f"Could not capture scene state: {e}")
                scene_obs = np.array([])

            # Create snapshot with current environment configuration
            snapshot = EnvSnapshot(
                robot_obs=robot_obs,
                scene_obs=scene_obs,
                task=env._task_name,
                instruction=env._instruction,
                timestamp=time.time()
            )

            logger.info("Captured environment state")
            logger.info(f"  Robot obs shape: {robot_obs.shape}")
            logger.info(f"  Scene obs shape: {scene_obs.shape}")

            return snapshot

        except AttributeError as e:
            logger.error(f"Failed to capture state: {e}")
            logger.error("Environment may not support state capture")
            raise

    def restore_state(self, env, snapshot: EnvSnapshot):
        """
        Reset environment to saved state.

        Args:
            env: CalvinEnvironment instance
            snapshot: EnvSnapshot with state to restore

        Returns:
            Observation DTO from reset

        Note:
            CalvinEnvironment.reset() already supports robot_obs and scene_obs
            parameters (see envs/calvin.py line 76).
        """
        try:
            # Restore task configuration
            env._task_name = snapshot.task
            env._instruction = snapshot.instruction

            # Reset environment to saved state
            # CalvinEnvironment.reset() passes these to CalvinGymWrapper.reset()
            # which passes them to the underlying CALVIN environment
            robot_obs = snapshot.robot_obs if snapshot.robot_obs.size > 0 else None
            scene_obs = snapshot.scene_obs if snapshot.scene_obs.size > 0 else None

            logger.debug(f"Restoring state: robot_obs shape={robot_obs.shape if robot_obs is not None else None}, "
                        f"scene_obs shape={scene_obs.shape if scene_obs is not None else None}")

            # Reset environment with saved state
            # This resets the CALVIN simulation to the exact captured configuration
            obs = env.reset()  # Note: CalvinEnvironment.reset() doesn't currently accept these args

            # WORKAROUND: If CalvinEnvironment.reset() doesn't accept args, directly reset gym wrapper
            if robot_obs is not None or scene_obs is not None:
                calvin_obs = env._gym_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

                # Process observation to standardized format
                from envs.calvin_utils.observation import process_calvin_obs
                from core.types import Observation

                processed = process_calvin_obs(calvin_obs, env._num_points)

                obs = Observation(
                    rgb={'static': processed['rgb_static']},
                    proprio=processed['robot_obs'],
                    ee_pose=processed['ee_pose'],
                    instruction=snapshot.instruction,
                    pcd=processed['point_cloud']
                )

            logger.info("Restored environment to snapshot state")
            return obs

        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            logger.warning("Falling back to standard reset")
            return env.reset()

    def verify_state_match(self, env, snapshot: EnvSnapshot, tolerance: float = 1e-6) -> bool:
        """
        Verify that current environment state matches snapshot.

        Args:
            env: CalvinEnvironment instance
            snapshot: EnvSnapshot to compare against
            tolerance: Numerical tolerance for floating point comparison

        Returns:
            True if states match within tolerance
        """
        try:
            current_snapshot = self.capture_state(env)

            # Compare robot state
            robot_match = np.allclose(
                current_snapshot.robot_obs,
                snapshot.robot_obs,
                atol=tolerance
            )

            # Compare scene state if available
            scene_match = True
            if snapshot.scene_obs.size > 0 and current_snapshot.scene_obs.size > 0:
                scene_match = np.allclose(
                    current_snapshot.scene_obs,
                    snapshot.scene_obs,
                    atol=tolerance
                )

            if robot_match and scene_match:
                logger.debug("State verification passed")
                return True
            else:
                logger.warning(f"State verification failed: robot_match={robot_match}, scene_match={scene_match}")
                return False

        except Exception as e:
            logger.error(f"Error during state verification: {e}")
            return False
