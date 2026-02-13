"""Gym wrapper for CALVIN environment.

Provides a simplified gym-compatible interface to CALVIN simulation.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class CalvinGymWrapper:
    """
    Gym-compatible wrapper for CALVIN environment.
    Handles environment initialization, reset, and stepping.
    """

    def __init__(self,
                 dataset_path: Optional[str] = None,
                 split: str = 'validation',
                 use_gui: bool = False,
                 num_points: int = 2048):
        """
        Args:
            dataset_path: Path to CALVIN dataset directory
            split: Dataset split ('training' or 'validation')
            use_gui: Whether to show PyBullet GUI
            num_points: Number of points in point cloud
        """
        self.dataset_path = dataset_path
        self.split = split
        self.use_gui = use_gui
        self.num_points = num_points
        self._env = None

        # Lazy import CALVIN environment
        self._initialize_env()

    def _initialize_env(self):
        """Initialize CALVIN environment with lazy import."""
        if self._env is not None:
            return

        try:
            # Lazy import to avoid dependency issues
            from calvin_env.envs.play_table_env import get_env
            from calvin_env.robot.robot import Robot
            import pybullet as p
            import pybullet_data

            # Monkey patch Robot.load() to add search path before loading URDF
            original_load = Robot.load
            def patched_load(self):
                # Set PyBullet data path before loading URDF
                try:
                    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
                except:
                    pass  # Ignore errors, might already be set
                return original_load(self)
            Robot.load = patched_load

            if self.dataset_path is not None:
                from pathlib import Path
                from omegaconf import OmegaConf

                val_folder = Path(self.dataset_path) / self.split

                # IMPORTANT: Remove tactile camera to avoid NumPy 2.0 incompatibility
                # This programmatically patches the CALVIN config without manual edits
                config_path = val_folder / '.hydra' / 'merged_config.yaml'
                if config_path.exists():
                    config = OmegaConf.load(config_path)
                    if 'env' in config and 'cameras' in config.env:
                        # Remove tactile camera if present
                        if 'tactile' in config.env.cameras:
                            del config.env.cameras['tactile']
                            logger.info("Removed tactile camera from config (NumPy 2.0 compatibility)")
                        # Save patched config temporarily
                        temp_config = val_folder / '.hydra' / 'merged_config_patched.yaml'
                        OmegaConf.save(config, temp_config)
                        # Temporarily rename files to use patched config
                        import shutil
                        config_backup = val_folder / '.hydra' / 'merged_config_original.yaml'
                        if not config_backup.exists():
                            shutil.copy(config_path, config_backup)
                        shutil.copy(temp_config, config_path)

                self._env = get_env(val_folder, show_gui=self.use_gui)
                logger.info(f"CALVIN environment initialized from: {val_folder}")
            else:
                logger.warning("No dataset path provided, CALVIN env may not work properly")
                # Try to create a basic environment
                self._env = get_env(None, show_gui=self.use_gui)

        except ImportError as e:
            logger.error(f"Failed to import CALVIN environment: {e}")
            logger.error("Please install calvin-env package or check CALVIN installation")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CALVIN environment: {e}")
            raise

    def reset(self, robot_obs: Optional[np.ndarray] = None, scene_obs: Optional[np.ndarray] = None) -> Dict:
        """
        Reset the environment to initial state.

        Args:
            robot_obs: Optional (15,) array for robot state
            scene_obs: Optional (24,) array for scene state

        Returns:
            Dictionary with CALVIN native observations:
                - 'rgb_obs': Dict with RGB images
                - 'depth_obs': Dict with depth images
                - 'robot_obs': Robot proprioception array
        """
        if self._env is None:
            self._initialize_env()

        try:
            if robot_obs is not None or scene_obs is not None:
                obs = self._env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            else:
                obs = self._env.reset()
            return obs
        except Exception as e:
            logger.error(f"Error resetting CALVIN environment: {e}")
            # Return dummy observation
            return self._get_dummy_observation()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: (7,) array with [x, y, z, euler_x, euler_y, euler_z, gripper]
                   Relative action in CALVIN format

        Returns:
            observation: Native CALVIN observation dict
            reward: Reward (typically 0 or 1 for success)
            done: Whether episode is finished
            info: Additional information dictionary
        """
        if self._env is None:
            self._initialize_env()

        try:
            obs, reward, done, info = self._env.step(action)
            return obs, reward, done, info
        except Exception as e:
            import traceback
            logger.error(f"Error stepping CALVIN environment: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return safe defaults
            return self._get_dummy_observation(), 0.0, True, {'error': str(e)}

    def render(self, mode='rgb_array'):
        """
        Render environment.

        Args:
            mode: Render mode ('rgb_array' or 'human')

        Returns:
            RGB image if mode='rgb_array', None otherwise
        """
        if self._env is None:
            return None

        try:
            if mode == 'rgb_array':
                # Extract RGB from latest observation
                if hasattr(self._env, 'render'):
                    return self._env.render(mode=mode)
                else:
                    # Fallback: return static camera view if available
                    return None
            elif mode == 'human':
                # GUI rendering handled by PyBullet
                return None
        except Exception as e:
            logger.error(f"Error rendering CALVIN environment: {e}")
            return None

    def close(self):
        """Close the environment and clean up resources."""
        if self._env is not None:
            try:
                if hasattr(self._env, 'close'):
                    self._env.close()
                logger.info("CALVIN environment closed")
            except Exception as e:
                logger.error(f"Error closing CALVIN environment: {e}")
            finally:
                self._env = None

    def _get_dummy_observation(self) -> Dict:
        """
        Get dummy observation for error handling.

        Returns:
            Dictionary with dummy CALVIN observations
        """
        return {
            'rgb_obs': {
                'rgb_static': np.zeros((200, 200, 3), dtype=np.uint8),
                'rgb_gripper': np.zeros((84, 84, 3), dtype=np.uint8),
            },
            'depth_obs': {
                'depth_static': np.zeros((200, 200), dtype=np.float32),
                'depth_gripper': np.zeros((84, 84), dtype=np.float32),
            },
            'robot_obs': np.zeros(15, dtype=np.float32),
        }

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
