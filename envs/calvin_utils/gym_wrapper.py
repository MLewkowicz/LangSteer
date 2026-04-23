"""Gym wrapper for CALVIN environment.

Provides a simplified gym-compatible interface to CALVIN simulation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def _find_calvin_data_dir() -> Path:
    """Find calvin_env's data directory, resolving symlinks.

    calvin_env's setup.py doesn't include the data/ directory as package data,
    so it's not installed to site-packages. This function checks for a symlink
    or falls back to the uv git cache via dist-info metadata.

    Returns the resolved real path (not a symlink), since PyBullet's EGL
    renderer cannot follow symlinks in search paths.
    """
    import calvin_env
    pkg_dir = Path(calvin_env.__file__).parent

    # Check if data/ exists in installed package (possibly via symlink)
    data_dir = pkg_dir / "data"
    if data_dir.exists():
        return data_dir.resolve()

    # Fallback: find data in uv git cache via dist-info metadata
    for dist_info in pkg_dir.parent.glob("calvin_env*.dist-info"):
        direct_url_file = dist_info / "direct_url.json"
        if direct_url_file.exists():
            info = json.loads(direct_url_file.read_text())
            if "vcs_info" in info:
                commit = info["vcs_info"]["commit_id"]
                uv_cache = Path.home() / ".cache" / "uv" / "git-v0" / "checkouts"
                if uv_cache.exists():
                    for repo_dir in uv_cache.iterdir():
                        # uv uses short commit hashes as directory names
                        for checkout in repo_dir.iterdir():
                            if commit.startswith(checkout.name):
                                candidate = checkout / "data"
                                if candidate.exists():
                                    return candidate.resolve()

    raise FileNotFoundError(
        "calvin_env data directory not found. The package does not install its "
        "data/ directory. Try: ln -s $(find ~/.cache/uv -path '*/calvin_env*/data' "
        f"-type d | head -1) {pkg_dir}/data"
    )


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

            # Resolve calvin_env data directory (may be in uv git cache).
            # Must use resolved real path — PyBullet's EGL renderer cannot
            # follow symlinks in search paths.
            calvin_data_dir = _find_calvin_data_dir()
            logger.info(f"Resolved calvin_env data directory: {calvin_data_dir}")

            # Monkey patch Robot.load() to set calvin_env data as the search
            # path before loading URDF. Note: setAdditionalSearchPath is NOT
            # additive — each call overwrites the previous, so we set only
            # the calvin data dir (which contains all needed URDFs).
            original_load = Robot.load
            def patched_load(self):
                try:
                    p.setAdditionalSearchPath(calvin_data_dir.as_posix(), physicsClientId=self.cid)
                except Exception:
                    pass
                return original_load(self)
            Robot.load = patched_load

            if self.dataset_path is not None:
                from omegaconf import OmegaConf

                val_folder = Path(self.dataset_path) / self.split

                # Patch CALVIN config: fix data path and NumPy 2.0 incompatibility
                config_path = val_folder / '.hydra' / 'merged_config.yaml'
                if config_path.exists():
                    config = OmegaConf.load(config_path)

                    # Fix data_path — resolve symlinks for EGL compatibility
                    if 'data_path' in config:
                        configured_path = Path(config.data_path)
                        resolved = configured_path.resolve() if configured_path.exists() else calvin_data_dir
                        if str(resolved) != config.data_path:
                            config.data_path = resolved.as_posix()
                            logger.info(f"Resolved data_path to: {resolved}")

                    if 'env' in config and 'cameras' in config.env:
                        # Remove tactile camera if present
                        if 'tactile' in config.env.cameras:
                            del config.env.cameras['tactile']
                            logger.info("Removed tactile camera from config (NumPy 2.0 compatibility)")

                    # Save patched config
                    temp_config = val_folder / '.hydra' / 'merged_config_patched.yaml'
                    OmegaConf.save(config, temp_config)
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
