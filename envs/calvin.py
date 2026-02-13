"""CALVIN environment adapter."""

from typing import Tuple, Dict, Any
import logging
import numpy as np
from core.env import BaseEnvironment
from core.types import Observation, Action

logger = logging.getLogger(__name__)


class CalvinEnvironment(BaseEnvironment):
    """
    Adapter for CALVIN manipulation environment.
    Handles conversion between CALVIN's native format and standardized Observation/Action.
    Bridges CALVIN simulation with LangSteer's abstract environment interface.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)

        # Import CALVIN utilities
        from envs.calvin_utils.gym_wrapper import CalvinGymWrapper
        from envs.calvin_utils.language_ann import load_language_annotations, get_instruction_for_task
        from envs.calvin_utils.task_configs import get_initial_condition_for_task, get_env_state_for_initial_condition

        # Initialize CALVIN gym wrapper
        self._gym_env = CalvinGymWrapper(
            dataset_path=cfg.get('dataset_path'),
            split=cfg.get('split', 'validation'),
            use_gui=cfg.get('use_gui', False),
            num_points=cfg.get('num_points', 2048)
        )

        # Load language annotations
        ann_path = cfg.get('lang_ann_path')
        self._task_instructions = load_language_annotations(ann_path)

        # Set current task
        self._task_name = cfg.get('task', 'open_drawer')
        self._instruction = get_instruction_for_task(self._task_name, self._task_instructions)

        # Environment settings
        self._num_points = cfg.get('num_points', 2048)
        self._render_mode = cfg.get('render_mode')
        self._max_steps = cfg.get('max_steps', 360)
        self._current_step = 0

        # Task-specific reset configuration
        self._use_task_initial_condition = cfg.get('use_task_initial_condition', False)
        self._get_initial_condition_fn = get_initial_condition_for_task
        self._get_env_state_fn = get_env_state_for_initial_condition

        logger.info(f"CalvinEnvironment initialized")
        logger.info(f"  Task: {self._task_name}")
        logger.info(f"  Instruction: {self._instruction}")
        logger.info(f"  Max steps: {self._max_steps}")
        logger.info(f"  Use task-specific reset: {self._use_task_initial_condition}")

    def reset(self) -> Observation:
        """
        Resets the environment to an initial state and returns the first observation.
        If use_task_initial_condition is True, resets to task-specific scene state.

        Returns:
            Observation DTO with point cloud, proprioception, EE pose, and instruction
        """
        # Get task-specific initial condition if enabled
        robot_obs, scene_obs = None, None
        if self._use_task_initial_condition:
            initial_condition = self._get_initial_condition_fn(self._task_name)
            robot_obs, scene_obs = self._get_env_state_fn(initial_condition)
            logger.info(f"Resetting to task-specific state: {initial_condition}")

        # Reset CALVIN environment with optional scene state
        calvin_obs = self._gym_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        # Process observation to standardized format
        from envs.calvin_utils.observation import process_calvin_obs
        processed = process_calvin_obs(calvin_obs, self._num_points)

        # Create Observation DTO
        obs = Observation(
            rgb={'static': processed['rgb_static']},
            proprio=processed['robot_obs'],
            ee_pose=processed['ee_pose'],
            instruction=self._instruction,
            pcd=processed['point_cloud']
        )

        self._current_step = 0
        logger.debug(f"Environment reset complete")

        return obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Executes an action in the environment.

        Args:
            action: Action DTO with trajectory and gripper state

        Returns:
            observation: Observation DTO
            reward: Reward value (0 or 1 for CALVIN)
            done: Whether episode is finished
            info: Additional information dictionary
        """
        # Extract first action from trajectory (CALVIN expects single 7D action)
        # Action format: [x, y, z, euler_x, euler_y, euler_z, gripper]
        calvin_action = action.trajectory[0].copy()  # (7,) relative action

        # Convert continuous gripper action to binary (-1 or 1)
        # CALVIN expects: -1 = open, 1 = close
        # Convert: > 0 -> 1 (close), <= 0 -> -1 (open)
        calvin_action[6] = 1.0 if calvin_action[6] > 0 else -1.0

        # Step CALVIN environment
        calvin_obs, reward, done, info = self._gym_env.step(calvin_action)

        # Process observation
        from envs.calvin_utils.observation import process_calvin_obs
        processed = process_calvin_obs(calvin_obs, self._num_points)

        # Create Observation DTO
        obs = Observation(
            rgb={'static': processed['rgb_static']},
            proprio=processed['robot_obs'],
            ee_pose=processed['ee_pose'],
            instruction=self._instruction,
            pcd=processed['point_cloud']
        )

        # Update step counter and check for timeout
        self._current_step += 1
        if self._current_step >= self._max_steps:
            done = True
            info['timeout'] = True

        return obs, reward, done, info

    @property
    def task_description(self) -> str:
        """Get the current task language instruction."""
        return self._instruction

    def render(self, mode='rgb_array'):
        """
        Render environment state.

        Args:
            mode: Render mode ('rgb_array' or 'human')

        Returns:
            RGB image if mode='rgb_array', None otherwise
        """
        return self._gym_env.render(mode=mode)

    def close(self):
        """Close the environment and clean up resources."""
        if self._gym_env is not None:
            self._gym_env.close()
            logger.info("CalvinEnvironment closed")
