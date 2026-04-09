"""CALVIN environment adapter."""

from typing import Tuple, Dict, Any, List, Optional
import logging
import random
from pathlib import Path
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
        self._done_on_success = cfg.get('done_on_success', True)
        self._current_step = 0

        # Per-pixel PCD images for 3D Diffuser Actor
        self._provide_pcd_images = cfg.get('provide_pcd_images', False)

        # Task-specific reset configuration
        self._use_task_initial_condition = cfg.get('use_task_initial_condition', False)
        self._get_initial_condition_fn = get_initial_condition_for_task
        self._get_env_state_fn = get_env_state_for_initial_condition

        # Random initial condition sampling from dataset episodes
        self._randomize_initial_condition = cfg.get('randomize_initial_condition', False)
        self._task_episode_ids: Dict[str, List[int]] = {}
        self._dataset_path = Path(cfg.get('dataset_path'))
        self._split = cfg.get('split', 'validation')
        if self._randomize_initial_condition:
            self._task_episode_ids = self._build_task_episode_index()

        # Initialize task oracle for success detection
        from calvin_env.envs.tasks import Tasks
        from omegaconf import OmegaConf
        import importlib.resources as pkg_resources
        tasks_cfg_path = (
            pkg_resources.files("conf") / "tasks" / "new_playtable_tasks.yaml"
        )
        tasks_cfg = OmegaConf.load(str(tasks_cfg_path))
        self._task_oracle = Tasks(tasks_cfg.tasks)
        self._start_info = None

        logger.info(f"CalvinEnvironment initialized")
        logger.info(f"  Task: {self._task_name}")
        logger.info(f"  Instruction: {self._instruction}")
        logger.info(f"  Max steps: {self._max_steps}")
        logger.info(f"  Use task-specific reset: {self._use_task_initial_condition}")
        logger.info(f"  Provide PCD images: {self._provide_pcd_images}")
        logger.info(f"  Randomize initial condition: {self._randomize_initial_condition}")
        if self._randomize_initial_condition and self._task_name in self._task_episode_ids:
            logger.info(f"  Available starting episodes for '{self._task_name}': {len(self._task_episode_ids[self._task_name])}")

    def _build_task_episode_index(self) -> Dict[str, List[int]]:
        """Build index mapping task names to their starting episode IDs in the dataset."""
        ann_path = self._dataset_path / self._split / "lang_annotations" / "auto_lang_ann.npy"
        ann = np.load(str(ann_path), allow_pickle=True).item()
        tasks = ann['language']['task']
        start_end = ann['info']['indx']  # list of (start_frame, end_frame) tuples

        task_index: Dict[str, List[int]] = {}
        for i, task_name in enumerate(tasks):
            if task_name not in task_index:
                task_index[task_name] = []
            task_index[task_name].append(start_end[i][0])  # start frame = episode ID

        logger.info(f"Built episode index for {len(task_index)} tasks")
        return task_index

    def _sample_random_initial_condition(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random starting state from dataset episodes for the current task."""
        episode_ids = self._task_episode_ids.get(self._task_name, [])
        if not episode_ids:
            logger.warning(f"No episodes for task '{self._task_name}', falling back to task-specific reset")
            initial_condition = self._get_initial_condition_fn(self._task_name)
            return self._get_env_state_fn(initial_condition)

        ep_id = random.choice(episode_ids)
        ep_path = self._dataset_path / self._split / f"episode_{ep_id:07d}.npz"
        data = np.load(str(ep_path))
        robot_obs = data['robot_obs'].astype(np.float32)
        scene_obs = data['scene_obs'].astype(np.float32)
        logger.info(f"Sampled random initial condition from episode {ep_id} for task '{self._task_name}'")
        return robot_obs, scene_obs

    def _process_obs(self, calvin_obs: Dict) -> Observation:
        """Convert raw CALVIN observation to Observation DTO."""
        if self._provide_pcd_images:
            from envs.calvin_utils.observation import prepare_visual_states
            cameras = self._gym_env._env.cameras
            processed = prepare_visual_states(calvin_obs, cameras)
            return Observation(
                rgb={
                    'static': processed['rgb_static'],
                    'gripper': processed['rgb_gripper'],
                },
                proprio=processed['robot_obs'],
                ee_pose=processed['ee_pose'],
                instruction=self._instruction,
                depth={
                    'static': processed['pcd_static'],
                    'gripper': processed['pcd_gripper'],
                },
            )
        else:
            from envs.calvin_utils.observation import process_calvin_obs
            processed = process_calvin_obs(calvin_obs, self._num_points)
            return Observation(
                rgb={'static': processed['rgb_static']},
                proprio=processed['robot_obs'],
                ee_pose=processed['ee_pose'],
                instruction=self._instruction,
                pcd=processed['point_cloud'],
            )

    def reset(self, robot_obs=None, scene_obs=None) -> Observation:
        """
        Resets the environment to an initial state and returns the first observation.

        Args:
            robot_obs: Optional (15,) array for robot state override (from reference trajectory)
            scene_obs: Optional (24,) array for scene state override (from reference trajectory)

        Returns:
            Observation DTO with point cloud, proprioception, EE pose, and instruction
        """
        # Priority: explicit override > random sampling > task-specific > CALVIN default
        if robot_obs is not None or scene_obs is not None:
            logger.info("Resetting to provided robot/scene state (reference trajectory)")
        elif self._randomize_initial_condition:
            robot_obs, scene_obs = self._sample_random_initial_condition()
        elif self._use_task_initial_condition:
            initial_condition = self._get_initial_condition_fn(self._task_name)
            robot_obs, scene_obs = self._get_env_state_fn(initial_condition)
            logger.info(f"Resetting to task-specific state: {initial_condition}")

        # Reset CALVIN environment with optional scene state
        calvin_obs = self._gym_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        obs = self._process_obs(calvin_obs)

        self._current_step = 0
        self._start_info = self._gym_env._env.get_info()
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
        # Execute ALL waypoints in the trajectory before returning.
        # The Diffuser Actor predicts a multi-step trajectory (e.g. 20 waypoints).
        # The original 3DA evaluation executes all waypoints per prediction
        # (EXECUTE_LEN=20), then re-predicts with fresh observations.
        #
        # CALVIN's Robot.apply_action dispatches on action type:
        #   flat (7,) array → treated as RELATIVE (scaled by max_rel_pos/orn)
        #   list of 3 arrays → treated as ABSOLUTE target pose
        # The Diffuser Actor outputs absolute poses, so we send list-of-3.
        total_reward = 0.0
        done = False
        info = {}

        for act_ind in range(action.trajectory.shape[0]):
            act = action.trajectory[act_ind]
            gripper_binary = np.array([1.0 if act[6] > 0 else -1.0])
            calvin_action = [act[:3].copy(), act[3:6].copy(), gripper_binary]

            calvin_obs, reward, done, info = self._gym_env.step(calvin_action)
            total_reward += reward

            self._current_step += 1
            if self._current_step >= self._max_steps:
                done = True
                info['timeout'] = True
                break

            if done:
                break

        # Check task success via oracle
        if self._start_info is not None and self._task_name in self._task_oracle.tasks:
            end_info = self._gym_env._env.get_info()
            achieved = self._task_oracle.get_task_info_for_set(
                self._start_info, end_info, {self._task_name}
            )
            if self._task_name in achieved:
                info['success'] = True
                if self._done_on_success:
                    done = True

        # Process final observation
        obs = self._process_obs(calvin_obs)

        return obs, total_reward, done, info

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
