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
        from envs.calvin_utils.language_ann import (
            load_language_annotations,
            load_perturbed_annotations,
            get_instruction_for_task,
        )
        from envs.calvin_utils.task_configs import get_initial_condition_for_task, get_env_state_for_initial_condition

        # Initialize CALVIN gym wrapper
        self._gym_env = CalvinGymWrapper(
            dataset_path=cfg.get('dataset_path'),
            split=cfg.get('split', 'validation'),
            use_gui=cfg.get('use_gui', False),
            num_points=cfg.get('num_points', 2048)
        )

        # Load language annotations (optionally overriding with a perturbed set for
        # a specific axis P1..P4). When perturbed_ann_path + perturbation_axis are
        # both set, instructions come from that file instead of auto_lang_ann.npy.
        self._perturbation_axis = cfg.get('perturbation_axis')
        perturbed_ann_path = cfg.get('perturbed_ann_path')
        if self._perturbation_axis is not None and perturbed_ann_path:
            self._task_instructions = load_perturbed_annotations(
                perturbed_ann_path, self._perturbation_axis
            )
            logger.info(f"Using perturbed annotations (axis={self._perturbation_axis})")
        else:
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
        from envs.calvin_utils.gym_wrapper import _find_calvin_data_dir
        tasks_cfg_path = (
            _find_calvin_data_dir().parent / "conf" / "tasks" / "new_playtable_tasks.yaml"
        )
        tasks_cfg = OmegaConf.load(str(tasks_cfg_path))
        self._task_oracle = Tasks(tasks_cfg.tasks)
        self._start_info = None

        # Optional per-waypoint render callback for video recording.
        # Set via set_waypoint_render_fn(); called with raw calvin_obs after each waypoint.
        self._waypoint_render_fn = None

        # Articulation-tracking offsets, populated lazily on first
        # _get_fixture_positions() call (needs env to be fully loaded).
        self._fixture_frame_offsets: Optional[Dict[str, np.ndarray]] = None

        logger.info(f"CalvinEnvironment initialized")
        logger.info(f"  Task: {self._task_name}")
        logger.info(f"  Instruction: {self._instruction}")
        logger.info(f"  Max steps: {self._max_steps}")
        logger.info(f"  Use task-specific reset: {self._use_task_initial_condition}")
        logger.info(f"  Provide PCD images: {self._provide_pcd_images}")
        logger.info(f"  Randomize initial condition: {self._randomize_initial_condition}")
        if self._randomize_initial_condition and self._task_name in self._task_episode_ids:
            logger.info(f"  Available starting episodes for '{self._task_name}': {len(self._task_episode_ids[self._task_name])}")

    def set_task(self, task_name: str) -> None:
        """Switch to a different task without re-creating the environment.

        Updates task name, instruction, and oracle check target.
        The underlying PyBullet environment and gym wrapper are reused.
        """
        from envs.calvin_utils.language_ann import get_instruction_for_task

        self._task_name = task_name
        self._instruction = get_instruction_for_task(task_name, self._task_instructions)
        logger.info(f"Switched task to: {task_name} -> '{self._instruction}'")

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

            if self._waypoint_render_fn is not None:
                self._waypoint_render_fn(calvin_obs)

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

    def set_waypoint_render_fn(self, fn):
        """
        Register a callback invoked after each individual waypoint inside step().

        Args:
            fn: Callable(calvin_obs) -> None, or None to disable.
                calvin_obs is the raw CALVIN observation dict with 'rgb_obs', 'depth_obs',
                and 'robot_obs' keys, as returned by the gym wrapper after each sub-step.
        """
        self._waypoint_render_fn = fn

    def render_high_res_gripper(self, width: int, height: int) -> np.ndarray:
        """
        Render the gripper camera at an arbitrary resolution using PyBullet directly.

        The gripper camera view matrix is recomputed dynamically each call
        since the camera is mounted on the robot arm.

        Args:
            width: Desired output width in pixels.
            height: Desired output height in pixels.

        Returns:
            (height, width, 3) uint8 RGB array.
        """
        import pybullet as p
        from training.policies.diffuser_actor.preprocessing.calvin_utils import get_gripper_camera_view_matrix
        cam = self._gym_env._env.cameras[1]
        view_matrix = get_gripper_camera_view_matrix(cam)
        proj_matrix = p.computeProjectionMatrixFOV(
            cam.fov, width / height, cam.nearval, cam.farval,
            physicsClientId=cam.cid
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            physicsClientId=cam.cid
        )
        return np.array(rgba, dtype=np.uint8)[:, :, :3]

    def render_high_res_static(self, width: int, height: int) -> np.ndarray:
        """
        Render the static overhead camera at an arbitrary resolution using PyBullet directly.

        This is independent of the policy's camera (200×200) and does not affect inference.

        Args:
            width: Desired output width in pixels.
            height: Desired output height in pixels.

        Returns:
            (height, width, 3) uint8 RGB array.
        """
        import pybullet as p
        cam = self._gym_env._env.cameras[0]
        proj_matrix = p.computeProjectionMatrixFOV(
            cam.fov, width / height, cam.nearval, cam.farval,
            physicsClientId=cam.cid
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width, height, cam.viewMatrix, proj_matrix,
            physicsClientId=cam.cid
        )
        return np.array(rgba, dtype=np.uint8)[:, :, :3]

    # ------------------------------------------------------------------
    # Scene state (for VoxPoser steering)
    # ------------------------------------------------------------------

    # Playtable link indices → fixture names (CALVIN playtable UID=5)
    _PLAYTABLE_UID = 5
    _LINK_FIXTURES = {
        0: 'button',
        1: 'switch',       # also 'light_switch'
        2: 'slider',
        3: 'drawer',
        4: 'led',
        5: 'lightbulb',
    }
    # Hardcoded size overrides with articulation tracking. PyBullet's link AABB
    # for `slide_link`, `drawer_link`, and similar includes the entire cabinet
    # mesh (frames, runners, interior walls) not just the visible interactive
    # surface, producing bboxes far too large. For these links we use a
    # hand-tuned size AND track position via (live link frame origin) + (const
    # offset). The offset is computed once by briefly resetting the joint to 0
    # to capture the link frame origin at rest (see _compute_fixture_frame_offsets).
    # Because all playtable joints are prismatic, the offset is a world-frame
    # constant that's added to worldLinkFramePosition at query time.
    _FIXTURE_AABB_OVERRIDES = {
        'slider': {
            'rest_position': np.array([0.040, 0.040, 0.555]),
            'size':          np.array([0.289, 0.10, 0.04]),
        },
        'drawer': {
            'rest_position': np.array([0.180, -0.100, 0.350]),
            'size':          np.array([0.15, 0.25, 0.08]),
        },
        'switch': {
            'rest_position': np.array([0.300, 0.037, 0.518]),
            'size':          np.array([0.06, 0.06, 0.06]),
        },
        'button': {
            'rest_position': np.array([-0.120, -0.120, 0.472]),
            'size':          np.array([0.07, 0.07, 0.03]),
        },
    }
    # Derived fixtures: small grasp regions or aliases computed from a parent link.
    # Each entry is {parent: link name, offset: (3,) from parent center, size: (3,) box
    # extent — or None to inherit parent size}.
    _DERIVED_OFFSETS = {
        'drawer_handle': {
            'parent': 'drawer',
            'offset': np.array([0.0, -0.145, 0.0]),    # flush with drawer front face
            'size':   np.array([0.11, 0.04, 0.03]),    # horizontal pull bar
        },
        'slider_handle': {
            'parent': 'slider',
            'offset': np.array([0.0, -0.05, 0.0]),
            'size':   np.array([0.03, 0.04, 0.11]),   # vertical groove
        },
        'light_switch': {
            'parent': 'switch',
            'offset': np.array([0.0, 0.0, 0.0]),
            'size':   None,
        },
    }

    def get_scene_state(self) -> Dict[str, Any]:
        """Return current robot obs, scene obs, and live fixture positions.

        Used by VoxPoser steering to refresh costmaps with current object
        positions.
        """
        calvin_obs = self._gym_env._env.get_obs()
        return {
            'robot_obs': calvin_obs.get('robot_obs', np.zeros(15)),
            'scene_obs': calvin_obs.get('scene_obs', np.zeros(24)),
            'fixture_positions': self._get_fixture_positions(),
            'block_aabbs': self._get_block_aabbs(),
        }

    def _get_fixture_positions(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Query PyBullet for current fixture positions.

        For overridden fixtures (slider, drawer, switch, button), uses
        worldLinkFramePosition + const offset to track articulation with a
        tight hand-tuned size. For other fixtures (led, lightbulb), uses the
        live per-link AABB which is already accurate for small static parts.

        Returns:
            Dict mapping fixture name → {'position': (3,), 'size': (3,)}
        """
        import pybullet as p
        cid = self._gym_env._env.cameras[0].cid

        # Lazily compute link-frame-origin → visible-center offsets once per env.
        if getattr(self, '_fixture_frame_offsets', None) is None:
            self._compute_fixture_frame_offsets()

        link_data = {}
        for link_idx, name in self._LINK_FIXTURES.items():
            if name in self._FIXTURE_AABB_OVERRIDES:
                override = self._FIXTURE_AABB_OVERRIDES[name]
                link_state = p.getLinkState(
                    self._PLAYTABLE_UID, link_idx,
                    computeForwardKinematics=1, physicsClientId=cid,
                )
                frame_origin = np.asarray(link_state[4], dtype=np.float32)
                offset = self._fixture_frame_offsets.get(
                    name,
                    override['rest_position'] - frame_origin,  # fallback
                )
                link_data[name] = {
                    'position': (frame_origin + offset).astype(np.float32),
                    'size':     override['size'].astype(np.float32).copy(),
                }
                continue
            aabb_min, aabb_max = p.getAABB(
                self._PLAYTABLE_UID, link_idx, physicsClientId=cid
            )
            center = (np.array(aabb_min) + np.array(aabb_max)) / 2
            size = np.array(aabb_max) - np.array(aabb_min)
            link_data[name] = {'position': center, 'size': size}

        # Compute derived fixture positions from parent link + offset, honoring
        # an explicit size override when provided (e.g. a small handle on a large door).
        for derived_name, spec in self._DERIVED_OFFSETS.items():
            parent = link_data.get(spec['parent'])
            if parent is None:
                continue
            size = spec['size'] if spec['size'] is not None else parent['size'].copy()
            link_data[derived_name] = {
                'position': parent['position'] + spec['offset'],
                'size':     np.asarray(size, dtype=np.float32),
            }

        return link_data

    def _compute_fixture_frame_offsets(self) -> None:
        """One-time calibration: compute (rest_position − rest_frame_origin) per fixture.

        For each prismatic-articulated fixture in _FIXTURE_AABB_OVERRIDES, we
        briefly resetJointState to 0, query worldLinkFramePosition at rest, then
        restore the original joint state. The resulting offset is a world-frame
        constant (orientations don't change for prismatic joints) that we add
        to the live link frame origin to get the current bbox center.

        resetJointState does not step physics, so this is non-destructive.
        """
        import pybullet as p
        cid = self._gym_env._env.cameras[0].cid
        self._fixture_frame_offsets = {}

        for link_idx, name in self._LINK_FIXTURES.items():
            if name not in self._FIXTURE_AABB_OVERRIDES:
                continue
            override = self._FIXTURE_AABB_OVERRIDES[name]
            try:
                # Snapshot current joint state
                js = p.getJointState(self._PLAYTABLE_UID, link_idx, physicsClientId=cid)
                saved_pos, saved_vel = js[0], js[1]

                # Force to rest (joint=0) and query
                p.resetJointState(
                    self._PLAYTABLE_UID, link_idx,
                    targetValue=0.0, targetVelocity=0.0,
                    physicsClientId=cid,
                )
                link_state = p.getLinkState(
                    self._PLAYTABLE_UID, link_idx,
                    computeForwardKinematics=1, physicsClientId=cid,
                )
                rest_frame_origin = np.asarray(link_state[4], dtype=np.float32)

                # Restore
                p.resetJointState(
                    self._PLAYTABLE_UID, link_idx,
                    targetValue=saved_pos, targetVelocity=saved_vel,
                    physicsClientId=cid,
                )

                self._fixture_frame_offsets[name] = (
                    override['rest_position'].astype(np.float32) - rest_frame_origin
                )
            except Exception as e:
                logger.warning(
                    f"Could not calibrate frame offset for '{name}' (link {link_idx}): {e}"
                )

    def _get_block_aabbs(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return live orientation-aware AABBs for movable blocks via PyBullet.

        Returns:
            Dict mapping canonical block name ('red_block', 'blue_block',
            'pink_block') → {'aabb_min': (3,), 'aabb_max': (3,), 'position': (3,)}.
            Empty dict if the scene exposes no movable objects.
        """
        import pybullet as p
        cid = self._gym_env._env.cameras[0].cid
        scene = getattr(self._gym_env._env, 'scene', None)
        if scene is None:
            return {}

        out: Dict[str, Dict[str, np.ndarray]] = {}
        for obj in getattr(scene, 'movable_objects', []):
            name = obj.name.lower()
            if 'red' in name:
                key = 'red_block'
            elif 'blue' in name:
                key = 'blue_block'
            elif 'pink' in name:
                key = 'pink_block'
            else:
                continue
            aabb_min, aabb_max = p.getAABB(obj.uid, -1, physicsClientId=cid)
            aabb_min = np.asarray(aabb_min, dtype=np.float32)
            aabb_max = np.asarray(aabb_max, dtype=np.float32)
            out[key] = {
                'aabb_min': aabb_min,
                'aabb_max': aabb_max,
                'position': (aabb_min + aabb_max) / 2,
            }
        return out

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
