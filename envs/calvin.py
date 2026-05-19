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

    def get_static_camera_matrices(
        self, width: int, height: int, fov: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (view_matrix, projection_matrix) used by `render_high_res_static`.

        Task 7 Phase 2 — needed by `voxposer/scene_image.py` to project world-frame
        OBB corners to pixel coords for annotation. The PyBullet camera tuple is
        column-major; we reshape into (4, 4) and return as numpy arrays so the
        caller can do `pixels = (P @ V @ world_h)` math.
        """
        import pybullet as p
        cam = self._gym_env._env.cameras[0]
        effective_fov = fov if fov is not None else cam.fov
        view_matrix = np.array(cam.viewMatrix, dtype=np.float64).reshape(4, 4, order="F")
        proj_matrix = np.array(
            p.computeProjectionMatrixFOV(
                effective_fov, width / height, cam.nearval, cam.farval,
                physicsClientId=cam.cid,
            ),
            dtype=np.float64,
        ).reshape(4, 4, order="F")
        return view_matrix, proj_matrix

    def render_high_res_static(self, width: int, height: int, fov: float | None = None) -> np.ndarray:
        """
        Render the static overhead camera at an arbitrary resolution using PyBullet directly.

        This is independent of the policy's camera (200×200) and does not affect inference.

        Args:
            width: Desired output width in pixels.
            height: Desired output height in pixels.
            fov: Optional field of view override in degrees. If None, uses the camera's native FOV.

        Returns:
            (height, width, 3) uint8 RGB array.
        """
        import pybullet as p
        cam = self._gym_env._env.cameras[0]
        effective_fov = fov if fov is not None else cam.fov
        proj_matrix = p.computeProjectionMatrixFOV(
            effective_fov, width / height, cam.nearval, cam.farval,
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

    # Playtable link indices → default canonical fixture name (CALVIN playtable UID=5).
    # Used only as a fallback for links without any entry in _FIXTURE_AABB_OVERRIDES
    # (led, lightbulb), which fall back to live PyBullet per-link AABBs.
    _PLAYTABLE_UID = 5
    _LINK_FIXTURES = {
        0: 'button',
        1: 'switch',
        2: 'slider',
        3: 'drawer',
        4: 'led',
        5: 'lightbulb',
    }
    # Hand-tuned OBB overrides with articulation tracking. PyBullet's per-link
    # AABB for `slide_link`, `drawer_link`, etc. includes the entire cabinet
    # mesh (frames, runners, interior walls), not just the interactive surface
    # — we override those with tight boxes. Position is tracked via (live link
    # frame origin) + (const offset); the offset is calibrated once by briefly
    # resetting the joint to 0 (see _compute_fixture_frame_offsets). All
    # playtable joints are prismatic, so the offset is a world-frame constant.
    #
    # `link_idx` is the playtable link this override tracks. Multiple overrides
    # CAN share a link (e.g. switch housing + light_switch lever both on link 1).
    # `euler_xyz_deg` is the OBB's world-frame rotation (XYZ intrinsic, degrees);
    # [0,0,0] reproduces the pre-rotation AABB behavior exactly.
    #
    # Values from the 2026-04-23 annotator pass (tmp/bbox_annotator/overrides_20260423_164037.json).
    _FIXTURE_AABB_OVERRIDES = {
        'slider': {
            'link_idx': 2,
            'rest_position': np.array([0.040, 0.065, 0.538]),
            'size':          np.array([0.280, 0.018, 0.218]),
            'euler_xyz_deg': np.array([-30.50, 0.0, 0.0]),
        },
        'drawer': {
            'link_idx': 3,
            'rest_position': np.array([0.178, -0.008, 0.354]),
            'size':          np.array([0.445, 0.345, 0.102]),
            'euler_xyz_deg': np.array([0.0, 0.0, 0.0]),
        },
        'switch': {
            'link_idx': 1,
            'rest_position': np.array([0.296, 0.039, 0.499]),
            'size':          np.array([0.06, 0.06, 0.06]),
            'euler_xyz_deg': np.array([0.0, 0.0, 0.0]),
        },
        'button': {
            'link_idx': 0,
            'rest_position': np.array([-0.120, -0.120, 0.472]),
            'size':          np.array([0.07, 0.07, 0.03]),
            'euler_xyz_deg': np.array([0.0, 0.0, 0.0]),
        },
        # light_switch shares playtable link 1 with 'switch' but describes the
        # tilted toggle lever rather than the housing — distinct geometry, so
        # it's a sibling top-level fixture, not a derived offset.
        'light_switch': {
            'link_idx': 1,
            'rest_position': np.array([0.302, 0.037, 0.518]),
            'size':          np.array([0.118, 0.061, 0.031]),
            'euler_xyz_deg': np.array([-31.50, 0.0, 0.0]),
        },
    }
    # Derived fixtures: small grasp regions computed from a parent link.
    # `offset` and `euler_xyz_deg` are in the parent's local frame, so a handle
    # follows its parent's rotation:
    #     world_center   = parent_center + R_parent @ offset
    #     world_rotation = R_parent @ R_local
    _DERIVED_OFFSETS = {
        'drawer_handle': {
            'parent': 'drawer',
            'offset':        np.array([0.002, -0.212, 0.006]),
            'size':          np.array([0.242, 0.077, 0.036]),
            'euler_xyz_deg': np.array([0.0, 0.0, 0.0]),
        },
        'slider_handle': {
            'parent': 'slider',
            'offset':        np.array([0.0, -0.038, 0.002]),
            'size':          np.array([0.034, 0.066, 0.161]),
            'euler_xyz_deg': np.array([0.0, 0.0, 0.0]),
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
            Dict mapping fixture name → {'position': (3,), 'size': (3,),
            'rotation': (3, 3)}. `rotation` is the OBB's world-frame rotation
            matrix, identity for non-rotated entries.
        """
        import pybullet as p
        from scipy.spatial.transform import Rotation
        cid = self._gym_env._env.cameras[0].cid

        # Lazily compute link-frame-origin → visible-center offsets once per env.
        if getattr(self, '_fixture_frame_offsets', None) is None:
            self._compute_fixture_frame_offsets()

        link_data = {}
        # 1. Overridden fixtures (iterate by name so multiple overrides on the
        #    same link — e.g. switch + light_switch both on link 1 — each get
        #    their own entry).
        for name, override in self._FIXTURE_AABB_OVERRIDES.items():
            link_idx = override['link_idx']
            link_state = p.getLinkState(
                self._PLAYTABLE_UID, link_idx,
                computeForwardKinematics=1, physicsClientId=cid,
            )
            frame_origin = np.asarray(link_state[4], dtype=np.float32)
            offset = self._fixture_frame_offsets.get(
                name,
                override['rest_position'] - frame_origin,  # fallback
            )
            euler = np.asarray(
                override.get('euler_xyz_deg', [0.0, 0.0, 0.0]), dtype=np.float32,
            )
            R_world = Rotation.from_euler('xyz', euler, degrees=True).as_matrix()
            link_data[name] = {
                'position': (frame_origin + offset).astype(np.float32),
                'size':     override['size'].astype(np.float32).copy(),
                'rotation': R_world.astype(np.float32),
            }

        # 2. Non-overridden fixtures (led, lightbulb) use the live per-link AABB.
        #    Skip links covered by any override above.
        covered_links = {o['link_idx'] for o in self._FIXTURE_AABB_OVERRIDES.values()}
        for link_idx, name in self._LINK_FIXTURES.items():
            if link_idx in covered_links:
                continue
            aabb_min, aabb_max = p.getAABB(
                self._PLAYTABLE_UID, link_idx, physicsClientId=cid
            )
            center = (np.array(aabb_min) + np.array(aabb_max)) / 2
            size = np.array(aabb_max) - np.array(aabb_min)
            link_data[name] = {
                'position': center.astype(np.float32),
                'size':     size.astype(np.float32),
                'rotation': np.eye(3, dtype=np.float32),
            }

        # Compute derived fixture positions from parent link + local offset,
        # applying the parent's rotation so handles follow tilted fixtures.
        # world_center = parent_center + R_parent @ local_offset
        # world_rotation = R_parent @ R_local
        for derived_name, spec in self._DERIVED_OFFSETS.items():
            parent = link_data.get(spec['parent'])
            if parent is None:
                continue
            size = spec['size'] if spec['size'] is not None else parent['size'].copy()
            R_parent = parent['rotation']
            local_offset = np.asarray(spec['offset'], dtype=np.float32)
            local_euler = np.asarray(
                spec.get('euler_xyz_deg', [0.0, 0.0, 0.0]), dtype=np.float32,
            )
            R_local = Rotation.from_euler(
                'xyz', local_euler, degrees=True,
            ).as_matrix().astype(np.float32)
            link_data[derived_name] = {
                'position': (parent['position'] + R_parent @ local_offset).astype(np.float32),
                'size':     np.asarray(size, dtype=np.float32),
                'rotation': (R_parent @ R_local).astype(np.float32),
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

        for name, override in self._FIXTURE_AABB_OVERRIDES.items():
            link_idx = override['link_idx']
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

    # Per-block canonical local-frame size (meters). Pink is non-cubic.
    _BLOCK_LOCAL_SIZES = {
        'red_block':  np.array([0.05, 0.05, 0.05]),
        'blue_block': np.array([0.05, 0.05, 0.05]),
        'pink_block': np.array([0.07, 0.05, 0.05]),
    }

    def _get_block_aabbs(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return live orientation-aware OBBs for movable blocks via PyBullet.

        Each block's pose is queried via `getBasePositionAndOrientation`, and
        the local-frame size comes from `_BLOCK_LOCAL_SIZES`. The world-axis
        envelope (`aabb_min`/`aabb_max`) is also included as the tight world
        AABB of the 8 rotated corners — useful for LLM prompts that do
        directional reasoning like "max_z of the block".

        Returns:
            Dict mapping canonical block name → {'position': (3,), 'size': (3,),
            'rotation': (3, 3), 'aabb_min': (3,), 'aabb_max': (3,)}. Empty
            dict if the scene exposes no movable objects.
        """
        import pybullet as p
        from scipy.spatial.transform import Rotation
        cid = self._gym_env._env.cameras[0].cid
        scene = getattr(self._gym_env._env, 'scene', None)
        if scene is None:
            return {}

        # Unit cube corners in local frame (each coord ∈ {-0.5, +0.5}).
        _LOCAL_UNIT = np.array([
            [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5],
            [-0.5, +0.5, -0.5], [+0.5, +0.5, -0.5],
            [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5],
            [-0.5, +0.5, +0.5], [+0.5, +0.5, +0.5],
        ], dtype=np.float32)

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

            (px, py, pz), quat_xyzw = p.getBasePositionAndOrientation(
                obj.uid, physicsClientId=cid,
            )
            position = np.array([px, py, pz], dtype=np.float32)
            R = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
            size = self._BLOCK_LOCAL_SIZES[key].astype(np.float32)

            # World-axis envelope of the rotated box.
            local = _LOCAL_UNIT * size            # (8, 3)
            world = (R @ local.T).T + position    # (8, 3)
            aabb_min = world.min(axis=0).astype(np.float32)
            aabb_max = world.max(axis=0).astype(np.float32)

            out[key] = {
                'position': position,
                'size':     size,
                'rotation': R,
                'aabb_min': aabb_min,
                'aabb_max': aabb_max,
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
