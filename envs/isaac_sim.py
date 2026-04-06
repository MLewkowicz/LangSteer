"""Isaac Sim environment adapter for LangSteer.

Wraps an Isaac Sim tabletop scene (Franka Panda + objects) behind the
BaseEnvironment interface.  All Isaac Sim imports are lazy — this module
can be imported on any machine, but instantiation requires Isaac Sim.
"""

from typing import Any, Dict, Tuple

import logging
import numpy as np

from core.env import BaseEnvironment
from core.types import Action, Observation

logger = logging.getLogger(__name__)


class IsaacSimEnvironment(BaseEnvironment):
    """Adapter for Isaac Sim Franka Panda tabletop manipulation.

    Observation modes (controlled by ``provide_pcd_images`` config):
      - False (default): fused point cloud from all cameras → ``Observation.pcd``
      - True: per-pixel PCD images per camera → ``Observation.depth`` holds
        world-space XYZ maps (same convention as CALVIN / Diffuser Actor)

    Action format: ``Action.trajectory`` is ``(H, 7)`` with columns
    ``[pos_x, pos_y, pos_z, euler_x, euler_y, euler_z, gripper]``.
    Gripper > 0 → open, ≤ 0 → closed.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)

        # Lazy import — only when actually constructing the env
        from envs.isaac_sim_utils.scene import IsaacSimScene
        from envs.isaac_sim_utils.task_configs import get_task_config, get_task_instruction
        from envs.isaac_sim_utils.task_oracle import IsaacSimTaskOracle

        self._scene = IsaacSimScene(cfg)
        self._oracle = IsaacSimTaskOracle()

        # Task
        self._task_name = cfg.get("task", "pick_up_red_cube")
        self._task_cfg = get_task_config(self._task_name)
        self._instruction = self._task_cfg["instruction"]

        # Env settings
        self._num_points = cfg.get("num_points", 2048)
        self._max_steps = cfg.get("max_steps", 360)
        self._provide_pcd_images = cfg.get("provide_pcd_images", False)
        self._physics_steps_per_action = cfg.get("physics_steps_per_action", 10)
        self._current_step = 0

        logger.info("IsaacSimEnvironment initialized")
        logger.info(f"  Task: {self._task_name}")
        logger.info(f"  Instruction: {self._instruction}")
        logger.info(f"  Max steps: {self._max_steps}")
        logger.info(f"  Provide PCD images: {self._provide_pcd_images}")

    # ------------------------------------------------------------------
    # Observation processing
    # ------------------------------------------------------------------

    def _process_obs(self) -> Observation:
        """Capture sensor data and build an Observation DTO."""
        rgb, depth, intrinsics, extrinsics = self._scene.get_camera_data()
        robot_obs = self._scene.get_robot_obs()

        if self._provide_pcd_images:
            from envs.isaac_sim_utils.observation import prepare_pcd_images

            processed = prepare_pcd_images(
                rgb, depth, intrinsics, extrinsics, robot_obs
            )
            return Observation(
                rgb={
                    "static": processed["rgb_static"],
                    "wrist": processed["rgb_wrist"],
                },
                proprio=processed["robot_obs"],
                ee_pose=processed["ee_pose"],
                instruction=self._instruction,
                depth={
                    "static": processed["pcd_static"],
                    "wrist": processed["pcd_wrist"],
                },
            )
        else:
            from envs.isaac_sim_utils.observation import process_isaac_obs

            processed = process_isaac_obs(
                rgb, depth, intrinsics, extrinsics, robot_obs, self._num_points
            )
            return Observation(
                rgb={"static": processed["rgb_static"]},
                proprio=processed["robot_obs"],
                ee_pose=processed["ee_pose"],
                instruction=self._instruction,
                pcd=processed["point_cloud"],
            )

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def reset(self, robot_obs=None, scene_obs=None) -> Observation:
        """Reset environment: respawn objects and return initial observation.

        Args:
            robot_obs: Unused (reserved for API compatibility with CALVIN).
            scene_obs: Unused (reserved for API compatibility with CALVIN).
        """
        self._scene.reset_robot()

        # Spawn task objects
        self._scene.spawn_objects(self._task_cfg["objects"])

        # Let scene settle
        self._scene.step_physics(num_steps=20)

        self._current_step = 0

        obs = self._process_obs()
        logger.debug("Environment reset complete")
        return obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Execute the full action trajectory, then check success.

        Each waypoint in ``action.trajectory`` is sent as an EE target
        pose and held for ``physics_steps_per_action`` simulation steps.
        """
        total_reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        for t in range(action.trajectory.shape[0]):
            act = action.trajectory[t]
            target_pos = act[:3]
            target_euler = act[3:6]
            gripper_open = bool(act[6] > 0)

            self._scene.apply_ee_action(target_pos, target_euler, gripper_open)
            self._scene.step_physics(num_steps=self._physics_steps_per_action)

            self._current_step += 1
            if self._current_step >= self._max_steps:
                done = True
                info["timeout"] = True
                break

        # Check task success via oracle
        object_poses = self._scene.get_object_poses()
        if self._oracle.check_success(self._task_cfg["success"], object_poses):
            info["success"] = True
            total_reward = 1.0
            done = True

        obs = self._process_obs()
        return obs, total_reward, done, info

    # ------------------------------------------------------------------
    # Scene state (for VoxPoser steering)
    # ------------------------------------------------------------------

    def get_scene_state(self) -> Dict[str, np.ndarray]:
        """Return robot and scene observation vectors.

        Used by VoxPoser steering to initialize object detection.

        Returns:
            Dict with ``robot_obs`` (15,) and ``scene_obs`` (24,).
        """
        return {
            "robot_obs": self._scene.get_robot_obs(),
            "scene_obs": self._scene.get_scene_obs(),
        }

    def get_object_poses(self) -> Dict[str, np.ndarray]:
        """Return ground-truth object positions from the scene graph.

        Returns:
            Dict mapping object name → (3,) world position.
        """
        return self._scene.get_object_poses()

    def get_object_aabbs(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return axis-aligned bounding boxes for all task objects.

        Returns:
            Dict mapping object name → (min_xyz, max_xyz).
        """
        return self._scene.get_object_aabbs()

    # ------------------------------------------------------------------
    # Properties and lifecycle
    # ------------------------------------------------------------------

    @property
    def task_description(self) -> str:
        """Current task language instruction."""
        return self._instruction

    def set_task(self, task_name: str) -> None:
        """Switch to a different task (takes effect on next reset)."""
        from envs.isaac_sim_utils.task_configs import get_task_config

        self._task_name = task_name
        self._task_cfg = get_task_config(task_name)
        self._instruction = self._task_cfg["instruction"]
        logger.info(f"Task switched to: {task_name} — '{self._instruction}'")

    def render(self, mode: str = "rgb_array"):
        """Return static camera RGB if mode='rgb_array'."""
        if mode == "rgb_array":
            rgb, _, _, _ = self._scene.get_camera_data()
            return rgb.get("static")
        return None

    def close(self) -> None:
        """Shut down Isaac Sim."""
        if self._scene is not None:
            self._scene.close()
            self._scene = None
            logger.info("IsaacSimEnvironment closed")
