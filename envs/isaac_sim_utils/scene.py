"""Isaac Sim scene setup for tabletop manipulation.

Creates a Franka Panda on a table with configurable camera placements
and task-specific object spawning.  All ``omni`` / ``isaacsim`` imports
are contained here so the rest of LangSteer can remain Isaac-free.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class IsaacSimScene:
    """Manages the USD stage: Franka, table, cameras, and task objects.

    Lazily imports Isaac Sim packages — instantiate only when Isaac Sim
    is available.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        # Deferred heavy imports
        from isaacsim import SimulationApp

        headless = not cfg.get("use_gui", True)
        self._sim_app = SimulationApp({"headless": headless})

        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.franka import Franka
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid

        self._World = World
        self._add_reference = add_reference_to_stage
        self._Franka = Franka
        self._DynamicCuboid = DynamicCuboid
        self._FixedCuboid = FixedCuboid

        self._cfg = cfg
        self._world: Optional[Any] = None
        self._franka: Optional[Any] = None
        self._controller: Optional[Any] = None
        self._cameras: Dict[str, Any] = {}
        self._objects: Dict[str, Any] = {}

        self._setup_world()

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _setup_world(self) -> None:
        """Create the world, ground plane, table, Franka, and cameras."""
        self._world = self._World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()

        # Table — a fixed cuboid
        table_size = self._cfg.get("table_size", [0.8, 1.0, 0.02])
        table_pos = self._cfg.get("table_position", [0.4, 0.0, 0.4])
        self._world.scene.add(
            self._FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array(table_pos),
                scale=np.array(table_size),
                color=np.array([0.4, 0.3, 0.2]),
            )
        )

        # Franka Panda
        franka_pos = self._cfg.get("franka_position", [0.0, 0.0, 0.41])
        self._franka = self._world.scene.add(
            self._Franka(
                prim_path="/World/Franka",
                name="franka",
                position=np.array(franka_pos),
            )
        )

        # Controller for end-effector pose targets
        from omni.isaac.franka.controllers import RMPFlowController

        self._controller = RMPFlowController(
            name="franka_rmpflow", robot_articulation=self._franka
        )

        # Cameras
        self._setup_cameras()

        # Warm up physics
        self._world.reset()
        logger.info("Isaac Sim scene initialized")

    def _setup_cameras(self) -> None:
        """Create static and wrist cameras from config."""
        from omni.isaac.sensor import Camera

        cam_cfgs = self._cfg.get("cameras", {})

        # Static camera (overhead / angled view)
        static_cfg = cam_cfgs.get("static", {})
        static_res = static_cfg.get("resolution", [256, 256])
        static_pos = static_cfg.get("position", [0.5, 0.0, 1.2])
        static_target = static_cfg.get("target", [0.4, 0.0, 0.42])
        static_cam = Camera(
            prim_path="/World/Cameras/static_camera",
            name="static",
            resolution=(static_res[0], static_res[1]),
            position=np.array(static_pos),
        )
        static_cam.set_focal_length(1.93)  # Approximate for ~90° FOV
        static_cam.set_clipping_range(0.01, 5.0)
        # Point camera at target
        static_cam.set_world_pose(
            position=np.array(static_pos),
            orientation=self._look_at_orientation(
                np.array(static_pos), np.array(static_target)
            ),
        )
        self._cameras["static"] = static_cam
        self._world.scene.add(static_cam)

        # Wrist camera (mounted on Franka EE link)
        wrist_cfg = cam_cfgs.get("wrist", {})
        wrist_res = wrist_cfg.get("resolution", [128, 128])
        wrist_cam = Camera(
            prim_path="/World/Franka/panda_hand/wrist_camera",
            name="wrist",
            resolution=(wrist_res[0], wrist_res[1]),
        )
        wrist_cam.set_focal_length(1.93)
        wrist_cam.set_clipping_range(0.01, 3.0)
        self._cameras["wrist"] = wrist_cam
        self._world.scene.add(wrist_cam)

    @staticmethod
    def _look_at_orientation(
        eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])
    ) -> np.ndarray:
        """Compute a quaternion (w, x, y, z) for a camera looking at *target*."""
        from scipy.spatial.transform import Rotation

        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        actual_up = np.cross(right, forward)

        rot_mat = np.stack([right, actual_up, -forward], axis=1)  # OpenGL convention
        quat_xyzw = Rotation.from_matrix(rot_mat).as_quat()
        # Isaac Sim uses (w, x, y, z)
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def spawn_objects(
        self, object_specs: List[Tuple[str, np.ndarray, np.ndarray]]
    ) -> None:
        """Spawn task objects on the table.

        Args:
            object_specs: List of (object_name, position, euler_deg).
        """
        from envs.isaac_sim_utils.task_configs import OBJECT_CATALOG
        from scipy.spatial.transform import Rotation

        self.clear_objects()

        for obj_name, position, euler_deg in object_specs:
            catalog_entry = OBJECT_CATALOG.get(obj_name)
            if catalog_entry is None:
                logger.warning(f"Object '{obj_name}' not in catalog, skipping")
                continue

            prim_path = f"/World/Objects/{obj_name}"
            quat_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
            orientation = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )

            size = catalog_entry["size"]
            color = np.array(catalog_entry["color"])
            mass = catalog_entry.get("mass", 0.05)

            if mass > 0:
                obj = self._DynamicCuboid(
                    prim_path=prim_path,
                    name=obj_name,
                    position=position,
                    orientation=orientation,
                    scale=size,
                    color=color,
                    mass=mass,
                )
            else:
                obj = self._FixedCuboid(
                    prim_path=prim_path,
                    name=obj_name,
                    position=position,
                    orientation=orientation,
                    scale=size,
                    color=color,
                )
            self._world.scene.add(obj)
            self._objects[obj_name] = obj

        logger.info(f"Spawned {len(self._objects)} objects: {list(self._objects.keys())}")

    def clear_objects(self) -> None:
        """Remove all task objects from the stage."""
        for name in list(self._objects.keys()):
            self._world.scene.remove_object(name)
        self._objects.clear()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_object_poses(self) -> Dict[str, np.ndarray]:
        """Return world-frame positions for all task objects.

        Returns:
            Dict mapping object name → (3,) position array.
        """
        poses = {}
        for name, obj in self._objects.items():
            pos, _ = obj.get_world_pose()
            poses[name] = np.array(pos)
        return poses

    def get_object_aabbs(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return axis-aligned bounding boxes for all task objects.

        Returns:
            Dict mapping object name → (min_xyz, max_xyz).
        """
        from envs.isaac_sim_utils.task_configs import OBJECT_CATALOG

        aabbs = {}
        for name, obj in self._objects.items():
            pos, _ = obj.get_world_pose()
            pos = np.array(pos)
            half = OBJECT_CATALOG.get(name, {}).get("size", np.array([0.04, 0.04, 0.04])) / 2
            aabbs[name] = (pos - half, pos + half)
        return aabbs

    def get_robot_obs(self) -> np.ndarray:
        """Return 15-dim robot observation matching CALVIN convention.

        Format: [tcp_pos(3), tcp_euler(3), gripper_width(1), joints(7), grip_action(1)]
        """
        from scipy.spatial.transform import Rotation

        ee_pos, ee_quat_wxyz = self._franka.get_world_pose()
        # panda_hand gives EE pose
        # Convert quaternion (w,x,y,z) → euler XYZ
        quat_xyzw = np.array(
            [ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]]
        )
        ee_euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")

        joint_positions = self._franka.get_joint_positions()
        # Franka has 9 joints: 7 arm + 2 finger
        arm_joints = joint_positions[:7]
        finger_pos = joint_positions[7:9]
        gripper_width = np.array([float(np.sum(finger_pos))])
        grip_action = np.array([1.0 if gripper_width[0] > 0.02 else -1.0])

        robot_obs = np.concatenate(
            [np.array(ee_pos), ee_euler, gripper_width, arm_joints, grip_action]
        )
        return robot_obs.astype(np.float32)

    def get_scene_obs(self) -> np.ndarray:
        """Return scene observation vector with all object poses.

        Packs up to 8 objects as [pos(3)] each → (24,) max.
        Padded with zeros if fewer objects.
        """
        max_objects = 8
        scene_obs = np.zeros(max_objects * 3, dtype=np.float32)
        poses = self.get_object_poses()
        for i, (name, pos) in enumerate(poses.items()):
            if i >= max_objects:
                break
            scene_obs[i * 3 : (i + 1) * 3] = pos
        return scene_obs

    # ------------------------------------------------------------------
    # Camera data
    # ------------------------------------------------------------------

    def get_camera_data(self) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, Dict[str, float]],
        Dict[str, np.ndarray],
    ]:
        """Capture RGB, depth, intrinsics, and extrinsics from all cameras.

        Returns:
            rgb: Camera name → (H, W, 3) uint8.
            depth: Camera name → (H, W) float32 meters.
            intrinsics: Camera name → dict(fx, fy, cx, cy).
            extrinsics: Camera name → (4, 4) cam-to-world matrix.
        """
        from scipy.spatial.transform import Rotation

        rgb_out = {}
        depth_out = {}
        intrinsics_out = {}
        extrinsics_out = {}

        for name, cam in self._cameras.items():
            # RGB
            rgba = cam.get_rgba()
            if rgba is not None:
                rgb_out[name] = rgba[:, :, :3].astype(np.uint8)
            else:
                res = cam.get_resolution()
                rgb_out[name] = np.zeros((res[1], res[0], 3), dtype=np.uint8)

            # Depth
            depth_data = cam.get_depth()
            if depth_data is not None:
                depth_out[name] = depth_data.astype(np.float32)
            else:
                res = cam.get_resolution()
                depth_out[name] = np.zeros((res[1], res[0]), dtype=np.float32)

            # Intrinsics from camera properties
            intrinsic_mat = cam.get_intrinsics_matrix()
            intrinsics_out[name] = {
                "fx": float(intrinsic_mat[0, 0]),
                "fy": float(intrinsic_mat[1, 1]),
                "cx": float(intrinsic_mat[0, 2]),
                "cy": float(intrinsic_mat[1, 2]),
            }

            # Extrinsics (camera-to-world)
            pos, quat_wxyz = cam.get_world_pose()
            quat_xyzw = np.array(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            )
            R = Rotation.from_quat(quat_xyzw).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.array(pos)
            extrinsics_out[name] = T

        return rgb_out, depth_out, intrinsics_out, extrinsics_out

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    def step_physics(self, num_steps: int = 1) -> None:
        """Advance the physics simulation."""
        for _ in range(num_steps):
            self._world.step(render=True)

    def apply_ee_action(
        self, target_pos: np.ndarray, target_euler: np.ndarray, gripper_open: bool
    ) -> None:
        """Command the Franka to move toward an EE pose target.

        Args:
            target_pos: (3,) target end-effector position in world frame.
            target_euler: (3,) target orientation as euler XYZ in radians.
            gripper_open: Whether the gripper should be open.
        """
        from scipy.spatial.transform import Rotation
        from omni.isaac.core.utils.types import ArticulationAction

        # Convert euler → quaternion for the controller
        quat_xyzw = Rotation.from_euler("xyz", target_euler).as_quat()
        target_orient = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        )

        action = self._controller.forward(
            target_end_effector_position=target_pos,
            target_end_effector_orientation=target_orient,
        )
        self._franka.apply_action(action)

        # Gripper
        gripper_pos = 0.04 if gripper_open else 0.0
        gripper_action = ArticulationAction(
            joint_positions=np.array([gripper_pos, gripper_pos]),
            joint_indices=np.array([7, 8]),
        )
        self._franka.apply_action(gripper_action)

    def reset_robot(self) -> None:
        """Reset Franka to its default joint configuration."""
        self._world.reset()
        self._controller.reset()

    def close(self) -> None:
        """Shut down the simulation."""
        if self._sim_app is not None:
            self._sim_app.close()
            self._sim_app = None
