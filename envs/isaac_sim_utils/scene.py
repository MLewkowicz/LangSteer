"""Isaac Sim scene setup for LangSteerCabinet.usd with a USD-placed Franka.

Strategy: load the kitchen USD as authored — including a Franka Panda the
user has positioned manually in the scene — and wrap that Franka prim as
an articulation (no programmatic spawning).  Optionally drop the legacy
Cortado prim if it's still present.

Control: **Operational Space Control (OSC)** with Cartesian impedance,
gravity compensation, and mass-matrix-weighted task-space dynamics.

  arm joints:     torque control  (kps=0, kds=0; OSC computes tau)
  gripper joints: position control (PDs untouched, command via joint_positions)

Per tick:
  1. Compute Cartesian error: pos_err = target - ee, rot_err = log(R_target * R_eeᵀ)
  2. EE velocity: ee_vel = J · qd
  3. Cartesian wrench:  F = Kp · err  − Kd · ee_vel
  4. Task-space mass:   Mₓ = (J · M⁻¹ · Jᵀ)⁻¹     (damped pinv for singularities)
  5. Joint torques:     τ = Jᵀ · Mₓ · F  + g(q)   (gravity comp added)
  6. Apply τ as joint efforts on the arm; apply gripper position separately.

Resulting EE dynamics (in the ideal mass-weighted formulation):
    ẍ + Kd·ẋ + Kp·(x − target) = 0
i.e. unit-mass second-order system → tune Kp for stiffness and Kd = 2√Kp
for critical damping.  The arm naturally accelerates, decelerates, and
yields under contact — same feel as robosuite's OSC_POSE controller.

Public API kept stable so teleop / inference scripts don't change:
    apply_ee_action(target_pos, target_euler, gripper_open)
    hold_targets()
    get_ik_frame_pose() -> (pos, euler)
    get_robot_obs() -> (15,) CALVIN convention
    get_camera_data() -> (rgb, depth, intrinsics, extrinsics) per camera
    spawn_objects, clear_objects, get_object_poses, get_object_aabbs
    step_physics, reset_robot, close
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_USD_PATH_DEFAULT = str(Path(__file__).parents[2] / "LangSteerCabinet.usd")

# Standard Franka Panda gripper geometry — prismatic finger joints
_FRANKA_GRIPPER_OPEN_M = 0.04   # each finger
_FRANKA_GRIPPER_CLOSED_M = 0.0


class IsaacSimScene:
    """Manages the kitchen USD + a USD-placed Franka articulation."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        from isaacsim import SimulationApp

        headless = not cfg.get("use_gui", True)
        self._sim_app = SimulationApp({"headless": headless})

        # Quiet the carb logger AND Python loggers under the isaacsim
        # namespace so the user-facing recorder banner is easy to spot in
        # the terminal.  Set env.verbose_isaac=true to restore the default
        # (info+warning) chatter when debugging.
        if not cfg.get("verbose_isaac", False):
            try:
                import carb
                carb.settings.get_settings().set("/log/level", "error")
                carb.settings.get_settings().set("/log/outputStreamLevel", "error")
                carb.settings.get_settings().set("/log/fileLogLevel", "error")
            except Exception:
                pass
            import logging as _logging
            for _name in (
                "isaacsim",
                "isaacsim.sensors",
                "isaacsim.sensors.camera",
                "isaacsim.sensors.camera.camera",
                "omni",
                "omni.kit",
                "omni.usd",
                "omni.hydra",
            ):
                _logging.getLogger(_name).setLevel(_logging.ERROR)

        self._cfg = cfg
        self._world: Optional[Any] = None
        self._robot: Optional[Any] = None
        self._ee_prim: Optional[Any] = None
        self._cameras: Dict[str, Any] = {}
        self._objects: Dict[str, Any] = {}
        self._DynamicCuboid: Optional[Any] = None
        self._FixedCuboid: Optional[Any] = None
        self._arm_dof_indices: Optional[np.ndarray] = None
        self._gripper_dof_indices: Optional[np.ndarray] = None
        self._target_q: Optional[np.ndarray] = None
        self._ik_body_index: Optional[int] = None
        self._jac_col_offset: int = 0
        self._dof_lower: Optional[np.ndarray] = None
        self._dof_upper: Optional[np.ndarray] = None

        self._setup_world()

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _setup_world(self) -> None:
        """Open the kitchen USD, optionally drop Cortado, wrap existing Franka."""
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import open_stage
        import omni.usd

        self._DynamicCuboid = DynamicCuboid
        self._FixedCuboid = FixedCuboid

        usd_path = self._cfg.get("usd_path") or _USD_PATH_DEFAULT
        open_stage(usd_path)

        stage = omni.usd.get_context().get_stage()

        # Optionally remove the legacy Cortado prim if it's still in the USD.
        # Idempotent — silently no-ops if the prim doesn't exist.
        if self._cfg.get("hide_cortado", True):
            cortado_path = self._cfg.get(
                "cortado_prim_path", "/World/cortado_imported"
            )
            cortado_prim = stage.GetPrimAtPath(cortado_path)
            if cortado_prim and cortado_prim.IsValid():
                stage.RemovePrim(cortado_path)
                logger.info(f"Removed Cortado prim at {cortado_path}")

        self._world = World(stage_units_in_meters=1.0)

        # Wrap the Franka prim authored in the USD as an articulation.
        # We do NOT spawn or reposition the robot — its location is
        # whatever the user authored in the USD.
        robot_prim_path = self._cfg.get("robot_prim_path", "/World/Franka")
        self._robot = self._world.scene.add(
            SingleArticulation(prim_path=robot_prim_path, name="franka")
        )

        self._world.reset()

        # End-effector reference prim — must exist before _setup_osc()
        # because OSC seeds its initial target from the live EE pose.
        ee_path = self._cfg.get(
            "ee_prim_path", f"{robot_prim_path}/panda_hand/tool_center"
        )
        self._ee_prim = SingleXFormPrim(prim_path=ee_path)

        self._setup_joint_indices()

        # Controller selection — default to differential IK (the
        # working baseline).  Set env.controller_type=osc to try the
        # operational-space controller.
        self._controller_type = str(
            self._cfg.get("controller_type", "diff_ik")
        ).lower()
        if self._controller_type == "osc":
            self._setup_osc()
        else:
            self._setup_diff_ik()

        self._setup_cameras()
        self._setup_camera_viewports()
        logger.info(
            f"Isaac Sim scene loaded: {usd_path} "
            f"(controller={self._controller_type})"
        )

    def _setup_joint_indices(self) -> None:
        """Locate the 7 arm + 2 gripper DOFs in the standard Franka."""
        dof_names = list(self._robot.dof_names)
        logger.info(f"Robot DOF names: {dof_names}")
        self._arm_dof_indices = np.array(
            [dof_names.index(f"panda_joint{i}") for i in range(1, 8)], dtype=int
        )
        self._gripper_dof_indices = np.array(
            [
                dof_names.index("panda_finger_joint1"),
                dof_names.index("panda_finger_joint2"),
            ],
            dtype=int,
        )
        logger.info(
            f"Arm DOF: {self._arm_dof_indices.tolist()}  "
            f"Gripper DOF: {self._gripper_dof_indices.tolist()}"
        )

        # Override the standard Franka USD's gripper gains.  Defaults
        # ship as kp=[400, 0] / kd=[80, 0] — only one finger is actively
        # driven (the other is a mimic), and 400 Nm/m gives ~1-2 N grip
        # at typical closing distances.  That's nowhere near enough to
        # hold a plate.  We force both fingers to the same stiff drive.
        finger_kp = float(self._cfg.get("gripper_kp", 10000.0))
        finger_kd = float(self._cfg.get("gripper_kd", 200.0))
        av = self._robot._articulation_view
        kps, kds = av.get_gains()
        new_kps = np.array(kps, dtype=np.float32, copy=True)
        new_kds = np.array(kds, dtype=np.float32, copy=True)
        new_kps[0, self._gripper_dof_indices] = finger_kp
        new_kds[0, self._gripper_dof_indices] = finger_kd
        av.set_gains(kps=new_kps, kds=new_kds)

        # Optional constant inward squeeze torque, applied as a
        # feedforward effort while gripper_open=False.  Adds on top of
        # the position-drive force for very heavy or slippery objects.
        # 0.0 disables; typical useful range 5-30 N·m equivalent.
        self._gripper_grasp_effort = float(
            self._cfg.get("gripper_grasp_effort", 5.0)
        )

        logger.info(
            f"Gripper: kp={finger_kp}, kd={finger_kd}, "
            f"grasp_effort={self._gripper_grasp_effort}"
        )

    def _setup_diff_ik(self) -> None:
        """Cache Jacobian indices, IK params, joint limits, and target_q.

        Differential-IK controller (default).  Per tick the teleop loop
        passes ``target_pos = current_ee + small_delta``; this controller
        solves J · dq = dx with damped LS, accumulates dq into the
        persistent ``_target_q`` joint vector, and pushes it to the PD
        position drives.

        Gravity compensation is applied as a feedforward joint torque on
        top of the position drive — without it, idle holding and
        snap-on-zero fight against gravity-induced drift.
        """
        av = self._robot._articulation_view
        body_names = list(av.body_names)
        dof_names = list(av.dof_names)

        wrist_body = self._cfg.get("ik_body_name", "panda_hand")
        self._ik_body_index = body_names.index(wrist_body)

        n_cols = av.get_jacobians().shape[3]
        self._jac_col_offset = n_cols - len(dof_names)

        self._ik_damping = float(self._cfg.get("ik_damping", 0.1))
        self._ik_max_step_pos = float(self._cfg.get("ik_max_step_pos", 0.05))
        self._ik_max_step_rot = float(self._cfg.get("ik_max_step_rot", 0.5))

        # Gravity compensation toggle — enabled by default.  Cancels the
        # arm's weight at the joints so PD doesn't have to develop
        # steady-state error to hold against gravity.
        self._gravity_comp = bool(self._cfg.get("gravity_comp", True))

        try:
            limits = av.get_dof_limits()
            self._dof_lower = np.asarray(limits[0, :, 0], dtype=np.float32)
            self._dof_upper = np.asarray(limits[0, :, 1], dtype=np.float32)
        except Exception:
            self._dof_lower = None
            self._dof_upper = None

        # Persistent joint-position target vector — primed at the live
        # joint state so the very first command says "stay where you are".
        self._target_q = np.asarray(
            self._robot.get_joint_positions(), dtype=np.float32
        )

        # Track whether the previous tick had user input. The release
        # snap (target_q ← current_q on zero input) is meant to kill the
        # PD chase tail after motion, but if applied every idle tick it
        # also surrenders to external disturbances (e.g. gripper grasp
        # reaction propagating up the arm). Snap once on the motion →
        # idle transition, then hold target_q so the PD drives can
        # actively resist any external torque.
        self._diff_ik_was_moving = False

        # Log the live PD gains so we can confirm we're in position-control
        # mode (any prior controller_type=osc run would have zeroed them).
        kps, kds = av.get_gains()
        logger.info(
            f"Diff-IK: wrist body '{wrist_body}' idx={self._ik_body_index}, "
            f"Jacobian col offset={self._jac_col_offset}, damping={self._ik_damping}, "
            f"gravity_comp={self._gravity_comp}"
        )
        logger.info(
            f"Diff-IK: arm kps[0..6]={np.round(kps[0, self._arm_dof_indices], 1).tolist()}, "
            f"kds[0..6]={np.round(kds[0, self._arm_dof_indices], 1).tolist()}"
        )

    def _setup_osc(self) -> None:
        """Switch arm joints to torque control and prime OSC parameters.

        - Disables PhysX position-control PDs on the 7 arm joints
          (kps=0, kds=0) so applied joint efforts pass through directly.
        - Leaves gripper PDs untouched so finger position control still
          works via standard joint_positions commands.
        - Caches Jacobian indices, body index, OSC stiffness/damping
          gains, and an internal target pose for hold_targets().
        """
        av = self._robot._articulation_view
        body_names = list(av.body_names)
        dof_names = list(av.dof_names)

        # Wrist body — panda_hand for standard Franka.  Jacobian rows here
        # represent the EE's spatial velocity in the world frame.
        wrist_body = self._cfg.get("ik_body_name", "panda_hand")
        self._ik_body_index = body_names.index(wrist_body)

        # PhysX may pad the Jacobian with floating-base columns; resolve
        # the offset dynamically.  For a fixed-base Franka this is 0.
        n_cols = av.get_jacobians().shape[3]
        self._jac_col_offset = n_cols - len(dof_names)

        # Disable PD on the arm DOFs only — gripper keeps original gains.
        # Without this, applied torques would compete with the position
        # drive trying to hold whatever target was last set.
        kps_orig, kds_orig = av.get_gains()
        new_kps = np.array(kps_orig, dtype=np.float32, copy=True)
        new_kds = np.array(kds_orig, dtype=np.float32, copy=True)
        new_kps[0, self._arm_dof_indices] = 0.0
        new_kds[0, self._arm_dof_indices] = 0.0
        av.set_gains(kps=new_kps, kds=new_kds)
        logger.info(
            f"OSC: disabled PD on arm DOFs {self._arm_dof_indices.tolist()}; "
            f"gripper PDs preserved"
        )

        # OSC Cartesian impedance gains (unit-mass equivalent because we
        # multiply by Mx).  Kd = 2√Kp gives critical damping.
        self._osc_kp_pos = float(self._cfg.get("osc_kp_pos", 200.0))
        self._osc_kp_rot = float(self._cfg.get("osc_kp_rot", 30.0))
        self._osc_kd_pos = float(
            self._cfg.get("osc_kd_pos", 2.0 * np.sqrt(self._osc_kp_pos))
        )
        self._osc_kd_rot = float(
            self._cfg.get("osc_kd_rot", 2.0 * np.sqrt(self._osc_kp_rot))
        )
        # Safety caps on Cartesian error — bounds peak torque magnitudes.
        self._osc_max_pos_err = float(self._cfg.get("osc_max_pos_err", 0.10))
        self._osc_max_rot_err = float(self._cfg.get("osc_max_rot_err", 0.50))
        # Damping for the task-space inertia inversion (handles singularities).
        self._osc_mx_rcond = float(self._cfg.get("osc_mx_rcond", 1e-3))

        # Internal target pose for hold_targets() — initialised to the
        # current EE pose so the very first OSC tick computes zero error.
        from scipy.spatial.transform import Rotation
        ee_pos, ee_euler = self.get_ik_frame_pose()
        self._last_target_pos = np.asarray(ee_pos, dtype=np.float64)
        self._last_target_R = Rotation.from_euler("xyz", ee_euler).as_matrix()
        self._last_gripper_open = True

        logger.info(
            f"OSC: wrist body '{wrist_body}' idx={self._ik_body_index}, "
            f"Kp_pos={self._osc_kp_pos}, Kd_pos={self._osc_kd_pos:.1f}, "
            f"Kp_rot={self._osc_kp_rot}, Kd_rot={self._osc_kd_rot:.1f}"
        )

    def _setup_cameras(self) -> None:
        """Attach cameras from the `cameras:` config block.

        Each camera entry supports the following keys:
            prim_path:        USD path. Default ``{robot_prim_path}/{name}_cam`` —
                              parenting under the robot prim makes the camera
                              follow the robot when the robot moves.
            resolution:       [W, H]. Default [200, 200].
            focal_length:     float. Default 1.93.
            clipping_range:   [near, far]. Default [0.01, 5.0].

        Pose (priority order — first match wins):
            local_position + local_target:       look-at in the parent frame
            local_position + local_orientation:  explicit quat (wxyz)
            world_position + world_target  (or `position`+`target`):
                                                  look-at in world frame
            (none of the above):                  leave at USD-authored pose
        """
        from isaacsim.sensors.camera import Camera

        robot_prim_path = self._cfg.get("robot_prim_path", "/World/Franka")
        cam_cfgs = self._cfg.get("cameras", {})

        for name, raw_cfg in cam_cfgs.items():
            cfg = dict(raw_cfg) if raw_cfg is not None else {}
            prim_path = cfg.get("prim_path", f"{robot_prim_path}/{name}_cam")
            resolution = tuple(cfg.get("resolution", [200, 200]))

            cam = Camera(
                prim_path=prim_path,
                name=name,
                resolution=resolution,
            )
            cam.set_focal_length(float(cfg.get("focal_length", 1.93)))
            clip = cfg.get("clipping_range", [0.01, 5.0])
            cam.set_clipping_range(float(clip[0]), float(clip[1]))

            if "local_position" in cfg:
                local_pos = np.array(cfg["local_position"], dtype=float)
                if "local_target" in cfg:
                    local_target = np.array(cfg["local_target"], dtype=float)
                    local_up = np.array(cfg.get("local_up", [0.0, 0.0, 1.0]), dtype=float)
                    orientation = self._look_at_orientation(
                        local_pos, local_target, up=local_up
                    )
                else:
                    orientation = np.array(
                        cfg.get("local_orientation", [1.0, 0.0, 0.0, 0.0]),
                        dtype=float,
                    )
                cam.set_local_pose(translation=local_pos, orientation=orientation)
            elif "world_position" in cfg or "position" in cfg:
                world_pos = np.array(
                    cfg.get("world_position", cfg.get("position")), dtype=float
                )
                world_target = np.array(
                    cfg.get("world_target", cfg.get("target", world_pos)), dtype=float
                )
                cam.set_world_pose(
                    position=world_pos,
                    orientation=self._look_at_orientation(world_pos, world_target),
                )
            # else: prim retains whatever pose the USD authored.

            self._cameras[name] = cam
            self._world.scene.add(cam)
            # Camera.get_rgba()/get_depth() return None until initialize()
            # has been called.  The world has already been reset above so
            # the render product can attach immediately.
            cam.initialize()
            # Explicitly attach the depth annotator.  Without this,
            # cam.get_depth() lazily attaches on first call and Isaac Sim
            # spams "Annotator 'distance_to_image_plane' not attached" on
            # every tick until it succeeds.
            try:
                cam.add_distance_to_image_plane_to_frame()
            except Exception as exc:
                logger.warning(
                    f"Could not pre-attach depth annotator for '{name}': {exc}"
                )
            logger.info(
                f"Camera '{name}' attached at {prim_path} "
                f"(res={resolution}, focal_length={cfg.get('focal_length', 1.93)})"
            )

    def _setup_camera_viewports(self) -> None:
        """Open Isaac Sim viewport windows pinned to each registered camera.

        Lets the teleop operator watch the policy-eye feeds (static +
        gripper) live inside Isaac's GUI alongside the main 3D editor view.
        Drag-dock the resulting panels next to the main viewport on first
        launch; Isaac persists the layout in user preferences across runs.

        No-op when use_gui=false (headless training/eval).  Failures are
        logged and swallowed because viewport setup is a UX nicety, not
        a correctness-critical path.
        """
        if not self._cfg.get("use_gui", True):
            return
        try:
            from omni.kit.viewport.utility import create_viewport_window
        except Exception as exc:
            logger.warning(f"Multi-viewport unavailable ({exc}); skipping.")
            return

        for name, cam in self._cameras.items():
            try:
                window = create_viewport_window(
                    name.capitalize(), width=300, height=300
                )
                if window is not None and hasattr(window, "viewport_api"):
                    window.viewport_api.set_active_camera(cam.prim_path)
                    logger.info(
                        f"Viewport panel '{name.capitalize()}' pinned to "
                        f"{cam.prim_path}"
                    )
            except Exception as exc:
                logger.warning(
                    f"Could not create viewport for camera '{name}': {exc}"
                )

    @staticmethod
    def _look_at_orientation(
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            # forward ∥ up → cross is zero; pick any vector not parallel to forward.
            fallback = (
                np.array([1.0, 0.0, 0.0])
                if abs(forward[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            right = np.cross(forward, fallback)
        right /= np.linalg.norm(right)
        actual_up = np.cross(right, forward)
        rot_mat = np.stack([right, actual_up, -forward], axis=1)
        quat_xyzw = Rotation.from_matrix(rot_mat).as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    # ------------------------------------------------------------------
    # Object management (task-spawned blocks, etc.)
    # ------------------------------------------------------------------

    def spawn_objects(
        self, object_specs: List[Tuple[str, np.ndarray, np.ndarray]]
    ) -> None:
        from envs.isaac_sim_utils.task_configs import OBJECT_CATALOG
        from scipy.spatial.transform import Rotation

        self.clear_objects()
        for obj_name, position, euler_deg in object_specs:
            entry = OBJECT_CATALOG.get(obj_name)
            if entry is None:
                logger.warning(f"Object '{obj_name}' not in catalog, skipping")
                continue
            prim_path = f"/World/Objects/{obj_name}"
            quat_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
            orientation = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )
            size = np.asarray(entry["size"])
            color = np.array(entry["color"])
            mass = entry.get("mass", 0.05)

            if mass > 0:
                obj = self._DynamicCuboid(
                    prim_path=prim_path, name=obj_name, position=position,
                    orientation=orientation, scale=size, color=color, mass=mass,
                )
            else:
                obj = self._FixedCuboid(
                    prim_path=prim_path, name=obj_name, position=position,
                    orientation=orientation, scale=size, color=color,
                )
            self._world.scene.add(obj)
            self._objects[obj_name] = obj
        logger.info(f"Spawned {len(self._objects)} objects: {list(self._objects.keys())}")

    def clear_objects(self) -> None:
        for name in list(self._objects.keys()):
            self._world.scene.remove_object(name)
        self._objects.clear()

    def get_object_poses(self) -> Dict[str, np.ndarray]:
        return {n: np.array(o.get_world_pose()[0]) for n, o in self._objects.items()}

    def get_object_aabbs(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        from envs.isaac_sim_utils.task_configs import OBJECT_CATALOG
        out = {}
        for name, obj in self._objects.items():
            pos = np.array(obj.get_world_pose()[0])
            half = np.asarray(
                OBJECT_CATALOG.get(name, {}).get("size", [0.04, 0.04, 0.04])
            ) / 2.0
            out[name] = (pos - half, pos + half)
        return out

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_ik_frame_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return EE pose (world frame) as (pos(3,), euler_xyz(3,))."""
        from scipy.spatial.transform import Rotation

        pos, quat_wxyz = self._ee_prim.get_world_pose()
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        )
        euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")
        return np.array(pos, dtype=np.float32), euler.astype(np.float32)

    def get_base_frame_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the robot base's pose: (position, base→world rotation matrix).

        Used by teleop to interpret SpaceMouse deltas in the robot's base
        frame instead of the world or EE frame — so a +X push always means
        "the way the robot is facing" regardless of how the base is rotated
        in the USD.
        """
        from scipy.spatial.transform import Rotation

        pos, quat_wxyz = self._robot.get_world_pose()
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        )
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        return np.array(pos, dtype=np.float32), R.astype(np.float32)

    def get_robot_obs(self) -> np.ndarray:
        """CALVIN-style 15-dim robot observation."""
        ee_pos, ee_euler = self.get_ik_frame_pose()
        all_q = self._robot.get_joint_positions()
        arm_joints = all_q[self._arm_dof_indices]
        finger_q = all_q[self._gripper_dof_indices]
        gripper_width = np.array([float(np.sum(finger_q))])
        grip_action = np.array([1.0 if gripper_width[0] > 0.04 else -1.0])
        return np.concatenate(
            [np.array(ee_pos), np.array(ee_euler), gripper_width, arm_joints, grip_action]
        ).astype(np.float32)

    def get_scene_obs(self) -> np.ndarray:
        max_objects = 8
        obs = np.zeros(max_objects * 3, dtype=np.float32)
        for i, (_, pos) in enumerate(self.get_object_poses().items()):
            if i >= max_objects:
                break
            obs[i * 3 : (i + 1) * 3] = pos
        return obs

    # ------------------------------------------------------------------
    # Camera capture
    # ------------------------------------------------------------------

    def get_camera_data(self) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, Dict[str, float]],
        Dict[str, np.ndarray],
    ]:
        from scipy.spatial.transform import Rotation

        rgb_out, depth_out, intrinsics_out, extrinsics_out = {}, {}, {}, {}
        for name, cam in self._cameras.items():
            rgba = cam.get_rgba()
            rgb_out[name] = (
                rgba[:, :, :3].astype(np.uint8)
                if rgba is not None
                else np.zeros((200, 200, 3), dtype=np.uint8)
            )
            depth_data = cam.get_depth()
            depth_out[name] = (
                depth_data.astype(np.float32)
                if depth_data is not None
                else np.zeros((200, 200), dtype=np.float32)
            )
            K = cam.get_intrinsics_matrix()
            intrinsics_out[name] = {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            }
            pos, quat_wxyz = cam.get_world_pose()
            quat_xyzw = np.array(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            )
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
            T[:3, 3] = np.array(pos)
            extrinsics_out[name] = T
        return rgb_out, depth_out, intrinsics_out, extrinsics_out

    # ------------------------------------------------------------------
    # Control dispatch — diff-IK (default) or OSC
    # ------------------------------------------------------------------

    def apply_ee_action(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        gripper_open: bool,
    ) -> None:
        """Drive the EE toward a target pose.

        Dispatches to whichever controller was selected at scene setup
        (env.controller_type = "diff_ik" or "osc").
        """
        if self._controller_type == "osc":
            self._apply_ee_action_osc(target_pos, target_euler, gripper_open)
        else:
            self._apply_ee_action_diff_ik(target_pos, target_euler, gripper_open)

    def hold_targets(self) -> None:
        """Re-issue the last commanded targets, no change.

        Used during init so PhysX never sees a tick without an applied
        target — without this, drives revert to URDF defaults / OSC
        leaves the arm with no torque.
        """
        if self._controller_type == "osc":
            self._step_osc(self._last_target_pos, self._last_target_R)
            self._command_gripper(self._last_gripper_open)
        else:
            from isaacsim.core.utils.types import ArticulationAction
            if self._gravity_comp:
                g = self._robot._articulation_view.get_generalized_gravity_forces()
                g = g[0] if g.ndim == 2 else g
                self._robot.apply_action(
                    ArticulationAction(
                        joint_positions=self._target_q,
                        joint_efforts=g.astype(np.float32),
                    )
                )
            else:
                self._robot.apply_action(
                    ArticulationAction(joint_positions=self._target_q)
                )

    # ------------------------------------------------------------------
    # Differential IK — joint-position control via Jacobian damped LS
    # ------------------------------------------------------------------

    def _apply_ee_action_diff_ik(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        gripper_open: bool,
    ) -> None:
        """Damped-LS differential IK against the persistent target_q.

        Solves J·dq = dx for one tick's worth of motion, accumulates dq
        into self._target_q[arm], and pushes the full target vector to
        the PD position drives.  Gripper slots are written based on
        ``gripper_open`` and pushed in the same command.
        """
        from scipy.spatial.transform import Rotation
        from isaacsim.core.utils.types import ArticulationAction

        # Update gripper slots
        finger_pos = (
            _FRANKA_GRIPPER_OPEN_M if gripper_open else _FRANKA_GRIPPER_CLOSED_M
        )
        self._target_q[self._gripper_dof_indices] = finger_pos

        # Cartesian error (world frame)
        current_pos, current_euler = self.get_ik_frame_pose()
        dx_pos = np.asarray(target_pos, dtype=np.float64) - current_pos
        R_target = Rotation.from_euler("xyz", target_euler)
        R_current = Rotation.from_euler("xyz", current_euler)
        dx_rot = (R_target * R_current.inv()).as_rotvec()

        # Per-tick step clip
        pos_norm = np.linalg.norm(dx_pos)
        if pos_norm > self._ik_max_step_pos:
            dx_pos = dx_pos * (self._ik_max_step_pos / pos_norm)
        rot_norm = np.linalg.norm(dx_rot)
        if rot_norm > self._ik_max_step_rot:
            dx_rot = dx_rot * (self._ik_max_step_rot / rot_norm)

        if pos_norm > 1e-6 or rot_norm > 1e-6:
            twist = np.concatenate([dx_pos, dx_rot])

            J_full = self._robot._articulation_view.get_jacobians()[
                0, self._ik_body_index
            ]
            arm_cols = self._arm_dof_indices + self._jac_col_offset
            J = J_full[:, arm_cols]
            lam2 = self._ik_damping ** 2
            A = J @ J.T + lam2 * np.eye(6)
            dq = J.T @ np.linalg.solve(A, twist)

            # On the idle → motion transition only, re-base target_q
            # from the live joint state. While idle, target_q is held
            # at the snap value q_snapped, but under external load
            # (e.g. a grasped cube) the joints sit at q_snapped + Δq
            # due to PD steady-state error. Without this re-base, the
            # first motion tick would tell PD to chase q_snapped + dq
            # while joints are at q_snapped + Δq, producing a spurious
            # -Δq command that undoes the load deflection on top of
            # the user's twist — visible as the EE rotating during
            # translation with a grasped object. During sustained
            # motion we keep integrating (target_q += dq) so the
            # natural PD lead builds up and the arm tracks the
            # commanded velocity.
            if not self._diff_ik_was_moving:
                current_q = self._robot.get_joint_positions()
                self._target_q[self._arm_dof_indices] = (
                    current_q[self._arm_dof_indices].astype(np.float32)
                )

            q_arm = self._target_q[self._arm_dof_indices] + dq
            if self._dof_lower is not None and self._dof_upper is not None:
                q_arm = np.clip(
                    q_arm,
                    self._dof_lower[self._arm_dof_indices],
                    self._dof_upper[self._arm_dof_indices],
                )
            self._target_q[self._arm_dof_indices] = q_arm.astype(np.float32)
            self._diff_ik_was_moving = True
        else:
            # No input. On the FIRST idle tick after motion, snap
            # target_q to the live joint state — this kills the ~100 ms
            # PD chase tail that target_q would otherwise produce by
            # lingering slightly ahead of the actual joints. On
            # SUBSEQUENT idle ticks we deliberately leave target_q
            # alone, so the PD drives have a fixed setpoint to resist
            # external disturbances (e.g. grasp reaction torques
            # propagating up the arm when the gripper closes on an
            # object). Snapping every tick instead lets the arm drift
            # in lockstep with any disturbance.
            if self._diff_ik_was_moving:
                current_q = self._robot.get_joint_positions()
                self._target_q[self._arm_dof_indices] = (
                    current_q[self._arm_dof_indices].astype(np.float32)
                )
                self._diff_ik_was_moving = False

        # Build the feedforward effort vector:
        #   • gravity compensation on every DOF (so PD doesn't fight gravity)
        #   • bonus inward squeeze on the fingers when closed (firmer grasp)
        n = self._robot.num_dof
        efforts = np.zeros(n, dtype=np.float32)

        if self._gravity_comp:
            g = self._robot._articulation_view.get_generalized_gravity_forces()
            g = g[0] if g.ndim == 2 else g
            efforts += g.astype(np.float32)

        if (not gripper_open) and self._gripper_grasp_effort > 0.0:
            # Negative effort closes the prismatic finger joint further;
            # PhysX sums this with the position-drive torque, giving a
            # firmer grasp that scales with the configured constant.
            efforts[self._gripper_dof_indices] -= self._gripper_grasp_effort

        if np.any(efforts != 0.0):
            self._robot.apply_action(
                ArticulationAction(
                    joint_positions=self._target_q,
                    joint_efforts=efforts,
                )
            )
        else:
            self._robot.apply_action(
                ArticulationAction(joint_positions=self._target_q)
            )

    # ------------------------------------------------------------------
    # OSC — Cartesian impedance on torque-controlled arm joints
    # ------------------------------------------------------------------

    def _apply_ee_action_osc(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        gripper_open: bool,
    ) -> None:
        """Drive the EE toward a target pose using Cartesian impedance OSC."""
        from scipy.spatial.transform import Rotation

        self._last_target_pos = np.asarray(target_pos, dtype=np.float64).copy()
        self._last_target_R = Rotation.from_euler("xyz", target_euler).as_matrix()
        self._last_gripper_open = bool(gripper_open)

        self._step_osc(self._last_target_pos, self._last_target_R)
        self._command_gripper(gripper_open)

    # ------------------------------------------------------------------

    def _step_osc(self, target_pos: np.ndarray, target_R: np.ndarray) -> None:
        """Compute and apply OSC joint torques on the arm for one tick.

        F = Kp · err  − Kd · ee_vel             (Cartesian wrench)
        τ = Jᵀ · Mₓ · F  +  g(q)                (mass-weighted torques + gravity)
        """
        from scipy.spatial.transform import Rotation
        from isaacsim.core.utils.types import ArticulationAction

        av = self._robot._articulation_view

        # ------------- 1. Read current state -------------
        qd = self._robot.get_joint_velocities()                    # (n,)
        ee_pos, ee_quat_wxyz = self._ee_prim.get_world_pose()
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        ee_quat_xyzw = np.array(
            [ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]]
        )
        R_ee = Rotation.from_quat(ee_quat_xyzw).as_matrix()

        # ------------- 2. Cartesian error -------------
        pos_err = target_pos - ee_pos
        pos_norm = np.linalg.norm(pos_err)
        if pos_norm > self._osc_max_pos_err:
            pos_err *= self._osc_max_pos_err / pos_norm

        # Rotation error as the axis-angle vector of R_target · R_eeᵀ.
        R_err = target_R @ R_ee.T
        rot_err = Rotation.from_matrix(R_err).as_rotvec()
        rot_norm = np.linalg.norm(rot_err)
        if rot_norm > self._osc_max_rot_err:
            rot_err *= self._osc_max_rot_err / rot_norm

        # ------------- 3. Arm Jacobian, mass matrix, gravity -------------
        J_full = av.get_jacobians()[0, self._ik_body_index]         # (6, n_cols)
        arm_cols = self._arm_dof_indices + self._jac_col_offset
        J = J_full[:, arm_cols].astype(np.float64)                  # (6, 7)

        M_full = av.get_mass_matrices()[0].astype(np.float64)       # (n, n)
        arm_idx = self._arm_dof_indices
        M_arm = M_full[np.ix_(arm_idx, arm_idx)]                    # (7, 7)

        g_full = av.get_generalized_gravity_forces()
        g_full = g_full[0] if g_full.ndim == 2 else g_full
        tau_g = g_full[arm_idx].astype(np.float64)                  # (7,)

        # ------------- 4. EE velocity -------------
        ee_vel = J @ qd[arm_idx].astype(np.float64)                 # (6,)

        # ------------- 5. Cartesian wrench -------------
        F_pos = self._osc_kp_pos * pos_err - self._osc_kd_pos * ee_vel[:3]
        F_rot = self._osc_kp_rot * rot_err - self._osc_kd_rot * ee_vel[3:]
        F = np.concatenate([F_pos, F_rot])                          # (6,)

        # ------------- 6. Task-space inertia (damped pseudo-inverse) -------------
        # Mₓ = (J · M⁻¹ · Jᵀ)⁻¹ , damped to handle near-singular configs.
        try:
            M_inv = np.linalg.inv(M_arm)
            Mx_inv = J @ M_inv @ J.T                                # (6, 6)
            Mx = np.linalg.pinv(Mx_inv, rcond=self._osc_mx_rcond)
        except np.linalg.LinAlgError:
            Mx = np.eye(6)

        # ------------- 7. Joint torques + gravity comp -------------
        tau = J.T @ Mx @ F + tau_g                                  # (7,)

        # ------------- 8. Apply -------------
        efforts = np.zeros(self._robot.num_dof, dtype=np.float32)
        efforts[arm_idx] = tau.astype(np.float32)
        self._robot.apply_action(ArticulationAction(joint_efforts=efforts))

    def _command_gripper(self, gripper_open: bool) -> None:
        """Issue a position command on the two finger joints.

        When closed, also apply a constant inward squeeze effort
        (configurable via ``gripper_grasp_effort``) on top of the PD —
        same firmer-grasp feedforward used by the diff-IK path.
        """
        from isaacsim.core.utils.types import ArticulationAction

        finger_pos = (
            _FRANKA_GRIPPER_OPEN_M if gripper_open else _FRANKA_GRIPPER_CLOSED_M
        )
        self._robot.apply_action(
            ArticulationAction(
                joint_positions=np.array([finger_pos, finger_pos], dtype=np.float32),
                joint_indices=self._gripper_dof_indices,
            )
        )
        if (not gripper_open) and self._gripper_grasp_effort > 0.0:
            self._robot.apply_action(
                ArticulationAction(
                    joint_efforts=np.full(
                        2, -self._gripper_grasp_effort, dtype=np.float32
                    ),
                    joint_indices=self._gripper_dof_indices,
                )
            )

    # ------------------------------------------------------------------
    # Simulation lifecycle
    # ------------------------------------------------------------------

    def step_physics(self, num_steps: int = 1) -> None:
        for _ in range(num_steps):
            self._world.step(render=True)

    def reset_robot(self) -> None:
        self._world.reset()
        if self._robot is None or self._arm_dof_indices is None:
            return
        if self._controller_type == "osc":
            from scipy.spatial.transform import Rotation
            ee_pos, ee_euler = self.get_ik_frame_pose()
            self._last_target_pos = np.asarray(ee_pos, dtype=np.float64)
            self._last_target_R = Rotation.from_euler("xyz", ee_euler).as_matrix()
        else:
            self._target_q = np.asarray(
                self._robot.get_joint_positions(), dtype=np.float32
            )

    def close(self) -> None:
        if self._sim_app is not None:
            self._sim_app.close()
            self._sim_app = None
