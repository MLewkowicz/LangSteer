"""SpaceMouse teleoperation and data collection in Isaac Sim.

Usage::

    python scripts/teleop_isaac.py instruction="pick up the mug" \\
        task_name=pick_up_mug object=mug

Controls:
    SpaceMouse 6-DOF   → delta EE position / orientation (EE frame)
    SpaceMouse left    → toggle gripper (open ↔ closed)
    SpaceMouse right   → toggle recording on / off (same button starts and saves)
    Ctrl-C             → quit (saves any active recording)

The Isaac GUI viewport opens with the cabinet USD; when ``use_gui=true`` the
scene wires two extra docked viewport panels pinned to the ``static`` (overhead)
and ``gripper`` (wrist) camera prims so you can teleop while watching what the
policy will see.  Drag-dock the two panels alongside the main 3D view on first
launch; Isaac persists the layout across sessions.

Each saved episode is an HDF5 file consumed by IsaacDataset in
training/policies/diffuser_actor/dataset_isaac.py.
"""

import logging
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal SpaceMouse deadband + smoothing filter (mirrors ThreeDMouseFilter)
# ---------------------------------------------------------------------------

class _SpaceMouseFilter:
    """Apply deadband + exponential smoothing to 6-DOF SpaceMouse input."""

    def __init__(
        self,
        translation_deadband: float = 0.08,
        rotation_deadband: float = 0.08,
        smoothing: float = 0.3,
        linear_scale: float = 0.001,
        angular_scale: float = 0.005,
    ) -> None:
        self._tdead = translation_deadband
        self._rdead = rotation_deadband
        self._alpha = smoothing
        self._lscale = linear_scale
        self._ascale = angular_scale
        self._v_smooth = np.zeros(3)
        self._w_smooth = np.zeros(3)

    def __call__(
        self, v_raw: np.ndarray, w_raw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return smoothed, deadbanded, scaled (v, w)."""
        v = v_raw.copy()
        w = w_raw.copy()
        v[np.abs(v) < self._tdead] = 0.0
        w[np.abs(w) < self._rdead] = 0.0
        self._v_smooth = self._alpha * self._v_smooth + (1 - self._alpha) * v
        self._w_smooth = self._alpha * self._w_smooth + (1 - self._alpha) * w
        return self._v_smooth * self._lscale, self._w_smooth * self._ascale


# ---------------------------------------------------------------------------
# SpaceMouse reader (pyspacemouse backend)
# ---------------------------------------------------------------------------

def _open_spacemouse():
    """Open SpaceMouse and return a reader callable.

    pyspacemouse 2.x API: ``pyspacemouse.open()`` returns a SpaceMouseDevice;
    state is read via ``device.read()`` (the legacy module-level
    ``pyspacemouse.read()`` was removed).

    Returns:
        Callable[[], tuple[np.ndarray, np.ndarray, list]] or None on failure.
    """
    try:
        import pyspacemouse

        device = pyspacemouse.open()
        if device is None or device is False:
            raise RuntimeError("pyspacemouse.open() returned no device")

        def _read():
            # Drain hidapi's queue every tick. The SpaceMouse firmware
            # streams HID reports faster than we tick (and on separate
            # channels for XYZ vs RPY), so a single device.read() pulls
            # only one queued report — leaving a backlog proportional to
            # how long the puck was held. After release we'd otherwise
            # play back stale "held" samples before seeing the zero.
            # SpaceMouseState.t advances only when a new report is
            # processed, so we loop until it stops changing.
            s = device.read()
            last_t = s.t
            for _ in range(128):
                s = device.read()
                if s.t == last_t:
                    break
                last_t = s.t
            v = np.array([s.x, s.y, s.z], dtype=float)
            w = np.array([s.roll, s.pitch, s.yaw], dtype=float)
            # print(f"Raw input: v={v} w={w} buttons={s.buttons}")
            return v, w, list(s.buttons)

        logger.info(f"SpaceMouse connected: {device.product_name}")
        return _read
    except Exception as e:
        logger.warning(f"SpaceMouse unavailable ({e})")
        return None


# ---------------------------------------------------------------------------
# Main teleop loop
# ---------------------------------------------------------------------------

def _run_teleop(cfg: DictConfig) -> None:
    from envs.isaac_sim_utils.scene import IsaacSimScene
    from envs.isaac_sim_utils.recorder import IsaacSimRecorder

    instruction = cfg.get("instruction", "")

    # Build scene cfg from the env-style hydra block if present; otherwise from
    # flat top-level keys. This lets teleop reuse conf/env/isaac_sim.yaml verbatim.
    env_cfg = cfg.get("env", None)
    if env_cfg is not None:
        from omegaconf import OmegaConf
        scene_cfg = OmegaConf.to_container(env_cfg, resolve=True)
    else:
        scene_cfg = {
            "use_gui": cfg.get("use_gui", True),
            "usd_path": cfg.get("usd_path", None),
            "robot_prim_path": cfg.get("robot_prim_path", "/World/franka"),
            "ee_prim_path": cfg.get("ee_prim_path", "/World/franka/robotiq_arg2f_tcp"),
            "cameras": {
                "static": {
                    "resolution": [200, 200],
                    "position": cfg.get("static_cam_pos", [0.5, 0.0, 1.2]),
                    "target": cfg.get("static_cam_target", [0.4, 0.0, 0.42]),
                },
                "gripper": {
                    "resolution": [200, 200],
                    "prim_path": cfg.get(
                        "gripper_cam_prim",
                        "/World/franka/fr3_link8/zedm_camera_link/gripper_cam",
                    ),
                },
            },
        }
    scene = IsaacSimScene(scene_cfg)
    # Push the initial PD targets (= current joint state) for a few ticks
    # so PhysX never sees a moment without an explicit target.  Without
    # this, the per-joint drives revert to URDF-default targets (typically
    # zero), and the arm yanks toward home pose on the very first step.
    for _ in range(20):
        scene.hold_targets()
        scene.step_physics(num_steps=1)

    recorder = IsaacSimRecorder(
        save_dir=cfg.data_dir,
        episode_name=cfg.get("episode_name", None),
    )

    filt = _SpaceMouseFilter(
        translation_deadband=cfg.spacemouse.translation_deadband,
        rotation_deadband=cfg.spacemouse.rotation_deadband,
        smoothing=cfg.spacemouse.smoothing,
        linear_scale=cfg.linear_scale,
        angular_scale=cfg.angular_scale,
    )

    read_sm = _open_spacemouse()
    if read_sm is None:
        logger.error("Cannot open SpaceMouse — aborting.")
        scene.close()
        return

    save_dir = Path(cfg.data_dir)
    episodes_saved = 0
    num_episodes = cfg.get("num_episodes", 0)  # 0 = unlimited until Ctrl-C

    obj = str(cfg.get("object", ""))

    print("Teleop ready.")
    if instruction:
        print(f"  Instruction: {instruction}")
    if obj:
        print(f"  Object:      {obj}")
    print("  LEFT button  → toggle gripper (open ↔ closed)")
    print("  RIGHT button → toggle recording on/off")
    print("  Ctrl-C       → quit")

    gripper_open = True
    prev_left = 0
    prev_right = 0
    episode_idx = len(list(save_dir.glob("episode_*.h5"))) if save_dir.exists() else 0
    dt = 1.0 / cfg.spacemouse.control_rate
    physics_substeps = max(1, int(cfg.get("physics_substeps", 2)))

    # Pick the teleop integration pattern based on the controller chosen
    # in env.controller_type.  Diff-IK works with "current + delta" each
    # tick (no accumulation needed); OSC needs target accumulation so the
    # impedance error builds up enough to drive meaningful motion.
    controller_type = str(cfg.env.get("controller_type", "diff_ik")).lower()

    # Initialize a persistent target pose at the live EE pose so the
    # first control tick computes zero error (no jolt at startup).  For
    # diff-IK this is overwritten each tick; for OSC it accumulates.
    from scipy.spatial.transform import Rotation as _Rot
    _ee_pos, _ee_euler = scene.get_ik_frame_pose()
    target_pos = np.asarray(_ee_pos, dtype=np.float64).copy()
    target_R = _Rot.from_euler("xyz", _ee_euler).as_matrix()

    # OSC-only: cap how far the target may lead the EE.
    target_lead_max = float(cfg.get("target_lead_max", 0.05))

    # SpaceMouse → Franka-base-frame axis remap.  This is purely a
    # hardware-axis convention (the 3Dconnexion puck's raw (x,y,z) doesn't
    # match the Franka's local +X-forward / +Y-left convention) — it's
    # NOT tied to how your Franka is oriented in the USD.  The robot's
    # actual orientation is auto-detected at runtime via R_base from
    # scene.get_base_frame_pose() and applied below.  These defaults work
    # for any Franka regardless of base rotation.
    _DEFAULT_LINEAR_REMAP = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    _DEFAULT_ANGULAR_REMAP = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    sm_linear_remap = np.array(
        cfg.spacemouse.get("linear_remap", _DEFAULT_LINEAR_REMAP), dtype=float,
    )
    sm_angular_remap = np.array(
        cfg.spacemouse.get("angular_remap", _DEFAULT_ANGULAR_REMAP), dtype=float,
    )

    # Auto-reset after each saved episode: snaps the robot back to its
    # USD-authored joint defaults, opens the gripper, and re-seeds the
    # controller targets from the fresh EE pose.  Lets the operator collect
    # many demos per teleop session without restarting the script.
    auto_reset = bool(cfg.get("auto_reset", True))
    reset_settle_steps = int(cfg.get("reset_settle_steps", 20))

    iter_count = 0
    log_every = max(1, int(cfg.spacemouse.control_rate * 5))  # ~once per 5 s

    try:
        while True:
            t0 = time.monotonic()
            iter_count += 1
            if iter_count == 1 or iter_count % log_every == 0:
                logger.info(
                    f"[loop] iter={iter_count} "
                    f"gripper={'open' if gripper_open else 'closed'} "
                    f"recording={recorder.recording}"
                )

            v_raw, w_raw, buttons = read_sm()
            left = int(buttons[0]) if len(buttons) > 0 else 0
            right = int(buttons[1]) if len(buttons) > 1 else 0

            # LEFT (rising edge): toggle gripper
            if left and not prev_left:
                gripper_open = not gripper_open
                logger.info(f"  Gripper → {'open' if gripper_open else 'closed'}")
            prev_left = left

            # RIGHT (rising edge): toggle recording.  Stopping a recording
            # auto-resets the scene (robot → home, gripper → open, controller
            # targets re-seeded from the fresh EE pose) so the operator can
            # immediately reposition objects and hit RIGHT again to start
            # the next episode — no Ctrl-C between demos.
            if right and not prev_right:
                was_recording = recorder.recording
                recorder.toggle(
                    instruction=instruction,
                    task_name=cfg.get("task_name", ""),
                    object=obj,
                )
                if was_recording and not recorder.recording:
                    episode_idx += 1
                    episodes_saved += 1
                    if auto_reset:
                        scene.reset_robot()
                        # Burn a few physics steps so the articulation settles
                        # at the USD-authored home pose before we sample the
                        # EE pose to seed the controller targets.
                        for _ in range(reset_settle_steps):
                            scene.hold_targets()
                            scene.step_physics(num_steps=1)
                        gripper_open = True
                        _p, _e = scene.get_ik_frame_pose()
                        target_pos = np.asarray(_p, dtype=np.float64).copy()
                        target_R = _Rot.from_euler("xyz", _e).as_matrix()
                        print(
                            "  Scene reset — reposition objects, then "
                            "RIGHT to record the next episode."
                        )
            prev_right = right

            v, w = filt(v_raw, w_raw)
            # Remap SpaceMouse axes to the Franka base frame convention:
            #   puck +Y (push forward away from user) → robot +X (forward)
            #   puck +X (push right)                  → robot -Y (right)
            #   puck +Z (lift)                        → robot +Z (up)
            v = sm_linear_remap @ v
            w = sm_angular_remap @ w

            # Transform base-frame deltas into world frame using the
            # robot's actual base orientation in the USD.
            _, R_base = scene.get_base_frame_pose()
            v_world = R_base @ v
            w_world = R_base @ w

            from scipy.spatial.transform import Rotation
            ee_pos_now, ee_euler_now = scene.get_ik_frame_pose()

            if controller_type == "osc":
                # Accumulate target pose, then cap lead distance.
                target_pos = target_pos + v_world
                dR = Rotation.from_rotvec(w_world).as_matrix()
                target_R = dR @ target_R
                lead = target_pos - ee_pos_now
                lead_norm = float(np.linalg.norm(lead))
                if lead_norm > target_lead_max:
                    target_pos = ee_pos_now + lead * (target_lead_max / lead_norm)
            else:
                # Diff-IK pattern: target = current_ee + delta each tick.
                # No accumulation — the controller's persistent _target_q
                # absorbs successive joint-space deltas internally.
                target_pos = ee_pos_now + v_world
                R_ee = Rotation.from_euler("xyz", ee_euler_now).as_matrix()
                dR = Rotation.from_rotvec(w_world).as_matrix()
                target_R = dR @ R_ee

            target_euler = Rotation.from_matrix(target_R).as_euler("xyz")
            scene.apply_ee_action(target_pos, target_euler, gripper_open)
            scene.step_physics(num_steps=physics_substeps)

            # Capture camera data only when recording — the live Isaac
            # GUI viewport panels show the static + gripper cams natively
            # (set up in IsaacSimScene when use_gui=true), so no extra
            # preview capture is needed.
            if recorder.recording:
                rgb, depth, intrinsics, extrinsics = scene.get_camera_data()

            # Record after stepping physics (fresh sensor data)
            if recorder.recording:
                from envs.isaac_sim_utils.observation import deproject_depth_per_pixel

                pcd_static = deproject_depth_per_pixel(
                    depth["static"], intrinsics["static"], extrinsics["static"]
                )
                pcd_gripper = deproject_depth_per_pixel(
                    depth["gripper"], intrinsics["gripper"], extrinsics["gripper"]
                )
                robot_obs = scene.get_robot_obs()
                recorder.step(
                    rgb_static=rgb["static"],
                    rgb_gripper=rgb["gripper"],
                    pcd_static=pcd_static,
                    pcd_gripper=pcd_gripper,
                    robot_obs=robot_obs,
                    ee_pose=robot_obs[:7].copy(),
                )

            if num_episodes > 0 and episodes_saved >= num_episodes:
                print(f"\nCollected {episodes_saved} episodes. Done.")
                break

            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        import traceback
        logger.error(f"Main loop crashed at iter={iter_count}: {type(exc).__name__}: {exc}")
        logger.error(traceback.format_exc())
    finally:
        recorder.close()
        scene.close()
        print(f"Saved {episodes_saved} episodes to {cfg.data_dir}")


@hydra.main(config_path="../conf", config_name="teleop_isaac", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    _run_teleop(cfg)


if __name__ == "__main__":
    main()
