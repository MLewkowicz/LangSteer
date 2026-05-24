"""Replay a recorded HDF5 episode in Isaac Sim via EE-pose playback.

Loads an episode saved by IsaacSimRecorder and feeds its ``ee_pose``
waypoints back through ``scene.apply_ee_action`` — the same diff-IK + PD
controller that captured the data.  Object positions come from the
recorded ``task_name`` attribute (deterministic spawn via task_configs);
the arm is seeded to the first recorded joint state so the start matches.

Usage::

    uv run python scripts/playback_isaac.py episode_path=data/isaac_sim_demos/grasp_rim.h5
    uv run python scripts/playback_isaac.py episode_path=... task_name=pick_up_red_block
    uv run python scripts/playback_isaac.py episode_path=... realtime_pacing=true

Caveat: Isaac Sim physics is not bit-deterministic, so object trajectories
on contact will be similar but not identical to the original recording.
"""

import logging
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _show_camera_preview(
    rgb_dict: dict,
    out_path: str,
    upscale: int = 2,
    label: str = "PLAYBACK",
) -> None:
    """Write a side-by-side static+gripper PNG (same shape teleop uses)."""
    import cv2

    static = rgb_dict.get("static")
    gripper = rgb_dict.get("gripper")
    if static is None or gripper is None:
        return

    static_bgr = cv2.cvtColor(static, cv2.COLOR_RGB2BGR)
    gripper_bgr = cv2.cvtColor(gripper, cv2.COLOR_RGB2BGR)

    h, w = static_bgr.shape[:2]
    static_disp = cv2.resize(
        static_bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST
    )
    gripper_disp = cv2.resize(
        gripper_bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST
    )

    cv2.putText(
        static_disp, label, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2,
    )
    cv2.putText(
        static_disp, "static", (8, h * upscale - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )
    cv2.putText(
        gripper_disp, "gripper", (8, h * upscale - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )

    combined = cv2.hconcat([static_disp, gripper_disp])
    cv2.imwrite(out_path, combined)


def _run_playback(cfg: DictConfig) -> None:
    from envs.isaac_sim_utils.scene import IsaacSimScene

    episode_path = Path(cfg.episode_path)
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode not found: {episode_path}")

    with h5py.File(episode_path, "r") as f:
        ee_pose = f["ee_pose"][:]              # (T, 7)
        robot_obs = f["robot_obs"][:]          # (T, 15)
        timestamps = f["timestamps"][:]        # (T,)
        hdf5_task_name = str(f.attrs.get("task_name", ""))
        hdf5_instruction = str(f.attrs.get("instruction", ""))
        num_steps = int(f.attrs.get("num_steps", ee_pose.shape[0]))

    task_name = cfg.get("task_name") or hdf5_task_name
    print(f"Replaying {episode_path.name}: {num_steps} steps, "
          f"duration={timestamps[-1]:.1f}s")
    if hdf5_instruction:
        print(f"  Instruction: {hdf5_instruction}")
    if task_name:
        print(f"  Task: {task_name}  (objects will be spawned)")
    else:
        print("  Task: <none> — replaying motion only, no task objects spawned")

    scene_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    scene = IsaacSimScene(scene_cfg)

    # Hold the live joint state for a few ticks so PhysX never sees a
    # moment without an explicit target (same idiom as teleop).
    for _ in range(20):
        scene.hold_targets()
        scene.step_physics(num_steps=1)

    if task_name:
        from envs.isaac_sim_utils.task_configs import get_task_config
        task_cfg = get_task_config(task_name)
        scene.spawn_objects(task_cfg["objects"])

    # Seed the arm to the recorded starting joint state.  robot_obs layout
    # (from scene.get_robot_obs): [ee_pos(3), ee_euler(3), gripper_width(1),
    # arm_joints(7), grip_action(1)] → arm joints at indices 7..14.
    initial_arm_q = robot_obs[0, 7:14].astype(np.float32)
    finger_open = float(ee_pose[0, 6]) > cfg.gripper_open_threshold
    finger_pos = 0.04 if finger_open else 0.0

    full_q = np.asarray(scene._robot.get_joint_positions(), dtype=np.float32)
    full_q[scene._arm_dof_indices] = initial_arm_q
    full_q[scene._gripper_dof_indices] = finger_pos
    scene._robot.set_joint_positions(full_q)

    # Resync the diff-IK controller's persistent target to the new state
    # so the very first apply_ee_action doesn't yank from URDF home.
    if scene._controller_type == "diff_ik":
        scene._target_q = full_q.copy()
    else:
        # OSC: seed last_target from the new live EE pose
        from scipy.spatial.transform import Rotation as _Rot
        ee_pos, ee_euler = scene.get_ik_frame_pose()
        scene._last_target_pos = np.asarray(ee_pos, dtype=np.float64)
        scene._last_target_R = _Rot.from_euler("xyz", ee_euler).as_matrix()

    # Let the start pose + spawned objects settle.
    for _ in range(int(cfg.start_settle_steps)):
        scene.hold_targets()
        scene.step_physics(num_steps=1)

    physics_substeps = max(1, int(cfg.physics_substeps))
    realtime = bool(cfg.realtime_pacing)
    preview = bool(cfg.preview_cameras)
    preview_every = max(1, int(cfg.preview_every))
    preview_path = str(cfg.preview_path)
    threshold = float(cfg.gripper_open_threshold)

    if preview:
        print(f"  Camera preview → {preview_path}")
        print(f"    open with:  feh --reload 0.1 {preview_path}")

    log_every = max(1, num_steps // 20)
    t_wall_start = time.monotonic()

    try:
        for t in range(num_steps):
            target_pos = ee_pose[t, 0:3].astype(np.float64)
            target_euler = ee_pose[t, 3:6].astype(np.float64)
            gripper_open = float(ee_pose[t, 6]) > threshold

            scene.apply_ee_action(target_pos, target_euler, gripper_open)
            scene.step_physics(num_steps=physics_substeps)

            if preview and (t % preview_every == 0):
                rgb, _, _, _ = scene.get_camera_data()
                _show_camera_preview(
                    rgb, preview_path,
                    label=f"PLAYBACK {t+1}/{num_steps}",
                )

            if t % log_every == 0:
                ee_pos_live, _ = scene.get_ik_frame_pose()
                pos_err = float(np.linalg.norm(ee_pos_live - target_pos))
                logger.info(
                    f"[playback] t={t}/{num_steps} "
                    f"gripper={'open' if gripper_open else 'closed'} "
                    f"target={np.round(target_pos, 3).tolist()} "
                    f"err={pos_err*1000:.1f}mm"
                )

            if realtime and t + 1 < num_steps:
                target_wall = float(timestamps[t + 1])
                elapsed = time.monotonic() - t_wall_start
                lag = target_wall - elapsed
                if lag > 0:
                    time.sleep(lag)

        print(f"\nPlayback finished: {num_steps} steps "
              f"in {time.monotonic() - t_wall_start:.1f}s wall.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        import traceback
        logger.error(f"Playback crashed at t={t}: {type(exc).__name__}: {exc}")
        logger.error(traceback.format_exc())
    finally:
        scene.close()


@hydra.main(config_path="../conf", config_name="playback_isaac", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    _run_playback(cfg)


if __name__ == "__main__":
    main()
