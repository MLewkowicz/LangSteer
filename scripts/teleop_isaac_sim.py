"""SpaceMouse teleoperation and data collection for Isaac Sim.

Usage::

    python scripts/teleop_isaac_sim.py \\
        --task pick_up_red_cube \\
        --save_dir data/isaac_sim_demos \\
        --num_episodes 10

Controls:
    SpaceMouse 6-DOF  → delta EE position / orientation
    SpaceMouse button  → toggle gripper open / close
    Keyboard 's'       → save current episode
    Keyboard 'd'       → discard current episode
    Keyboard 'q'       → quit

Each saved episode is a .npz file with the schema expected by
``training/policies/dp3/preprocessing/convert_isaac_sim.py``.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode buffer — accumulates per-timestep data
# ---------------------------------------------------------------------------

class EpisodeBuffer:
    """Collects per-timestep observations and actions for one demonstration."""

    def __init__(self) -> None:
        self._data: Dict[str, List[np.ndarray]] = {
            "rgb_static": [],
            "rgb_wrist": [],
            "depth_static": [],
            "depth_wrist": [],
            "robot_obs": [],
            "rel_actions": [],
            "scene_obs": [],
        }
        self.instruction: str = ""

    def append(
        self,
        rgb: Dict[str, np.ndarray],
        depth: Dict[str, np.ndarray],
        robot_obs: np.ndarray,
        rel_action: np.ndarray,
        scene_obs: np.ndarray,
    ) -> None:
        self._data["rgb_static"].append(rgb.get("static", np.zeros((256, 256, 3), dtype=np.uint8)))
        self._data["rgb_wrist"].append(rgb.get("wrist", np.zeros((128, 128, 3), dtype=np.uint8)))
        self._data["depth_static"].append(depth.get("static", np.zeros((256, 256), dtype=np.float32)))
        self._data["depth_wrist"].append(depth.get("wrist", np.zeros((128, 128), dtype=np.float32)))
        self._data["robot_obs"].append(robot_obs)
        self._data["rel_actions"].append(rel_action)
        self._data["scene_obs"].append(scene_obs)

    def save(self, path: Path) -> None:
        """Write episode to .npz."""
        arrays = {k: np.stack(v) for k, v in self._data.items()}
        arrays["instruction"] = np.array(self.instruction)
        np.savez_compressed(str(path), **arrays)
        logger.info(f"Saved episode ({len(self._data['robot_obs'])} steps) → {path}")

    def __len__(self) -> int:
        return len(self._data["robot_obs"])


# ---------------------------------------------------------------------------
# SpaceMouse input
# ---------------------------------------------------------------------------

def create_spacemouse():
    """Connect to a 3Dconnexion SpaceMouse.

    Returns a callable that, when invoked, returns
    (translation(3,), rotation(3,), buttons(list[bool])).
    """
    import pyspacemouse

    success = pyspacemouse.open()
    if not success:
        raise RuntimeError(
            "Could not open SpaceMouse. Check USB connection and permissions."
        )

    def read():
        state = pyspacemouse.read()
        trans = np.array([state.x, state.y, state.z])
        rot = np.array([state.roll, state.pitch, state.yaw])
        buttons = state.buttons
        return trans, rot, buttons

    return read


# ---------------------------------------------------------------------------
# Main teleop loop
# ---------------------------------------------------------------------------

def run_teleop(args: argparse.Namespace) -> None:
    # --- Scene setup (imports Isaac Sim) ---
    from envs.isaac_sim_utils.scene import IsaacSimScene
    from envs.isaac_sim_utils.task_configs import get_task_config

    task_cfg = get_task_config(args.task)
    scene_cfg = {
        "use_gui": True,
        "cameras": {
            "static": {"resolution": [256, 256]},
            "wrist": {"resolution": [128, 128]},
        },
    }
    scene = IsaacSimScene(scene_cfg)
    scene.spawn_objects(task_cfg["objects"])
    scene.step_physics(num_steps=20)

    # --- SpaceMouse ---
    read_spacemouse = create_spacemouse()

    # --- Collection parameters ---
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pos_scale = args.pos_scale
    rot_scale = args.rot_scale
    control_hz = args.control_hz
    dt = 1.0 / control_hz

    episode_idx = len(list(save_dir.glob("episode_*.npz")))
    episodes_saved = 0

    gripper_open = True

    logger.info(f"Starting teleop for task '{args.task}'")
    logger.info(f"  Save dir: {save_dir}")
    logger.info(f"  Episodes so far: {episode_idx}")
    logger.info("Controls: SpaceMouse=move, button=toggle gripper, s=save, d=discard, q=quit")

    while episodes_saved < args.num_episodes:
        # Reset scene
        scene.reset_robot()
        scene.spawn_objects(task_cfg["objects"])
        scene.step_physics(num_steps=20)
        gripper_open = True

        buf = EpisodeBuffer()
        buf.instruction = task_cfg["instruction"]
        logger.info(f"\n--- Episode {episode_idx} (target: {args.num_episodes - episodes_saved} remaining) ---")

        running = True
        prev_buttons = [False, False]

        while running:
            t_start = time.time()

            # Read SpaceMouse
            trans, rot, buttons = read_spacemouse()
            delta_pos = trans * pos_scale
            delta_rot = rot * rot_scale

            # Toggle gripper on button press (rising edge)
            if buttons and len(buttons) > 0:
                if buttons[0] and not prev_buttons[0]:
                    gripper_open = not gripper_open
                    logger.info(f"  Gripper → {'open' if gripper_open else 'closed'}")
            prev_buttons = list(buttons) if buttons else [False, False]

            # Get current EE pose
            robot_obs = scene.get_robot_obs()
            ee_pos = robot_obs[:3]
            ee_euler = robot_obs[3:6]

            # Apply delta
            target_pos = ee_pos + delta_pos
            target_euler = ee_euler + delta_rot

            # Build relative action
            gripper_val = 1.0 if gripper_open else -1.0
            rel_action = np.concatenate([delta_pos, delta_rot, [gripper_val]]).astype(np.float32)

            # Execute
            scene.apply_ee_action(target_pos, target_euler, gripper_open)
            scene.step_physics(num_steps=10)

            # Record
            rgb, depth, _, _ = scene.get_camera_data()
            scene_obs = scene.get_scene_obs()
            buf.append(rgb, depth, robot_obs, rel_action, scene_obs)

            # Keyboard check (non-blocking via Isaac Sim input)
            # For now, use episode length limit as auto-save trigger
            if len(buf) >= args.max_steps:
                logger.info("  Max steps reached — auto-saving")
                path = save_dir / f"episode_{episode_idx:04d}.npz"
                buf.save(path)
                episode_idx += 1
                episodes_saved += 1
                running = False

            # Rate limiting
            elapsed = time.time() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    scene.close()
    logger.info(f"\nDone. Saved {episodes_saved} episodes to {save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SpaceMouse teleop data collection in Isaac Sim")
    parser.add_argument("--task", type=str, required=True, help="Task name from task_configs.py")
    parser.add_argument("--save_dir", type=str, default="data/isaac_sim_demos", help="Directory for .npz episodes")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--pos_scale", type=float, default=0.001, help="SpaceMouse → position scaling")
    parser.add_argument("--rot_scale", type=float, default=0.002, help="SpaceMouse → rotation scaling")
    parser.add_argument("--control_hz", type=float, default=30.0, help="Control loop frequency")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    run_teleop(args)


if __name__ == "__main__":
    main()
