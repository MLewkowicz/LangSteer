"""Offline diagnostic: does the trained model differentiate primitive=0 vs primitive=1?

Loads best.pth, picks one validation .dat per primitive, builds the same
observation tensors the deploy script would (PCD, RGB, gripper history,
primitive+object ids), and runs the model with primitive_id forced to 0 then
1 on the *same* observation. If the two predicted trajectories are nearly
identical, the primitive embedding is not influencing the output and we have
a conditioning bug. If they're meaningfully different, the model is fine and
the deploy-time symptom is on the deploy side (camera frame, ee pose, etc.).

Usage:
    uv run python scripts/diagnose_inference_conditioning.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import blosc
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policies.diffuser_actor import build_diffuser_actor_policy
from core.types import Observation


CKPT = Path("outputs/checkpoints/diffuser_actor_realworld_primitive_object_v2/best.pth")
POLICY_CFG = Path("conf/policy/diffuser_actor_realworld_primitive_object.yaml")
VAL_DIR = Path("/home/clear/Documents/michal/realworld_3da/validation/D+0")
VAL_ANN = Path("/home/clear/Documents/michal/realworld_3da/validation/lang_annotations/primitive_object_lang_ann.npy")


def _episode_obs(ep, frame_idx: int) -> Observation:
    """Build an Observation from a packaged .dat at a given frame index."""
    # ep[1] shape: (T, 2 cameras, 2 modes, 3, H, W); cameras=[front, wrist], modes=[rgb, pcd]
    rgbpcd = ep[1][frame_idx]                                  # (2, 2, 3, H, W)
    rgb_front = ((rgbpcd[0, 0].numpy() + 1.0) * 127.5).astype(np.uint8).transpose(1, 2, 0)
    rgb_wrist = ((rgbpcd[1, 0].numpy() + 1.0) * 127.5).astype(np.uint8).transpose(1, 2, 0)
    pcd_front = rgbpcd[0, 1].numpy().transpose(1, 2, 0).astype(np.float32)
    pcd_wrist = rgbpcd[1, 1].numpy().transpose(1, 2, 0).astype(np.float32)

    # ep[4] gripper_tensors are (1, 7) = [pos(3), euler_XYZ(3), gripper(1)]
    g = ep[4][frame_idx].numpy().flatten()
    ee_pose = g.astype(np.float32)

    return Observation(
        rgb={"front": rgb_front, "wrist": rgb_wrist},
        depth={"front": pcd_front, "wrist": pcd_wrist},
        proprio=np.zeros(0, dtype=np.float32),
        ee_pose=ee_pose,
        instruction="",
    )


def _pick_one_per_primitive(ann_path: Path, dat_dir: Path) -> dict[str, Path]:
    ann = np.load(ann_path, allow_pickle=True).item()
    prims = ann["info"]["primitive"]
    out = {}
    for i, p in enumerate(prims):
        if p in out:
            continue
        f = dat_dir / f"ann_{i}.dat"
        if f.exists():
            out[p] = f
        if len(out) == 2:
            break
    return out


def main() -> int:
    cfg = OmegaConf.load(POLICY_CFG)
    cfg.ckpt_path = str(CKPT)
    policy = build_diffuser_actor_policy(cfg)
    policy.load_checkpoint(str(CKPT))

    picks = _pick_one_per_primitive(VAL_ANN, VAL_DIR)
    print(f"Sampled .dat files: {picks}")

    for primitive_label, dat in picks.items():
        ep = pickle.loads(blosc.decompress(dat.read_bytes()))
        n_frames = len(ep[4])
        # Pick a frame ~1/4 into the episode (skip very-start padding).
        frame_idx = max(0, n_frames // 4)
        obs = _episode_obs(ep, frame_idx)

        print(f"\n=== File: {dat.name} (true primitive='{primitive_label}'), frame {frame_idx}/{n_frames} ===")
        print(f"  ee_pose: pos={obs.ee_pose[:3]} euler={obs.ee_pose[3:6]} grip={obs.ee_pose[6]:.3f}")
        print(f"  GT action target (next-keypose): {ep[2][frame_idx].numpy().flatten()}")
        # Ground truth dense trajectory shape:
        gt_traj = ep[5][frame_idx].numpy()
        print(f"  GT dense traj shape: {gt_traj.shape}  start->end pos delta: {gt_traj[-1, :3] - gt_traj[0, :3]}")

        results = {}
        for force_prim in (0, 1):
            policy.reset()
            policy.set_primitive(force_prim)
            policy.set_object(0)
            action = policy.forward(obs)
            traj = action.trajectory
            results[force_prim] = traj
            print(f"  [primitive={force_prim}] traj[0] = {traj[0]}")
            print(f"  [primitive={force_prim}] traj[-1] = {traj[-1]}")
            print(f"  [primitive={force_prim}] start->end pos delta = {traj[-1, :3] - traj[0, :3]}")
            print(f"  [primitive={force_prim}] gripper bit per step = {traj[:, 6].astype(int).tolist()}")

        # Compare conditioning effect
        diff_pos = float(np.abs(results[0][..., :3] - results[1][..., :3]).mean())
        diff_rot = float(np.abs(results[0][..., 3:6] - results[1][..., 3:6]).mean())
        diff_grip = float(np.abs(results[0][..., 6] - results[1][..., 6]).mean())
        print(f"  mean |p0-p1| pos={diff_pos:.4f} m  rot={diff_rot:.4f} rad  grip={diff_grip:.3f}")
        if diff_pos < 1e-3 and diff_rot < 1e-3:
            print("  >>> WARNING: primitive=0 and primitive=1 produce nearly IDENTICAL trajectories.")
            print("      Primitive embedding is not influencing the output — conditioning failed to train.")
        else:
            print("  Conditioning is active (outputs differ).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
