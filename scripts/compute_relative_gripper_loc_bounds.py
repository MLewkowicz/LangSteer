"""Compute *relative-frame* gripper_loc_bounds from real-world .dat shards.

Mirror of `compute_relative_gripper_loc_bounds_isaac.py` but for the packaged
real-world dataset (output of `convert_realworld_for_diffuser_actor.py`),
not raw Isaac HDF5. Operates on the same union of delta sources the model
sees during training with `dataset.relative_action=True` + `policy.relative=True`:

  1. Trajectory deltas: trajectories[i][k, :3] - trajectories[i][0, :3]
     for each (kept-)frame i and each waypoint k in [1, len-1].
  2. Gripper-history deltas: gripper_tensors[i-k][0, :3] - gripper_tensors[i][0, :3]
     for k in [1, nhist-1], clipped at 0.
  3. Action-target delta: action_tensors[i][0, :3] - gripper_tensors[i][0, :3].

PCD is intentionally not scanned; distant background points are clipped by
DiffuserActor.normalize_pos.

Usage:
    uv run python scripts/compute_relative_gripper_loc_bounds.py \\
        --dat_dir /home/clear/Documents/michal/realworld_3da_v2 \\
        --nhist 3 --margin 0.10
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import blosc
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dat_dir", type=Path, required=True,
                        help="Root containing ann_*.dat (recurses into training/ and validation/).")
    parser.add_argument("--nhist", type=int, default=3,
                        help="Gripper-history length used at training time (matches policy.nhist).")
    parser.add_argument("--margin", type=float, default=0.10,
                        help="Per-side margin (metres) added to the raw union bounds.")
    parser.add_argument("--horizon_frames", type=int, default=None,
                        help="If set, compute trajectory deltas over fixed-horizon "
                             "sliding windows (matching dataset.type=realworld with "
                             "this horizon_frames) instead of per-keypose intervals. "
                             "MUST match the training config's dataset.horizon_frames.")
    args = parser.parse_args()

    traj_deltas: list[np.ndarray] = []
    hist_deltas: list[np.ndarray] = []
    act_deltas: list[np.ndarray] = []
    n_dat = 0

    def _reconstruct_dense(trajs):
        """Stitch per-keypose trajectories into one dense path, dropping the
        shared boundary frame. Mirrors RealworldSlidingWindowDataset."""
        parts = [trajs[0].numpy()]
        anchor_start = [0]
        cum = trajs[0].shape[0] - 1
        for m in range(1, len(trajs)):
            anchor_start.append(cum)
            parts.append(trajs[m][1:].numpy())
            cum += trajs[m].shape[0] - 1
        return np.concatenate(parts, axis=0), anchor_start

    for dat in sorted(args.dat_dir.rglob("ann_*.dat")):
        # Layout: [frame_ids, rgb_pcd, action_tensors, camera_dicts,
        #          gripper_tensors, trajectories, ann_ids]
        episode = pickle.loads(blosc.decompress(dat.read_bytes()))
        action_tensors = episode[2]
        gripper_tensors = episode[4]
        trajectories = episode[5]
        n_frames = len(gripper_tensors)

        # Each tensor is (1, 7) [pos(3), euler(3), gripper(1)]; trajectories[i]
        # is (L_i, 7) with the same layout (first frame == current).
        gripper_pos = np.stack([g[0, :3].numpy() for g in gripper_tensors], axis=0)
        action_pos = np.stack([a[0, :3].numpy() for a in action_tensors], axis=0)

        dense = anchor_start = None
        if args.horizon_frames is not None and len(trajectories) > 0:
            dense, anchor_start = _reconstruct_dense(trajectories)

        for i in range(n_frames):
            base = gripper_pos[i]

            if args.horizon_frames is not None and dense is not None:
                # Sliding-window trajectory deltas: window[k] - window[0].
                start = anchor_start[i]
                window = dense[start:start + args.horizon_frames]
                if window.shape[0] >= 2:
                    for k in range(1, window.shape[0]):
                        traj_deltas.append(window[k, :3] - window[0, :3])
            else:
                # Per-keypose-interval deltas (matches dataset.type=calvin).
                traj = trajectories[i].numpy()
                if traj.ndim == 2 and traj.shape[0] >= 2:
                    for k in range(1, traj.shape[0]):
                        traj_deltas.append(traj[k, :3] - traj[0, :3])

            for k in range(1, args.nhist):
                src = max(i - k, 0)
                hist_deltas.append(gripper_pos[src] - base)

            act_deltas.append(action_pos[i] - base)

        n_dat += 1

    if not traj_deltas:
        print(f"ERROR: no ann_*.dat files found under {args.dat_dir}")
        return

    traj = np.stack(traj_deltas, axis=0)
    hist = np.stack(hist_deltas, axis=0)
    act = np.stack(act_deltas, axis=0)

    def _stats(name: str, t: np.ndarray) -> None:
        lo, hi = t.min(0), t.max(0)
        p_lo = np.quantile(t, 0.005, axis=0)
        p_hi = np.quantile(t, 0.995, axis=0)
        print(f"--- {name}  (n={len(t)}) ---")
        print(f"  min      : [{lo[0]: .4f}, {lo[1]: .4f}, {lo[2]: .4f}]")
        print(f"  max      : [{hi[0]: .4f}, {hi[1]: .4f}, {hi[2]: .4f}]")
        print(f"  0.5-99.5%: [{p_lo[0]: .4f}, {p_lo[1]: .4f}, {p_lo[2]: .4f}] "
              f"-> [{p_hi[0]: .4f}, {p_hi[1]: .4f}, {p_hi[2]: .4f}]")

    print(f"Scanned {n_dat} .dat files (nhist={args.nhist})")
    print()
    _stats("trajectory deltas (traj[k] - traj[0])", traj)
    _stats("gripper-history deltas (g[i-k] - g[i])", hist)
    _stats("action-target deltas (action[i] - g[i])", act)

    all_d = np.concatenate([traj, hist, act], axis=0)
    lo = all_d.min(0)
    hi = all_d.max(0)
    lo_m = lo - args.margin
    hi_m = hi + args.margin

    print()
    print(f"=== union, margin={args.margin} m ===")
    print(f"  raw min: [{lo[0]: .4f}, {lo[1]: .4f}, {lo[2]: .4f}]")
    print(f"  raw max: [{hi[0]: .4f}, {hi[1]: .4f}, {hi[2]: .4f}]")
    print()
    print("YAML snippet (paste into BOTH training and deploy yamls):")
    print(f"  gripper_loc_bounds: [[{lo_m[0]:.4f}, {lo_m[1]:.4f}, {lo_m[2]:.4f}], "
          f"[{hi_m[0]:.4f}, {hi_m[1]:.4f}, {hi_m[2]:.4f}]]")


if __name__ == "__main__":
    main()
