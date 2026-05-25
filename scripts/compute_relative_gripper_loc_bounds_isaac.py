"""Compute *relative-frame* gripper_loc_bounds from Isaac Sim HDF5 demos.

Walks every `episode_*.h5` under the given root, reads `ee_pose` (shape
(T, 7) = [pos(3), euler(3), gripper(1)]), and reports the union of the
position displacements the model actually sees when training with
`dataset.relative_action=True` + `policy.relative=True`:

  1. Trajectory deltas: ee_pose[t + k, :3] − ee_pose[t, :3] for k ∈ [1, traj_len]
  2. Gripper-history deltas: ee_pose[t − k, :3] − ee_pose[t, :3] for k ∈ [1, nhist-1]
  3. Action-target deltas: ee_pose[t + execute_every, :3] − ee_pose[t, :3]

PCD is not scanned — distant background points are intentionally clipped by
`DiffuserActor.normalize_pos`.

Usage:
    uv run python scripts/compute_relative_gripper_loc_bounds_isaac.py \\
        --data_dir /home/clear/Documents/michal/isaac_sim_demos \\
        --margin 0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Root containing episode_*.h5 (recurses into training/ and validation/).")
    parser.add_argument("--traj_len", type=int, default=20,
                        help="Future-horizon length the model sees (= interpolation_length).")
    parser.add_argument("--nhist", type=int, default=3,
                        help="Gripper-history length used at training time.")
    parser.add_argument("--execute_every", type=int, default=4,
                        help="Lookahead used to build the keypose action target.")
    parser.add_argument("--margin", type=float, default=0.10,
                        help="Per-side margin (metres) added to the raw union bounds.")
    args = parser.parse_args()

    traj_deltas: list[np.ndarray] = []
    hist_deltas: list[np.ndarray] = []
    act_deltas: list[np.ndarray] = []
    n_ep = 0
    n_frames = 0

    for ep in sorted(args.data_dir.rglob("episode_*.h5")):
        with h5py.File(ep, "r") as f:
            if "ee_pose" not in f:
                print(f"  SKIP {ep.name}: no ee_pose dataset")
                continue
            ee = f["ee_pose"][:, :3].astype(np.float64)  # (T, 3)
        T = ee.shape[0]
        if T < max(args.nhist, 2) + max(args.execute_every, args.traj_len):
            continue

        for t in range(T):
            # Trajectory: t+1 … t+traj_len, clipped to T-1.
            for k in range(1, args.traj_len + 1):
                src = min(t + k, T - 1)
                traj_deltas.append(ee[src] - ee[t])

            # History: t-1 … t-(nhist-1), clipped to 0.
            for k in range(1, args.nhist):
                src = max(t - k, 0)
                hist_deltas.append(ee[src] - ee[t])

            # Action target.
            src = min(t + args.execute_every, T - 1)
            act_deltas.append(ee[src] - ee[t])

        n_ep += 1
        n_frames += T

    if not traj_deltas:
        print(f"ERROR: no usable episode_*.h5 files under {args.data_dir}")
        return

    traj = np.stack(traj_deltas, axis=0)
    hist = np.stack(hist_deltas, axis=0)
    act = np.stack(act_deltas, axis=0)

    def _stats(name: str, t: np.ndarray) -> None:
        lo, hi = t.min(0), t.max(0)
        p_lo = np.quantile(t, 0.005, axis=0)
        p_hi = np.quantile(t, 0.995, axis=0)
        print(f"--- {name}  (n={len(t)}) ---")
        print(f"  min     : [{lo[0]: .4f}, {lo[1]: .4f}, {lo[2]: .4f}]")
        print(f"  max     : [{hi[0]: .4f}, {hi[1]: .4f}, {hi[2]: .4f}]")
        print(f"  0.5–99.5%: [{p_lo[0]: .4f}, {p_lo[1]: .4f}, {p_lo[2]: .4f}] "
              f"→ [{p_hi[0]: .4f}, {p_hi[1]: .4f}, {p_hi[2]: .4f}]")

    print(f"Scanned {n_ep} episodes / {n_frames} frames "
          f"(traj_len={args.traj_len}, nhist={args.nhist}, execute_every={args.execute_every})")
    print()
    _stats("trajectory deltas (ee[t+k] - ee[t])", traj)
    _stats("gripper-history deltas (ee[t-k] - ee[t])", hist)
    _stats("action-target deltas (ee[t+execute_every] - ee[t])", act)

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
    print("YAML snippet:")
    print(f"  gripper_loc_bounds: [[{lo_m[0]:.4f}, {lo_m[1]:.4f}, {lo_m[2]:.4f}], "
          f"[{hi_m[0]:.4f}, {hi_m[1]:.4f}, {hi_m[2]:.4f}]]")


if __name__ == "__main__":
    main()
