"""Compute gripper_loc_bounds from a preprocessed .dat dataset.

Scans all ann_*.dat files under --dat_dir (both training/ and validation/
splits), collects every absolute gripper position and action-target position,
and reports the recommended gripper_loc_bounds with a configurable margin.

These bounds must cover:
  1. All gripper history positions   (curr_gripper[:, :, :3] in model)
  2. All action/keyframe positions   (gt_trajectory[:, :, :3] in model)
  3. All PCD XYZ values that matter  (pcd_obs in model — handled by the
     clamp added to DiffuserActor.normalize_pos; bounds only need to cover
     the workspace, not distant background points)

Usage:
    uv run python scripts/compute_gripper_loc_bounds.py \\
        --dat_dir /home/clear/Documents/michal/realworld_3da \\
        --margin 0.3
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import blosc
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dat_dir", type=Path, required=True,
                        help="Root of the preprocessed dataset (contains training/ and/or validation/)")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Extra margin (metres) added to each side of the raw bounds. Default 0.3.")
    args = parser.parse_args()

    mins, maxs = [], []
    n_dat = 0

    for dat in sorted(args.dat_dir.rglob("ann_*.dat")):
        episode = pickle.loads(blosc.decompress(dat.read_bytes()))
        # episode[4]: gripper_tensors — list of (1, 7) tensors [pos(3), quat(4)]
        for g in episode[4]:
            pos = g[0, :3]
            mins.append(pos)
            maxs.append(pos)
        # episode[2]: action_tensors — list of (1, 7) tensors [pos(3), euler(3), gripper(1)]
        for a in episode[2]:
            pos = a[0, :3]
            mins.append(pos)
            maxs.append(pos)
        n_dat += 1

    if not mins:
        print(f"ERROR: no ann_*.dat files found under {args.dat_dir}")
        return

    lo = torch.stack(mins).min(0).values
    hi = torch.stack(maxs).max(0).values
    lo_m = lo - args.margin
    hi_m = hi + args.margin

    print(f"Scanned {n_dat} .dat files")
    print(f"Raw gripper+action pos min: [{lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}]")
    print(f"Raw gripper+action pos max: [{hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f}]")
    print()
    print(f"Recommended gripper_loc_bounds (margin={args.margin} m):")
    print(f"  min: [{lo_m[0]:.4f}, {lo_m[1]:.4f}, {lo_m[2]:.4f}]")
    print(f"  max: [{hi_m[0]:.4f}, {hi_m[1]:.4f}, {hi_m[2]:.4f}]")
    print()
    print("YAML snippet:")
    print(f"  gripper_loc_bounds: [[{lo_m[0]:.4f}, {lo_m[1]:.4f}, {lo_m[2]:.4f}], "
          f"[{hi_m[0]:.4f}, {hi_m[1]:.4f}, {hi_m[2]:.4f}]]")


if __name__ == "__main__":
    main()
