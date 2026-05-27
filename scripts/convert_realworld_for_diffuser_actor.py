"""Convert real-world Franka teleop data → 3D Diffuser Actor packaged format.

The collected data lives under a directory with one episode per triple:
    episode_<ts>.h5                       (state + camera timestamps + extrinsics)
    episode_<ts>_hand_video.hdf5          (wrist ZED RGB + depth)
    episode_<ts>_third_person_video.hdf5  (overhead RGB + depth)

This script:
  1. Reads state (ee_pos, ee_rot, gripper_open, timestamps).
  2. For each state index, picks the nearest camera frame via timestamps.
  3. Unprojects each camera's depth into the robot base frame using the
     intrinsics + extrinsics stored at the JSON paths supplied on the CLI
     (or in the h5 root attrs as a fallback).
  4. Center-crops + resizes RGB/depth to 200×200 (the size CalvinDataset
     expects before its 20:180 crop down to 160×160).
  5. Builds a 7-D proprio = [ee_x, ee_y, ee_z, euler_X, euler_Y, euler_Z, gripper]
     where euler is pytorch3d's "XYZ" intrinsic Euler convention so that
     `convert_rotation` in calvin_utils reconstructs the correct quaternion.
  6. Runs CALVIN's keypoint_discovery, then packages the 7-tuple
     (frame_ids, rgb_pcd, action_tensors, camera_dicts, gripper_tensors,
      trajectories, annotation_id) into a blosc-compressed .dat file —
     identical schema to package_calvin.py.
  7. Emits a primitive_object_lang_ann.npy whose (start, end) ranges index
     the .dat files (one annotation per episode), with `info.primitive` and
     `info.object` arrays populated from a small per-episode JSON.

Annotation JSON format (one entry per episode_*.h5 basename, without .h5):
    {
      "episode_20260522_163745": {
        "primitive": "grasp",
        "object": "red_block",
        "split": "training"  // optional; defaults below
      },
      ...
    }

Allowed primitive vocab: grasp/push/pull/place/rotate.
Allowed object vocab: block, blue_block, drawer_handle, led_button,
lightbulb_switch, pink_block, red_block, slider_handle.

Example:
    uv run python scripts/convert_realworld_for_diffuser_actor.py \\
        --raw_dir /home/clear/Documents/michal/replayed \\
        --annotations /home/clear/Documents/michal/realworld_annotations.json \\
        --save_path /home/clear/Documents/michal/realworld_3da \\
        --val_fraction 0.15
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import blosc
import cv2
import h5py
import numpy as np
import torch

# Make `training.*` and `policies.*` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.policies.diffuser_actor.preprocessing.calvin_utils import (
    keypoint_discovery,
)
from training.policies.diffuser_actor.preprocessing.pytorch3d_transforms import (
    matrix_to_euler_angles,
)


# The converter only validates that primitive labels are in this canonical set
# (these are the well-defined manipulation primitives the model architecture
# supports). Object labels are NOT validated here — the trainer config owns the
# object vocabulary (`policy.object_vocab`), so any string you emit must match
# a key in that override (or in trainer.OBJECT_VOCAB when no override is set).
PRIMITIVE_VOCAB = {"grasp": 0, "push": 1, "pull": 2, "place": 3, "rotate": 4}

IMG_SIZE = 200  # output H = W (matches CALVIN package_calvin output)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def parse_extrinsics(json_blob_or_path: str) -> dict:
    """Accept either a JSON string (from h5 attrs) or a file path."""
    if json_blob_or_path.strip().startswith("{"):
        return json.loads(json_blob_or_path)
    with open(json_blob_or_path) as f:
        return json.load(f)


def load_extrinsics_for_episode(state_h5: h5py.File,
                                hand_json: Optional[Path],
                                tp_json: Optional[Path]) -> tuple[dict, dict]:
    """Per-episode extrinsics. Prefer the JSON paths the user passed (they
    were calibrated separately); fall back to the JSON blobs the collection
    code dumped into the h5 root attrs."""
    if hand_json is not None:
        hand = parse_extrinsics(str(hand_json))
    else:
        hand = parse_extrinsics(state_h5.attrs["extrinsics_hand"])
    if tp_json is not None:
        tp = parse_extrinsics(str(tp_json))
    else:
        tp = parse_extrinsics(state_h5.attrs["extrinsics_third_person"])
    return hand, tp


def center_square_crop_resize(img: np.ndarray, out_size: int,
                              interp: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Center-crop to a square, then resize to (out_size, out_size).

    Returns (resized, (crop_offset_x, crop_size)) so the intrinsics can be
    adjusted to match the crop+resize.
    """
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    resized = cv2.resize(crop, (out_size, out_size), interpolation=interp)
    return resized, (x0, y0, side)


def adjust_K_for_crop_resize(K: np.ndarray, crop_x: int, crop_y: int,
                              crop_size: int, out_size: int) -> np.ndarray:
    """Adjust a 3×3 intrinsic matrix to match a center-crop + isotropic resize.

    Effect on intrinsics: first subtract the crop origin from (cx, cy), then
    scale (fx, fy, cx, cy) by out_size / crop_size.
    """
    K = K.copy()
    K[0, 2] -= crop_x
    K[1, 2] -= crop_y
    s = out_size / crop_size
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] *= s
    K[1, 2] *= s
    return K


def depth_to_camera_xyz(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unproject a HxW depth map (meters, NaN allowed) using intrinsics K.

    Returns (H, W, 3) XYZ in the camera frame using the standard pinhole
    convention (+Z forward, +X right, +Y down). NaNs become 0.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64))
    z = np.nan_to_num(depth.astype(np.float64), nan=0.0,
                      posinf=0.0, neginf=0.0)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def transform_xyz(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous transform to an (H, W, 3) point cloud."""
    h, w, _ = xyz.shape
    flat = xyz.reshape(-1, 3)
    hom = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    out = (T @ hom.T).T[:, :3]
    return out.reshape(h, w, 3)


def make_T_gripper_to_base(ee_pos: np.ndarray, ee_rot: np.ndarray) -> np.ndarray:
    """Build a 4×4 transform from gripper (end-effector) frame to base frame
    out of an ee_pos (3,) and ee_rot (3,3)."""
    T = np.eye(4)
    T[:3, :3] = ee_rot
    T[:3, 3] = ee_pos
    return T


def nearest_frame_indices(state_ts: np.ndarray,
                          cam_ts: np.ndarray) -> np.ndarray:
    """For each state timestamp, return the index of the closest camera
    timestamp (1D int array of length len(state_ts))."""
    # cam_ts is monotonically non-decreasing in our data; use searchsorted.
    idx = np.searchsorted(cam_ts, state_ts)
    idx = np.clip(idx, 1, len(cam_ts) - 1)
    left = cam_ts[idx - 1]
    right = cam_ts[idx]
    choose_left = (state_ts - left) <= (right - state_ts)
    out = np.where(choose_left, idx - 1, idx)
    return out


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

def detect_gripper_segments(gripper: np.ndarray,
                            post_open_frames: int = 5) -> list[tuple[int, int]]:
    """Detect (start, end) frame index pairs for pick-and-place episodes.

    Assumes the episode starts with the gripper open, closes once around the
    target object, optionally opens once to release it, and is then truncated.
    Returns:
        [(grasp_start, grasp_end), (place_start, place_end)]   if both events
        [(grasp_start, grasp_end)]                              if no release
        []                                                      if no close
    The endpoints are *inclusive*: grasp_end is the first frame where the
    gripper is observed closed; place_end is post_open_frames past the first
    frame where the gripper is observed open again, so the release motion is
    fully captured in the training segment.
    """
    diffs = np.diff(gripper.astype(np.float64))
    closes = np.where(diffs < -0.5)[0]
    opens = np.where(diffs > 0.5)[0]
    if len(closes) == 0:
        return []
    close_idx = int(closes[0])             # last frame with gripper OPEN
    grasp_end = close_idx + 1              # first frame with gripper CLOSED
    grasp_seg = (0, grasp_end)
    # Find the first open *after* the close (ignores any spurious early opens).
    opens_after = opens[opens > close_idx]
    if len(opens_after) == 0:
        return [grasp_seg]
    open_idx = int(opens_after[0])         # last frame with gripper CLOSED
    # Extend slightly past the open command so the release motion is included.
    place_end = min(open_idx + 1 + post_open_frames, len(gripper) - 1)
    place_seg = (grasp_end, place_end)     # share the close frame as boundary
    return [grasp_seg, place_seg]


def _process_segment(state_ts: np.ndarray,
                     ee_pos: np.ndarray,
                     ee_rot: np.ndarray,
                     gripper: np.ndarray,
                     hand_cam_ts: np.ndarray,
                     tp_cam_ts: np.ndarray,
                     hand_ext: dict,
                     tp_ext: dict,
                     state_stride: int,
                     hand_path: Path,
                     tp_path: Path,
                     out_dat_path: Path,
                     annotation_id: int) -> int:
    """Package a single (already-sliced) state segment into a .dat file.

    Returns the number of keyframes written.
    """
    # Subsample to make episodes tractable (preserves keypose detection).
    # Always include the final frame — for auto-segmented episodes the closing
    # / opening transition lands on the last index, and dropping it would hide
    # the gripper command from the action target.
    sel = np.arange(0, len(state_ts), state_stride)
    if len(state_ts) > 0 and sel[-1] != len(state_ts) - 1:
        sel = np.append(sel, len(state_ts) - 1)
    state_ts = state_ts[sel]
    ee_pos = ee_pos[sel]
    ee_rot = ee_rot[sel]
    gripper = gripper[sel]

    hand_idx = nearest_frame_indices(state_ts, hand_cam_ts)
    tp_idx = nearest_frame_indices(state_ts, tp_cam_ts)

    K_hand = np.asarray(hand_ext["intrinsics"]["K"], dtype=np.float64)
    K_tp = np.asarray(tp_ext["intrinsics"]["K"], dtype=np.float64)
    T_cam2gripper = np.asarray(hand_ext["T_cam2gripper"], dtype=np.float64)
    T_cam2base_tp = np.asarray(tp_ext["T_cam2base"], dtype=np.float64)

    static_rgbs: list[np.ndarray] = []
    static_pcds: list[np.ndarray] = []
    gripper_rgbs: list[np.ndarray] = []
    gripper_pcds: list[np.ndarray] = []
    proprios: list[np.ndarray] = []

    # Compute euler once with pytorch3d's "XYZ" intrinsic convention to match
    # what CalvinDataset.convert_rotation reconstructs at load time.
    ee_rot_t = torch.from_numpy(ee_rot).to(torch.float64)
    euler_t = matrix_to_euler_angles(ee_rot_t, "XYZ").numpy()  # (N, 3)

    # Pre-adjust intrinsics for the crop+resize we apply below. We need a
    # representative image to know its raw H/W; assume both cameras are
    # 720×1280 (HD720) per the collection metadata.
    raw_h, raw_w = 720, 1280
    crop_x = (raw_w - raw_h) // 2  # center-square crop along x (1280 → 720)
    crop_y = 0
    side = raw_h
    K_hand_adj = adjust_K_for_crop_resize(K_hand, crop_x, crop_y, side, IMG_SIZE)
    K_tp_adj = adjust_K_for_crop_resize(K_tp, crop_x, crop_y, side, IMG_SIZE)

    with h5py.File(hand_path, "r") as fh, h5py.File(tp_path, "r") as ft:
        rgb_hand = fh["rgb"]
        depth_hand = fh["depth"]
        rgb_tp = ft["rgb"]
        depth_tp = ft["depth"]
        for i in range(len(state_ts)):
            hi = int(hand_idx[i])
            ti = int(tp_idx[i])

            # --- third-person ("static") ---
            tp_rgb_full = rgb_tp[ti]
            tp_depth_full = depth_tp[ti]
            tp_rgb_crop, _ = center_square_crop_resize(
                tp_rgb_full, IMG_SIZE, cv2.INTER_AREA
            )
            tp_depth_crop, _ = center_square_crop_resize(
                tp_depth_full, IMG_SIZE, cv2.INTER_NEAREST
            )
            tp_cam_xyz = depth_to_camera_xyz(tp_depth_crop, K_tp_adj)
            tp_base_xyz = transform_xyz(tp_cam_xyz, T_cam2base_tp)

            # --- hand ("gripper") ---
            h_rgb_full = rgb_hand[hi]
            h_depth_full = depth_hand[hi]
            h_rgb_crop, _ = center_square_crop_resize(
                h_rgb_full, IMG_SIZE, cv2.INTER_AREA
            )
            h_depth_crop, _ = center_square_crop_resize(
                h_depth_full, IMG_SIZE, cv2.INTER_NEAREST
            )
            h_cam_xyz = depth_to_camera_xyz(h_depth_crop, K_hand_adj)
            # gripper-frame XYZ first, then base-frame via T_gripper2base(t).
            h_gripper_xyz = transform_xyz(h_cam_xyz, T_cam2gripper)
            T_g2b = make_T_gripper_to_base(ee_pos[i], ee_rot[i])
            h_base_xyz = transform_xyz(h_gripper_xyz, T_g2b)

            # CalvinDataset expects RGB normalised to [-1, 1].
            static_rgbs.append(tp_rgb_crop.astype(np.float32) / 255.0 * 2 - 1)
            static_pcds.append(tp_base_xyz.astype(np.float32))
            gripper_rgbs.append(h_rgb_crop.astype(np.float32) / 255.0 * 2 - 1)
            gripper_pcds.append(h_base_xyz.astype(np.float32))

            proprios.append(np.concatenate([
                ee_pos[i].astype(np.float32),
                euler_t[i].astype(np.float32),
                np.array([1.0 if gripper[i] > 0.5 else 0.0], dtype=np.float32),
            ]))

    # Stack as (T, 2, 2, 3, H, W): cameras=[static, gripper], modes=[RGB, PCD]
    static_rgb = np.stack(static_rgbs, axis=0)    # (T, H, W, 3)
    static_pcd = np.stack(static_pcds, axis=0)
    gripper_rgb = np.stack(gripper_rgbs, axis=0)
    gripper_pcd = np.stack(gripper_pcds, axis=0)
    rgb = np.stack([static_rgb, gripper_rgb], axis=1)   # (T, 2, H, W, 3)
    pcd = np.stack([static_pcd, gripper_pcd], axis=1)
    rgb_pcd = np.stack([rgb, pcd], axis=2)              # (T, 2, 2, H, W, 3)
    rgb_pcd = rgb_pcd.transpose(0, 1, 2, 5, 3, 4)       # (T, 2, 2, 3, H, W)
    rgb_pcd_t = torch.as_tensor(rgb_pcd, dtype=torch.float32)

    # Keypose discovery on the proprio trajectory.
    _, keyframe_inds = keypoint_discovery(proprios)
    keyframe_inds = np.asarray(keyframe_inds)

    # Map every gripper index → next keyframe index (action target).
    keyframe_indices = torch.as_tensor(keyframe_inds)[None, :]
    gripper_indices = torch.arange(len(proprios)).view(-1, 1)
    action_indices = torch.argmax(
        (gripper_indices < keyframe_indices).float(), dim=1
    ).tolist()
    action_indices[-1] = len(keyframe_inds) - 1
    actions = [proprios[keyframe_inds[i]] for i in action_indices]
    action_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1) for a in actions
    ]

    # camera_dicts: use ("front", "wrist") so the trainer's
    # cameras=("front","wrist") reordering is a no-op.
    camera_dicts = [{"front": (0, 0), "wrist": (0, 0)}]

    gripper_tensors = [
        torch.as_tensor(p, dtype=torch.float32).view(1, -1) for p in proprios
    ]

    # Keypose-mode trajectories: from current frame to its target keyframe.
    trajectories = []
    for i in range(len(action_indices)):
        target_frame = keyframe_inds[action_indices[i]]
        trajectories.append(torch.cat([
            torch.as_tensor(p, dtype=torch.float32).view(1, -1)
            for p in proprios[i:target_frame + 1]
        ], dim=0))

    # Thin to keyframe indices (prepend 0 so the first frame is included).
    kept = [0] + keyframe_inds[:-1].tolist()
    kept_t = torch.as_tensor(kept)
    rgb_pcd_t = torch.index_select(rgb_pcd_t, 0, kept_t)
    action_tensors = [action_tensors[i] for i in kept]
    gripper_tensors = [gripper_tensors[i] for i in kept]
    trajectories = [trajectories[i] for i in kept]

    frame_ids = list(range(rgb_pcd_t.shape[0]))
    state_dict = [
        frame_ids,
        rgb_pcd_t,
        action_tensors,
        camera_dicts,
        gripper_tensors,
        trajectories,
        [annotation_id] * len(frame_ids),  # per-frame annotation id
    ]
    out_dat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_dat_path, "wb") as f:
        f.write(blosc.compress(pickle.dumps(state_dict)))

    return len(frame_ids)


# Effective post-stride state rate to target. Chosen to match v0's per-keypose
# traversal distribution: v0 (28 eps) had interval-span median 50.8mm / p75
# 151.8mm at its native ~5 Hz effective. Because v1 teleop covered more distance
# per reach, matching v0's *traversal* (not just its rate) requires a HIGHER
# effective rate: ~10 Hz on v1 reproduces v0's p75 tail (140mm vs 152mm), while
# 5 Hz over-fattens the tail (p75 222mm) and blows out the gripper_loc_bounds.
# Measured via the stride sweep in the session notes; see DIFFUSER_ACTOR_DEPLOY_DEBUG.md.
TARGET_EFFECTIVE_RATE_HZ = 10.0


def _auto_state_stride(state_ts: np.ndarray, default_stride: int = 2) -> int:
    """Pick a stride that brings state_ts down to ~TARGET_EFFECTIVE_RATE_HZ.

    Returns the larger of `default_stride` and the rate-derived stride, so
    legacy low-rate recordings keep their original stride.
    """
    if len(state_ts) < 2:
        return default_stride
    duration = float(state_ts[-1] - state_ts[0])
    if duration <= 0:
        return default_stride
    actual_hz = len(state_ts) / duration
    derived = max(1, int(round(actual_hz / TARGET_EFFECTIVE_RATE_HZ)))
    return max(default_stride, derived)


def _load_cam_ts(state_h5_keys, fs_root, video_path: Path,
                 state_h5_key: str) -> np.ndarray:
    """Get camera timestamps either from the legacy in-state group or, if
    that group is absent (v1+ collections), from the video file's
    `timestamps` dataset."""
    if state_h5_key in state_h5_keys:
        return fs_root[state_h5_key][:]
    with h5py.File(video_path, "r") as fv:
        if "timestamps" not in fv:
            raise RuntimeError(
                f"{video_path.name} has no 'timestamps' dataset and the state "
                f"h5 has no {state_h5_key!r}; cannot align cameras to state."
            )
        return fv["timestamps"][:]


def process_episode(state_path: Path,
                    out_dat_path: Path,
                    annotation_id: int,
                    hand_extrinsics_json: Optional[Path],
                    tp_extrinsics_json: Optional[Path],
                    state_stride: int = 2,
                    frame_range: Optional[tuple[int, int]] = None,
                    auto_state_stride: bool = True) -> int:
    """Convert one episode (or a slice of one) into a single packaged .dat.

    `frame_range`, when provided, is an inclusive (start, end) tuple of indices
    into the *raw* (pre-subsample) state arrays. The returned int is the
    number of keyframes written.

    When `auto_state_stride` is True (the default), the actual stride applied
    is `max(state_stride, derived)` where derived brings the state rate down
    to ~TARGET_EFFECTIVE_RATE_HZ. Pass `auto_state_stride=False` to honour the
    caller's stride verbatim.
    """
    hand_path = state_path.with_name(state_path.stem + "_hand_video.hdf5")
    tp_path = state_path.with_name(state_path.stem + "_third_person_video.hdf5")

    with h5py.File(state_path, "r") as fs:
        state_ts = fs["timestamps"][:]
        ee_pos = fs["ee_pos"][:]
        ee_rot = fs["ee_rot"][:]
        gripper = fs["gripper_open"][:]
        keys = set(fs.keys())
        # camera_timestamps used to live as a group in the state h5; newer
        # collections store them only in the *_video.hdf5 files. Tolerate both.
        cam_ts_keys = set(fs["camera_timestamps"].keys()) if "camera_timestamps" in keys else set()
        hand_cam_ts = (fs["camera_timestamps/hand"][:]
                       if "hand" in cam_ts_keys else None)
        tp_cam_ts = (fs["camera_timestamps/third_person"][:]
                     if "third_person" in cam_ts_keys else None)
        hand_ext, tp_ext = load_extrinsics_for_episode(
            fs, hand_extrinsics_json, tp_extrinsics_json
        )

    if hand_cam_ts is None:
        with h5py.File(hand_path, "r") as fv:
            hand_cam_ts = fv["timestamps"][:]
    if tp_cam_ts is None:
        with h5py.File(tp_path, "r") as fv:
            tp_cam_ts = fv["timestamps"][:]

    if frame_range is not None:
        s, e = frame_range
        state_ts = state_ts[s:e + 1]
        ee_pos = ee_pos[s:e + 1]
        ee_rot = ee_rot[s:e + 1]
        gripper = gripper[s:e + 1]

    effective_stride = state_stride
    if auto_state_stride:
        effective_stride = _auto_state_stride(state_ts, default_stride=state_stride)
        if effective_stride != state_stride:
            duration = float(state_ts[-1] - state_ts[0]) if len(state_ts) >= 2 else 0.0
            actual_hz = len(state_ts) / duration if duration > 0 else float("nan")
            print(f"  [auto-stride] {state_path.stem}: state rate ~{actual_hz:.1f} Hz, "
                  f"raising stride {state_stride} -> {effective_stride} "
                  f"to land near {TARGET_EFFECTIVE_RATE_HZ:.1f} Hz")

    return _process_segment(
        state_ts=state_ts,
        ee_pos=ee_pos,
        ee_rot=ee_rot,
        gripper=gripper,
        hand_cam_ts=hand_cam_ts,
        tp_cam_ts=tp_cam_ts,
        hand_ext=hand_ext,
        tp_ext=tp_ext,
        state_stride=effective_stride,
        hand_path=hand_path,
        tp_path=tp_path,
        out_dat_path=out_dat_path,
        annotation_id=annotation_id,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_template_annotations(raw_dir: Path, out_json: Path) -> None:
    """Emit a primitive=grasp / object=block placeholder for each episode."""
    out: dict = {}
    state_files = sorted(raw_dir.glob("episode_*[0-9].h5"))
    for sp in state_files:
        out[sp.stem] = {
            "primitive": "grasp",
            "object": "block",
            "split": "training",
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(
        f"Wrote template annotation file with {len(out)} placeholder entries "
        f"to {out_json}. Edit each entry's primitive/object/split before running "
        "the converter with --annotations."
    )


def split_for_episode(name: str, ann_entry: dict,
                      val_fraction: float, all_names: list[str]) -> str:
    """Use an explicit per-episode split when set, otherwise sample the
    final `val_fraction` of episodes (sorted) into the validation split."""
    if "split" in ann_entry:
        return ann_entry["split"]
    val_count = max(1, int(round(len(all_names) * val_fraction)))
    return "validation" if name in set(all_names[-val_count:]) else "training"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", type=Path, required=True,
                        help="Directory holding episode_*.h5 + *_hand_video.hdf5 + *_third_person_video.hdf5")
    parser.add_argument("--save_path", type=Path, required=True,
                        help="Output root; gets {training,validation}/D+0/ann_*.dat + lang_annotations/")
    parser.add_argument("--annotations", type=Path, default=None,
                        help="JSON mapping episode stem → {primitive, object, split}. "
                             "If omitted with --emit_template, a placeholder is written.")
    parser.add_argument("--emit_template", action="store_true",
                        help="Write a placeholder annotation JSON at --annotations and exit.")
    parser.add_argument("--hand_extrinsics", type=Path, default=None,
                        help="Optional path to extrinsics_hand.json overriding the per-episode root attr.")
    parser.add_argument("--third_person_extrinsics", type=Path, default=None,
                        help="Optional path to extrinsics_third_person.json overriding the per-episode root attr.")
    parser.add_argument("--state_stride", type=int, default=2,
                        help="Subsample factor on the state trajectory before keypose detection. Default 2 (~5Hz from ~10Hz state). With --auto_state_stride (default), the actual stride is max(state_stride, derived) where derived = round(raw_hz / TARGET_EFFECTIVE_RATE_HZ).")
    parser.add_argument("--auto_state_stride", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Auto-raise state_stride to keep the effective post-stride rate near ~5 Hz regardless of how fast state was streamed during collection. Use --no-auto_state_stride to honour --state_stride verbatim (legacy behaviour).")
    parser.add_argument("--val_fraction", type=float, default=0.15,
                        help="Fraction of episodes routed to validation when an entry lacks an explicit `split` field.")
    parser.add_argument("--scene_tag", default="D",
                        help="Output scene folder; the dataset loader globs {scene}+0/*.dat. Default D matches the existing CALVIN configs.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N episodes (debug).")
    parser.add_argument("--post_open_frames", type=int, default=5,
                        help="Extra frames to include after the gripper-open event when "
                             "auto-segmenting the place sub-episode. At state_stride=2 (~5Hz), "
                             "5 frames ≈ 1 second of release motion. Default 5.")
    parser.add_argument("--auto_segment_object", default=None,
                        help="If set, every episode is auto-split by gripper "
                             "transitions into a 'grasp <obj>' segment (until "
                             "the first close) and a 'place <obj>' segment "
                             "(close → first open); anything after the open is "
                             "discarded. Replaces --annotations entirely.")
    args = parser.parse_args()

    state_files = sorted(args.raw_dir.glob("episode_*[0-9].h5"))
    if not state_files:
        print(f"ERROR: no episode_*.h5 files under {args.raw_dir}")
        return 2

    if args.emit_template:
        if args.annotations is None:
            print("--emit_template requires --annotations <path>")
            return 2
        make_template_annotations(args.raw_dir, args.annotations)
        return 0

    auto_segment = args.auto_segment_object is not None
    if auto_segment:
        # In auto mode --annotations is optional; the per-episode split lookup
        # falls back to val_fraction over the alphabetised episode list.
        ann_table = {}
        if args.annotations is not None and args.annotations.is_file():
            with open(args.annotations) as f:
                ann_table = json.load(f)
    else:
        if args.annotations is None or not args.annotations.is_file():
            print("ERROR: pass either --annotations <json> or "
                  "--auto_segment_object <obj>. (Run --emit_template to scaffold.)")
            return 2
        with open(args.annotations) as f:
            ann_table = json.load(f)
        # Validate the annotation file early — fail fast on typos. Object
        # strings are not validated here; the trainer config owns that vocab.
        all_names_check = [p.stem for p in state_files]
        missing = [n for n in all_names_check if n not in ann_table]
        if missing:
            print(f"WARNING: {len(missing)} episodes have no annotation entry; "
                  f"they will be skipped. First few: {missing[:5]}")
        for n, entry in ann_table.items():
            if entry["primitive"] not in PRIMITIVE_VOCAB:
                print(f"ERROR: episode {n} has unknown primitive '{entry['primitive']}'. "
                      f"Allowed: {sorted(PRIMITIVE_VOCAB)}")
                return 2

    if args.limit > 0:
        state_files = state_files[:args.limit]
    all_names = [p.stem for p in state_files]

    # We need a separate annotation counter per split (annotation_id is the
    # index used by the trainer to look up primitive/object ids, and each
    # split has its own ann file).
    split_state: dict[str, dict] = {
        "training": {"counter": 0, "ann": [], "task": [], "indx": [],
                     "primitive": [], "object": []},
        "validation": {"counter": 0, "ann": [], "task": [], "indx": [],
                       "primitive": [], "object": []},
    }

    def _emit_segment(sp: Path, split: str, primitive: str, object_: str,
                      frame_range: Optional[tuple[int, int]],
                      tag: str) -> None:
        ann_id = split_state[split]["counter"]
        out_dat = args.save_path / split / f"{args.scene_tag}+0" / f"ann_{ann_id}.dat"
        print(f"[{split}] ann={ann_id} {sp.stem} [{tag}] range={frame_range} → {out_dat}")
        n_kept = process_episode(
            state_path=sp,
            out_dat_path=out_dat,
            annotation_id=ann_id,
            hand_extrinsics_json=args.hand_extrinsics,
            tp_extrinsics_json=args.third_person_extrinsics,
            state_stride=args.state_stride,
            auto_state_stride=args.auto_state_stride,
            frame_range=frame_range,
        )
        label = f"{primitive} {object_}"
        s = split_state[split]
        s["ann"].append(label)
        s["task"].append(label)
        s["indx"].append((0, n_kept - 1))   # within-.dat keyframe range
        s["primitive"].append(primitive)
        s["object"].append(object_)
        s["counter"] += 1

    for sp in state_files:
        name = sp.stem
        if not auto_segment and name not in ann_table:
            continue

        # Pick split. In auto mode we honour an optional per-episode "split"
        # entry if present, otherwise val_fraction over the sorted list.
        entry = ann_table.get(name, {})
        split = split_for_episode(name, entry, args.val_fraction, all_names)

        if auto_segment:
            # Read the gripper trajectory once to find transitions.
            with h5py.File(sp, "r") as fs:
                gripper_full = fs["gripper_open"][:]
            segs = detect_gripper_segments(gripper_full,
                                           post_open_frames=args.post_open_frames)
            if not segs:
                print(f"  SKIP {name}: no gripper close detected")
                continue
            obj = args.auto_segment_object
            segment_labels = ["grasp", "place"]
            for (start_idx, end_idx), prim in zip(segs, segment_labels):
                # Guard against segments too short for keypose detection.
                if end_idx - start_idx + 1 < 4:
                    print(f"  SKIP {name} [{prim}]: segment too short "
                          f"({end_idx - start_idx + 1} frames)")
                    continue
                _emit_segment(sp, split, prim, obj,
                              frame_range=(start_idx, end_idx), tag=prim)
        else:
            _emit_segment(sp, split, entry["primitive"], entry["object"],
                          frame_range=None, tag="full")

    # Write per-split primitive_object_lang_ann.npy alongside the .dat files.
    for split, s in split_state.items():
        if s["counter"] == 0:
            continue
        emb = np.zeros((s["counter"], 1, 384), dtype=np.float32)
        out_npy_dir = args.save_path / split / "lang_annotations"
        out_npy_dir.mkdir(parents=True, exist_ok=True)
        out_npy = out_npy_dir / "primitive_object_lang_ann.npy"
        payload = {
            "language": {
                "ann": s["ann"],
                "task": s["task"],
                "emb": emb,
            },
            "info": {
                "indx": s["indx"],
                "episodes": [],
                "parent_task": list(s["task"]),
                "primitive": s["primitive"],
                "object": s["object"],
            },
        }
        np.save(out_npy, payload, allow_pickle=True)
        print(f"Wrote {s['counter']} annotations to {out_npy}")

        # Helpful summary for gripper_loc_bounds tuning later.
        from collections import Counter
        print(f"  primitives: {dict(Counter(s['primitive']))}")
        print(f"  objects:    {dict(Counter(s['object']))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
