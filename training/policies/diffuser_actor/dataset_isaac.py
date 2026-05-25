"""Dataset for 3D Diffuser Actor training on Isaac Sim HDF5 demonstrations.

Loads episodes recorded by envs/isaac_sim_utils/recorder.py and produces
samples whose dict keys and shapes match CalvinDataset, so the existing
diffuser_actor trainer / collate function works unchanged.

The primitive+object policy ignores the language `instr` tensor — it derives
its conditioning from `primitive_id` + `object_id` instead — so this dataset
emits a zero `instr` tensor and does *not* load CLIP.

HDF5 episode schema (from IsaacSimRecorder):
  rgb_static  [T, 200, 200, 3] uint8
  rgb_gripper [T, 200, 200, 3] uint8
  pcd_static  [T, 200, 200, 3] float32
  pcd_gripper [T, 200, 200, 3] float32
  robot_obs   [T, 15]          float32   CALVIN 15-dim convention
  ee_pose     [T, 7]           float32   [pos(3), euler_xyz(3), gripper_width(1)]
  attrs: instruction, task_name, object

Per-frame `primitive_id` is auto-spliced from gripper_width: every frame before
the gripper first closes is `grasp`, every frame after is `place`.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from training.policies.diffuser_actor.preprocessing.calvin_utils import (
    convert_rotation,
)

logger = logging.getLogger(__name__)


_CROP_OFFSET = 20   # Same 20-pixel crop as CalvinDataset (200→160)
_CROP_SIZE = 160


# Primitive vocabulary for Isaac Sim demos.  Smaller than CALVIN's because
# Isaac teleop currently only collects pick-and-place demos and the policy
# is trained with num_primitives=2.  Order is significant — index = id.
ISAAC_PRIMITIVE_VOCAB: Dict[str, int] = {
    "grasp": 0,
    "place": 1,
}

# Object vocabulary.  Append-only and alphabetical so adding objects doesn't
# shift existing ids and invalidate trained checkpoints.
ISAAC_OBJECT_VOCAB: Dict[str, int] = {
    "bowl": 0,
    "mug": 1,
}


def _crop(arr: np.ndarray) -> np.ndarray:
    """Crop 200×200 → 160×160 at the CALVIN-matching offset."""
    return arr[_CROP_OFFSET:_CROP_OFFSET + _CROP_SIZE,
               _CROP_OFFSET:_CROP_OFFSET + _CROP_SIZE]


def _euler_to_quat_concat(ee: np.ndarray) -> np.ndarray:
    """Convert a [..., 7] array of [pos, euler_xyz, grip] to [..., 8] of
    [pos, quat_wxyz, grip] using the same conversion as CalvinDataset.
    """
    pos = ee[..., :3]
    euler = ee[..., 3:6]
    quat = convert_rotation(euler.reshape(-1, 3)).reshape(*euler.shape[:-1], 4)
    grip = ee[..., 6:]
    return np.concatenate([pos, quat, grip], axis=-1)


class IsaacDataset(Dataset):
    """Dataset of Isaac Sim HDF5 demonstrations for Diffuser Actor training.

    Each __getitem__ returns a chunk of `chunk_size` consecutive frames from
    a single episode, in the same key/shape layout CalvinDataset uses — so
    `traj_collate_fn` and the rest of the trainer work without changes.
    """

    def __init__(
        self,
        data_dir: str,
        nhist: int = 3,
        chunk_size: int = 5,
        execute_every: int = 4,
        traj_len: int = 20,
        gripper_close_threshold: float = 0.04,
        return_low_lvl_trajectory: bool = True,
        training: bool = True,
        max_episodes: int = -1,
        relative_action: bool = True,
    ) -> None:
        # When `relative_action=True`, the per-chunk current ee_pose is
        # subtracted from action / trajectory position+euler before quaternion
        # conversion — matching CalvinDataset's `to_relative_action`. This is
        # required to keep the frame consistent with the model's `convert2rel`
        # (which shifts pcd_obs and curr_gripper by curr_gripper); without it,
        # gt_trajectory stays absolute while pcd/gripper are gripper-relative,
        # and `normalize_pos` clamps either the trajectory or the PCD depending
        # on how `gripper_loc_bounds` is sized. Gripper width is never shifted.
        self._nhist = nhist
        self._chunk_size = chunk_size
        self._execute_every = execute_every
        self._traj_len = traj_len
        self._gripper_close_threshold = gripper_close_threshold
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        self._training = training
        self._relative_action = relative_action

        data_path = Path(data_dir)
        episode_files = sorted(data_path.glob("episode_*.h5"))
        if max_episodes > 0:
            episode_files = episode_files[:max_episodes]
        if not episode_files:
            raise FileNotFoundError(f"No episode_*.h5 files found in {data_dir}")

        # Pre-scan: keep only episodes long enough for one chunk + lookahead.
        min_T = max(nhist, 2) + max(execute_every, traj_len)
        self._episodes: List[Dict] = []
        for ep_path in episode_files:
            with h5py.File(ep_path, "r") as f:
                T = int(f["ee_pose"].shape[0])
                if T < min_T:
                    logger.warning(
                        f"Skipping {ep_path.name} (T={T} < min {min_T})"
                    )
                    continue
                self._episodes.append({
                    "path": ep_path,
                    "T": T,
                    "instruction": str(f.attrs.get("instruction", "")),
                    "task_name": str(f.attrs.get("task_name", "")),
                    "object": str(f.attrs.get("object", "")),
                })

        if not self._episodes:
            raise ValueError(f"No usable episodes found in {data_dir}")

        logger.info(
            f"IsaacDataset: {len(self._episodes)} episodes from {data_dir} "
            f"(chunk_size={chunk_size}, execute_every={execute_every}, "
            f"traj_len={traj_len}, return_low_lvl_trajectory={return_low_lvl_trajectory})"
        )

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self._episodes[idx]
        T = meta["T"]

        with h5py.File(meta["path"], "r") as f:
            ee_pose = f["ee_pose"][:].astype(np.float32)        # (T, 7)
            rgb_static_all = f["rgb_static"]                    # h5 lazy view
            rgb_gripper_all = f["rgb_gripper"]
            pcd_static_all = f["pcd_static"]
            pcd_gripper_all = f["pcd_gripper"]

            # Pick the chunk's start frame.  Need at least `nhist - 1` frames
            # of history before `t_start`, and `max(execute_every, traj_len)`
            # frames of lookahead after `t_start + chunk_size - 1`.
            chunk_size = min(self._chunk_size, T)
            lookahead = max(self._execute_every, self._traj_len)
            earliest = max(0, self._nhist - 1)
            latest = T - chunk_size - lookahead
            latest = max(latest, earliest)
            t_start = random.randint(earliest, latest) if self._training else earliest
            frame_ids = list(range(t_start, t_start + chunk_size))

            # Stack RGB / PCD over the chunk.  HDF5 supports a sorted-list
            # index, so this is one read per camera-stream per chunk.
            rgb_s = np.stack([_crop(rgb_static_all[t]) for t in frame_ids])     # (N,H,W,3)
            rgb_g = np.stack([_crop(rgb_gripper_all[t]) for t in frame_ids])
            pcd_s = np.stack([_crop(pcd_static_all[t]) for t in frame_ids])
            pcd_g = np.stack([_crop(pcd_gripper_all[t]) for t in frame_ids])

        # ------------------------------------------------------------------
        # Images: (N, ncam=2, 3, H, W) float in [0, 1]
        # ------------------------------------------------------------------
        rgbs = np.stack([rgb_s, rgb_g], axis=1)                                  # (N,2,H,W,3)
        rgbs = rgbs.transpose(0, 1, 4, 2, 3).astype(np.float32) / 255.0
        rgbs = torch.from_numpy(rgbs)

        pcds = np.stack([pcd_s, pcd_g], axis=1)                                  # (N,2,H,W,3)
        pcds = pcds.transpose(0, 1, 4, 2, 3).astype(np.float32)
        pcds = torch.from_numpy(pcds)

        # ------------------------------------------------------------------
        # Gripper proprio + history, in CALVIN [pos, quat_wxyz, grip] layout.
        # ------------------------------------------------------------------
        gripper_euler = ee_pose[frame_ids]                                       # (N,7)
        history_euler = np.stack(
            [
                np.stack(
                    [ee_pose[max(0, t - i)] for i in range(self._nhist - 1, -1, -1)],
                    axis=0,
                )
                for t in frame_ids
            ],
            axis=0,
        )                                                                        # (N,nhist,7)
        gripper = torch.as_tensor(_euler_to_quat_concat(gripper_euler), dtype=torch.float32)
        gripper_history = torch.as_tensor(
            _euler_to_quat_concat(history_euler), dtype=torch.float32
        )

        # ------------------------------------------------------------------
        # Action: ee_pose at t + execute_every — a continuous-mode keypose
        # proxy. Subtract curr ee_pose (gripper_euler) per chunk frame so the
        # action target lives in the same gripper-relative frame as pcd/
        # curr_gripper after the model's convert2rel; gripper width is left
        # absolute. Matches CalvinDataset's `to_relative_action`.
        # ------------------------------------------------------------------
        action_euler = np.stack(
            [ee_pose[min(t + self._execute_every, T - 1)] for t in frame_ids],
            axis=0,
        )                                                                        # (N,7)
        if self._relative_action:
            action_euler[..., :3] -= gripper_euler[..., :3]
            action_euler[..., 3:6] -= gripper_euler[..., 3:6]
        action = torch.as_tensor(_euler_to_quat_concat(action_euler), dtype=torch.float32)

        # ------------------------------------------------------------------
        # Primitive id (per frame): grasp until the gripper first closes,
        # place after.  All-grasp if the gripper never closed in this demo.
        # ------------------------------------------------------------------
        widths = ee_pose[:, 6]
        closed_frames = np.where(widths < self._gripper_close_threshold)[0]
        if closed_frames.size == 0:
            t_close = T  # whole episode is grasp
            logger.warning(
                f"{meta['path'].name}: gripper never closed "
                f"(min width={widths.min():.3f} ≥ {self._gripper_close_threshold}); "
                "labeling all frames as `grasp`."
            )
        else:
            t_close = int(closed_frames[0])
        primitive_ids = np.array(
            [
                ISAAC_PRIMITIVE_VOCAB["place"] if t >= t_close
                else ISAAC_PRIMITIVE_VOCAB["grasp"]
                for t in frame_ids
            ],
            dtype=np.int64,
        )
        primitive_id = torch.as_tensor(primitive_ids, dtype=torch.long).unsqueeze(-1)  # (N,1)

        # ------------------------------------------------------------------
        # Object id (one per episode, broadcast to N).
        # ------------------------------------------------------------------
        obj_str = meta["object"]
        oid = ISAAC_OBJECT_VOCAB.get(obj_str, -1)
        if oid < 0 and obj_str:
            logger.warning(
                f"{meta['path'].name}: object '{obj_str}' not in "
                f"ISAAC_OBJECT_VOCAB; emitting -1."
            )
        object_id = torch.full((chunk_size, 1), oid, dtype=torch.long)

        # ------------------------------------------------------------------
        # Instruction tensor — vestigial in primitive+object mode (the model
        # reads primitive_id/object_id embeddings instead).  Emit zeros so
        # the collate function's required-keys check passes.
        # ------------------------------------------------------------------
        instr = torch.zeros((chunk_size, 53, 512), dtype=torch.float32)

        ret_dict: Dict[str, torch.Tensor] = {
            "task": [meta["task_name"] for _ in frame_ids],
            "rgbs": rgbs,
            "pcds": pcds,
            "action": action,
            "instr": instr,
            "primitive_id": primitive_id,
            "object_id": object_id,
            "curr_gripper": gripper,
            "curr_gripper_history": gripper_history,
        }

        if self._return_low_lvl_trajectory:
            # Per-frame future trajectory window of length `traj_len`,
            # padded by repeating the final pose when the episode ends
            # before traj_len steps elapse.  trajectory_mask=True marks
            # padded (invalid) steps.
            traj_euler = np.zeros((chunk_size, self._traj_len, 7), dtype=np.float32)
            traj_lens = np.zeros(chunk_size, dtype=np.int64)
            for i, t in enumerate(frame_ids):
                remaining = T - 1 - t
                steps = min(self._traj_len, max(remaining, 0))
                traj_lens[i] = steps
                for k in range(self._traj_len):
                    src = min(t + 1 + k, T - 1)
                    traj_euler[i, k] = ee_pose[src]

            if self._relative_action:
                # Shift each waypoint by the current ee_pose (gripper_euler)
                # so the trajectory frame matches the pcd/curr_gripper frame
                # after the model's convert2rel. Position + euler only;
                # gripper width is preserved.
                traj_euler[..., :3] -= gripper_euler[:, None, :3]
                traj_euler[..., 3:6] -= gripper_euler[:, None, 3:6]

            traj = torch.as_tensor(_euler_to_quat_concat(traj_euler), dtype=torch.float32)

            traj_mask = torch.zeros((chunk_size, self._traj_len), dtype=torch.bool)
            for i, tl in enumerate(traj_lens.tolist()):
                if tl < self._traj_len:
                    traj_mask[i, tl:] = True

            ret_dict["trajectory"] = traj
            ret_dict["trajectory_mask"] = traj_mask

        return ret_dict
