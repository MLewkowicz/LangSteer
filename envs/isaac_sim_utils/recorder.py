"""HDF5 recorder for Isaac Sim teleoperation demos.

Records synchronized robot state + RGB images + per-pixel PCD images per
timestep.  Designed for low overhead in a ~30 Hz teleop loop: buffers are
lists of numpy arrays accumulated in memory; the HDF5 file is written only
when stop() is called.

HDF5 structure per episode::

    episode_YYYYMMDD_HHMMSS.h5
      rgb_static    [T, 200, 200, 3]  uint8
      rgb_gripper   [T, 200, 200, 3]  uint8
      pcd_static    [T, 200, 200, 3]  float32   world-space XYZ per pixel
      pcd_gripper   [T, 200, 200, 3]  float32
      robot_obs     [T, 15]           float32   CALVIN 15-dim convention
      ee_pose       [T, 7]            float32   [pos(3), euler_xyz(3), gripper_width(1)]
      timestamps    [T]               float64   seconds since episode start
    attrs:
      instruction   str
      task_name     str
      object        str   target object name (e.g. "mug") for primitive+object policy
      start_time    str   (YYYYMMDD_HHMMSS)
      num_steps     int
      duration_s    float
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np


# ANSI colors — recorder messages must be visible through Isaac Sim's
# console chatter, so we wrap them in a bright banner that pops visually.
_USE_COLOR = sys.stdout.isatty()
_GREEN = "\033[1;32m" if _USE_COLOR else ""
_YELLOW = "\033[1;33m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""
_BAR = "=" * 64


def _banner(color: str, lines: list) -> None:
    out = [f"\n{color}{_BAR}"]
    for line in lines:
        out.append(f"{color}>>> {line}")
    out.append(f"{color}{_BAR}{_RESET}\n")
    print("\n".join(out), flush=True)


class IsaacSimRecorder:

    def __init__(
        self,
        save_dir: str = "./data/isaac_sim_demos",
        capacity: int = 3000,
        episode_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            save_dir: Directory for episode HDF5 files.
            capacity: Initial buffer capacity in timesteps.
                      Buffers grow automatically; 3000 ≈ 100s at 30 Hz.
            episode_name: Optional base name for saved episodes. If set, the
                first recording saves as ``{episode_name}.h5`` and subsequent
                ones get a numeric suffix (``{episode_name}_2.h5`` …) to
                avoid clobbering. If None, falls back to timestamp-based
                ``episode_YYYYMMDD_HHMMSS.h5`` naming.
        """
        self._save_dir = Path(save_dir)
        self._capacity = capacity
        self._episode_name = episode_name
        self._recording = False
        self._count = 0
        self._start_time = 0.0
        self._start_wall = ""
        self._instruction = ""
        self._task_name = ""
        self._object = ""
        self._alloc_buffers()

    def _alloc_buffers(self) -> None:
        n = self._capacity
        self._buffers: Dict[str, List[np.ndarray]] = {
            "rgb_static": [],
            "rgb_gripper": [],
            "pcd_static": [],
            "pcd_gripper": [],
            "robot_obs": [],
            "ee_pose": [],
            "timestamps": [],
        }

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def num_steps(self) -> int:
        return self._count

    def toggle(self, instruction: str = "", task_name: str = "", object: str = "") -> None:
        if self._recording:
            self.stop()
        else:
            self.start(instruction=instruction, task_name=task_name, object=object)

    def start(self, instruction: str = "", task_name: str = "", object: str = "") -> None:
        if self._recording:
            return
        for key in self._buffers:
            self._buffers[key] = []
        self._count = 0
        self._start_time = time.monotonic()
        self._start_wall = time.strftime("%Y%m%d_%H%M%S")
        self._instruction = instruction
        self._task_name = task_name
        self._object = object
        self._recording = True
        msg = [f"RECORDING STARTED"]
        if instruction:
            msg.append(f"instruction: {instruction}")
        if task_name:
            msg.append(f"task: {task_name}")
        if object:
            msg.append(f"object: {object}")
        _banner(_GREEN, msg)

    def stop(self) -> Optional[str]:
        """Stop recording and save episode to HDF5.

        Returns:
            Path to saved file, or None if no data was recorded.
        """
        if not self._recording:
            return None
        self._recording = False

        if self._count == 0:
            _banner(_YELLOW, ["NO DATA RECORDED — nothing saved"])
            return None

        path = self._save_episode()
        _banner(
            _GREEN,
            [
                f"SAVED {self._count} steps",
                f"→ {path}",
            ],
        )
        return str(path)

    def step(
        self,
        rgb_static: np.ndarray,
        rgb_gripper: np.ndarray,
        pcd_static: np.ndarray,
        pcd_gripper: np.ndarray,
        robot_obs: np.ndarray,
        ee_pose: np.ndarray,
    ) -> None:
        """Record one timestep.

        Args:
            rgb_static: (200, 200, 3) uint8
            rgb_gripper: (200, 200, 3) uint8
            pcd_static: (200, 200, 3) float32 world-space XYZ
            pcd_gripper: (200, 200, 3) float32 world-space XYZ
            robot_obs: (15,) float32 CALVIN robot observation
            ee_pose: (7,) float32 [pos(3), euler_xyz(3), gripper_width(1)]
        """
        if not self._recording:
            return

        self._buffers["rgb_static"].append(rgb_static.astype(np.uint8))
        self._buffers["rgb_gripper"].append(rgb_gripper.astype(np.uint8))
        self._buffers["pcd_static"].append(pcd_static.astype(np.float32))
        self._buffers["pcd_gripper"].append(pcd_gripper.astype(np.float32))
        self._buffers["robot_obs"].append(robot_obs.astype(np.float32))
        self._buffers["ee_pose"].append(ee_pose.astype(np.float32))
        self._buffers["timestamps"].append(
            np.float64(time.monotonic() - self._start_time)
        )
        self._count += 1

    def _save_episode(self) -> Path:
        self._save_dir.mkdir(parents=True, exist_ok=True)
        fname = self._resolve_filename()
        n = self._count

        with h5py.File(fname, "w") as f:
            for key, frames in self._buffers.items():
                arr = np.stack(frames[:n])
                f.create_dataset(key, data=arr, compression="gzip", compression_opts=1)

            f.attrs["instruction"] = self._instruction
            f.attrs["task_name"] = self._task_name
            f.attrs["object"] = self._object
            f.attrs["start_time"] = self._start_wall
            f.attrs["num_steps"] = n
            f.attrs["duration_s"] = float(self._buffers["timestamps"][n - 1])

        return fname

    def _resolve_filename(self) -> Path:
        """Pick an output path. If `episode_name` is set, use it (with a
        numeric suffix if a file with that name already exists). Otherwise
        fall back to the timestamp-based default.
        """
        if self._episode_name is None:
            return self._save_dir / f"episode_{self._start_wall}.h5"

        base = self._save_dir / f"{self._episode_name}.h5"
        if not base.exists():
            return base
        i = 2
        while True:
            candidate = self._save_dir / f"{self._episode_name}_{i}.h5"
            if not candidate.exists():
                return candidate
            i += 1

    def close(self) -> None:
        if self._recording:
            self.stop()
