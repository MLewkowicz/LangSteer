"""MP4 video recorder for CALVIN camera streams.

Iter 2 of Task 5: stripped the per-step PNG / matplotlib display paths.
The class now only handles MP4 recording for the static + gripper cameras.
Iter 3 will rename `CameraRenderer` → `VideoRecorder` to match the
post-strip responsibility.

Lifecycle (Renderer Protocol):
    on_episode_start(eid, video_cfg)  → open writers lazily on first frame
    on_waypoint(frames)               → write one frame per camera
    on_episode_end() / close()        → flush + release writers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class CameraRenderer:
    """Records CALVIN camera streams to MP4 (static overhead + gripper)."""

    def __init__(self, config: Optional[Any] = None) -> None:
        """Args:
            config: `VideoConfig` (or None). Currently unused at construction;
                video parameters are passed on `on_episode_start(video_cfg=...)`.
                Stashed so future renderer state can read it without changing
                the Protocol.
        """
        self.config = config

        # Video state (writers are opened lazily on first frame so their
        # dimensions match the actual data — avoids resize mismatches when
        # the env upsamples gripper frames to match static resolution).
        self._video_writers: Dict[str, Any] = {}
        self._video_writer_sizes: Dict[str, tuple] = {}
        self._video_save_path: Optional[Path] = None
        self._video_fps: int = 15
        self._video_codec: str = "mp4v"
        self._video_episode_id: int = 0
        self._static_record_size: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Renderer Protocol
    # ------------------------------------------------------------------

    def update_state(self, state: dict) -> None:
        """No-op — video reacts to lifecycle hooks, not per-step state."""
        return None

    def tick(self) -> None:
        """No-op — frames are written via `on_waypoint`, not `tick`."""
        return None

    def close(self) -> None:
        """Flush + release any open video writers."""
        self._stop_video()

    def on_episode_start(self, episode_id: int, *, video_cfg: Any = None) -> None:
        """Open video writers for a new episode (lazy on first frame).

        `video_cfg` is the resolved `VideoConfig` (Manager passes it on
        dispatch). If `video_cfg.enabled` is False, this is a no-op.
        """
        if video_cfg is None or not video_cfg.enabled:
            return
        self._start_video(
            episode_id=episode_id,
            save_path=video_cfg.save_path,
            fps=video_cfg.fps,
            codec=video_cfg.codec,
            static_record_width=video_cfg.static_record_width,
            static_record_height=video_cfg.static_record_height,
        )

    def on_episode_end(self) -> None:
        """Flush writers at episode end."""
        self._stop_video()

    def on_waypoint(self, frames: Dict[str, np.ndarray]) -> None:
        """Write one frame per camera (sub-step granularity)."""
        self._write_frame(frames)

    # ------------------------------------------------------------------
    # Video pipeline
    # ------------------------------------------------------------------

    def _start_video(
        self,
        episode_id: int,
        save_path: str,
        fps: int = 15,
        codec: str = "mp4v",
        static_record_width: int = 0,
        static_record_height: int = 0,
    ) -> None:
        """Prepare video recording for an episode (writers open on first frame)."""
        if not _CV2_AVAILABLE:
            logger.error(
                "cv2 not available — video recording disabled. Install opencv-python."
            )
            return

        self._stop_video()  # release any open writers first
        self._video_save_path = Path(save_path)
        self._video_save_path.mkdir(parents=True, exist_ok=True)
        self._video_fps = fps
        self._video_codec = codec
        self._video_episode_id = episode_id
        if static_record_width > 0 and static_record_height > 0:
            self._static_record_size = (static_record_width, static_record_height)
        else:
            self._static_record_size = None

    def _open_writer(self, cam_key: str, frame: np.ndarray) -> Optional[Any]:
        """Open a cv2.VideoWriter for `cam_key`, sized to match the frame."""
        if not _CV2_AVAILABLE or self._video_save_path is None:
            return None

        h, w = frame.shape[:2]
        if cam_key == "static" and self._static_record_size is not None:
            w, h = self._static_record_size

        filename = f"episode_{self._video_episode_id:04d}_{cam_key}.mp4"
        out_path = self._video_save_path / filename
        fourcc = cv2.VideoWriter_fourcc(*self._video_codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, self._video_fps, (w, h))
        if not writer.isOpened():
            logger.warning(f"Failed to open video writer for {cam_key} at {out_path}")
            return None
        logger.info(f"Recording {cam_key} at {w}×{h} → {out_path}")
        self._video_writer_sizes[cam_key] = (w, h)
        return writer

    def _write_frame(self, obs_rgb: Dict[str, np.ndarray]) -> None:
        """Write one frame per camera to the open video writers."""
        if self._video_save_path is None:
            return

        for cam_key, frame in obs_rgb.items():
            if frame is None:
                continue
            frame = self._normalize_image(frame)

            # Open writer lazily on first frame so dimensions come from real data.
            if cam_key not in self._video_writers:
                writer = self._open_writer(cam_key, frame)
                if writer is None:
                    continue
                self._video_writers[cam_key] = writer

            writer = self._video_writers[cam_key]
            writer_w, writer_h = self._video_writer_sizes[cam_key]
            if (frame.shape[1], frame.shape[0]) != (writer_w, writer_h):
                frame = cv2.resize(
                    frame, (writer_w, writer_h), interpolation=cv2.INTER_LANCZOS4
                )

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    def _stop_video(self) -> None:
        """Release all open video writers."""
        for cam_key, writer in self._video_writers.items():
            writer.release()
            logger.info(f"Finished recording {cam_key}")
        self._video_writers.clear()
        self._video_writer_sizes.clear()

    @staticmethod
    def _normalize_image(img: np.ndarray) -> np.ndarray:
        """Normalize to uint8 [0, 255]."""
        if img.dtype != np.uint8:
            return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return img

    # ------------------------------------------------------------------
    # Legacy aliases — preserved through iter 2 for `run_experiment.py`'s
    # `viz_manager.start_recording/record_step/stop_recording` deprecated
    # paths. Iter 3 removes both these aliases and the Manager methods that
    # call them.
    # ------------------------------------------------------------------

    def start_video(self, **kwargs: Any) -> None:
        self._start_video(**kwargs)

    def write_frame(self, obs_rgb: Dict[str, np.ndarray]) -> None:
        self._write_frame(obs_rgb)

    def stop_video(self) -> None:
        self._stop_video()
