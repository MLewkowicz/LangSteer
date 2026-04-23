"""Camera view renderer - displays RGB images from CALVIN cameras.

Extracted from scripts/visualize_cameras.py
"""

import numpy as np
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, Optional
import matplotlib.pyplot as plt

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class CameraRenderer:
    """Renders camera views from CALVIN environment (static overhead + gripper cameras)."""

    def __init__(self, config):
        """
        Initialize camera renderer.

        Args:
            config: CameraVisualizationConfig instance
        """
        self.config = config
        self.output_dir = Path(config.save_dir)
        self.step_counter = 0

        if config.save_images:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Video recording state
        self._video_writers: Dict[str, any] = {}
        self._video_writer_sizes: Dict[str, tuple] = {}  # cam_key → (w, h) for the open writer
        self._video_save_path: Optional[Path] = None
        self._video_fps: int = 15
        self._video_codec: str = "mp4v"
        self._video_episode_id: int = 0
        self._static_record_size: Optional[tuple] = None  # (w, h) override for static cam

    def render_step(self, calvin_obs: Dict, step: int = None) -> Dict[str, np.ndarray]:
        """
        Render camera views for a single step.

        Args:
            calvin_obs: Raw CALVIN observation dict with 'rgb_static' and 'rgb_gripper' keys
            step: Step number for filename (uses internal counter if None)

        Returns:
            Dictionary with camera names as keys and RGB images as values
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1

        images = {}

        # Extract and save static camera
        rgb_static = calvin_obs.get('rgb_static')
        if rgb_static is not None:
            rgb_static = self._normalize_image(rgb_static)
            images['static'] = rgb_static

            if self.config.save_images:
                img_static = Image.fromarray(rgb_static)
                static_path = self.output_dir / f"step_{step:04d}_static.png"
                img_static.save(static_path)
                logger.debug(f"Saved static camera: {static_path}")

        # Extract and save gripper camera
        rgb_gripper = calvin_obs.get('rgb_gripper')
        if rgb_gripper is not None:
            rgb_gripper = self._normalize_image(rgb_gripper)
            images['gripper'] = rgb_gripper

            if self.config.save_images:
                img_gripper = Image.fromarray(rgb_gripper)
                gripper_path = self.output_dir / f"step_{step:04d}_gripper.png"
                img_gripper.save(gripper_path)
                logger.debug(f"Saved gripper camera: {gripper_path}")

        # Display live if requested
        if self.config.show_live and images:
            self.display_cameras(images)

        return images

    def display_cameras(self, images: Dict[str, np.ndarray]):
        """Display camera views side-by-side using matplotlib."""
        n_cameras = len(images)
        if n_cameras == 0:
            return

        if self.config.display_mode == "side_by_side":
            fig, axes = plt.subplots(1, n_cameras, figsize=(6 * n_cameras, 6))
            if n_cameras == 1:
                axes = [axes]

            for ax, (name, img) in zip(axes, images.items()):
                ax.imshow(img)
                ax.set_title(f"{name.capitalize()} Camera", fontsize=14)
                ax.axis('off')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)  # Non-blocking update

        elif self.config.display_mode == "overlay":
            # Simple overlay mode - just show most recent frame
            for name, img in images.items():
                plt.clf()
                plt.imshow(img)
                plt.title(f"{name.capitalize()} Camera")
                plt.axis('off')
                plt.draw()
                plt.pause(0.001)

    @staticmethod
    def _normalize_image(img: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 [0, 255] range."""
        if img.dtype != np.uint8:
            return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return img

    def reset(self):
        """Reset step counter for new episode."""
        self.step_counter = 0

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_video(
        self,
        episode_id: int,
        save_path: str,
        fps: int = 15,
        codec: str = "mp4v",
        static_record_width: int = 0,
        static_record_height: int = 0,
    ):
        """
        Prepare video recording for an episode.

        Writers are opened lazily on the first write_frame() call so that
        frame dimensions are always derived from actual data (avoiding
        mismatches when the env resizes gripper images to match static res).

        Args:
            episode_id: Episode index used in the output filename.
            save_path: Directory where MP4 files will be written.
            fps: Frames per second for the output video.
            codec: FourCC codec string (e.g. 'mp4v', 'avc1').
            static_record_width: Override width for static camera recording (0 = use frame width).
            static_record_height: Override height for static camera recording (0 = use frame height).
        """
        if not _CV2_AVAILABLE:
            logger.error("cv2 not available — video recording disabled. Install opencv-python.")
            return

        self.stop_video()  # Release any open writers first

        self._video_save_path = Path(save_path)
        self._video_save_path.mkdir(parents=True, exist_ok=True)
        self._video_fps = fps
        self._video_codec = codec
        self._video_episode_id = episode_id
        # None means "use native frame size"; set only when overrides are both nonzero
        if static_record_width > 0 and static_record_height > 0:
            self._static_record_size = (static_record_width, static_record_height)
        else:
            self._static_record_size = None

    def _open_writer(self, cam_key: str, frame: np.ndarray) -> Optional[any]:
        """
        Open a cv2.VideoWriter for cam_key, sized to match the (possibly upscaled) frame.
        Returns the writer, or None on failure.
        """
        if not _CV2_AVAILABLE or self._video_save_path is None:
            return None

        h, w = frame.shape[:2]
        if cam_key == "static" and self._static_record_size is not None:
            w, h = self._static_record_size  # override dimensions for static cam

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

    def write_frame(self, obs_rgb: Dict[str, np.ndarray]):
        """
        Write one frame per camera to the video writers.

        Writers are opened on the first call using actual frame dimensions,
        so there is no risk of size mismatch regardless of env preprocessing.

        Args:
            obs_rgb: Dict mapping camera keys ('static', 'gripper') to RGB arrays
                     (uint8 or float32 [0,1]).  Matches Observation.rgb.
        """
        if self._video_save_path is None:
            return

        for cam_key, frame in obs_rgb.items():
            if frame is None:
                continue
            frame = self._normalize_image(frame)

            # Open writer lazily on first frame so dimensions come from actual data
            if cam_key not in self._video_writers:
                writer = self._open_writer(cam_key, frame)
                if writer is None:
                    continue
                self._video_writers[cam_key] = writer

            writer = self._video_writers[cam_key]

            # Resize to writer dimensions if needed (handles static upscale and
            # any source inconsistency between initial obs and per-waypoint raw obs)
            writer_w, writer_h = self._video_writer_sizes[cam_key]
            if (frame.shape[1], frame.shape[0]) != (writer_w, writer_h):
                frame = cv2.resize(frame, (writer_w, writer_h), interpolation=cv2.INTER_LANCZOS4)

            # cv2 expects BGR and uint8
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    def stop_video(self):
        """Release all open video writers."""
        for cam_key, writer in self._video_writers.items():
            writer.release()
            logger.info(f"Finished recording {cam_key}")
        self._video_writers.clear()
        self._video_writer_sizes.clear()
