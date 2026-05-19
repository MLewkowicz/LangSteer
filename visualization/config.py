"""Configuration dataclasses for the visualization system.

Iter 2 of Task 5: dropped `CameraVisualizationConfig`,
`TrajectoryVisualizationConfig`, `ReferenceVisualizationConfig`, and
`RolloutVisualizationConfig` alongside the master `render` / `cameras` /
`trajectory_3d` / `reference_plot` toggles. The surviving three blocks
drive the three artifacts the user-facing brief retained:
    - `html`           — stage HTML (via `StageHtmlRenderer`)
    - `live_costmap`   — live tk costmap window (via `LiveCostmapWindow`)
    - `video`          — MP4 video recording (via `CameraRenderer`)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HtmlConfig:
    """Configuration for the stage HTML renderer (`StageHtmlRenderer`).

    The renderer emits one Plotly HTML file per stage activation under
    `save_dir`. `null` save_dir uses the Hydra output directory at runtime.
    `quality` is passed through to `ValueMapVisualizer` (one of 'low',
    'medium', 'high').
    """
    enabled: bool = False
    save_dir: Optional[str] = None
    quality: str = "low"


@dataclass
class LiveCostmapConfig:
    """Configuration for the live tkinter costmap window.

    Drives `visualization/renderers/costmap_tk.py:LiveCostmapWindow`. The
    window mirrors `VoxPoserSteering._value_map` in real time alongside the
    PyBullet view, replacing the old browser-based Dash server.
    """
    enabled: bool = False
    refresh_interval: int = 1      # tick the window every N env steps
    downsample: int = 4            # voxel-grid stride for affordance/avoidance scatters
    point_threshold: float = 0.05  # min normalized intensity to render a voxel


@dataclass
class VideoConfig:
    """Configuration for MP4 video recording.

    Drives `visualization/renderers/camera_renderer.py:CameraRenderer`'s
    video path (`start_video` / `write_frame` / `stop_video`). Iter 3
    renames the class to `VideoRecorder` and strips the per-step PNG path.
    """
    enabled: bool = False
    save_path: str = "outputs/videos"
    fps: int = 30
    codec: str = "mp4v"
    # Optional high-res rendering for recording (does not affect policy inputs).
    # Set to 0 to use native camera resolution (200x200 static, 84x84 gripper).
    static_record_width: int = 0
    static_record_height: int = 0
    gripper_record_width: int = 0
    gripper_record_height: int = 0
    # Optional FOV override for static camera re-renders (degrees).
    # `None` = use CALVIN's native FOV.
    static_camera_fov: Optional[float] = None


@dataclass
class VisualizationConfig:
    """Master config for the three surviving visualization artifacts."""

    html: HtmlConfig = field(default_factory=HtmlConfig)
    live_costmap: LiveCostmapConfig = field(default_factory=LiveCostmapConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisualizationConfig":
        """Build from a nested dict (Hydra-friendly).

        Unknown keys are silently ignored so legacy YAML configs with
        `cameras: false` / `render: false` / etc. still load cleanly during
        the iter 1→3 transition. Iter 4 wires this into `run_evaluation.py`.
        """
        return cls(
            html=HtmlConfig(**config_dict.get("html", {})),
            live_costmap=LiveCostmapConfig(**config_dict.get("live_costmap", {})),
            video=VideoConfig(**config_dict.get("video", {})),
        )

    def is_any_enabled(self) -> bool:
        """True iff at least one renderer is enabled."""
        return self.html.enabled or self.live_costmap.enabled or self.video.enabled
