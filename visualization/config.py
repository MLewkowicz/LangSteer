"""Configuration dataclasses for visualization system."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class CameraVisualizationConfig:
    """Configuration for camera view visualization."""
    save_dir: str = "outputs/cameras"
    display_mode: str = "side_by_side"  # "side_by_side" or "overlay"
    save_images: bool = True
    show_live: bool = False


@dataclass
class TrajectoryVisualizationConfig:
    """Configuration for 3D trajectory visualization (multi-rollout analysis)."""
    num_rollouts: int = 10
    max_steps: int = 50
    save_dir: str = "outputs/trajectory_viz"
    modes: List[str] = field(default_factory=lambda: ["scatter", "lines"])
    plot_options: Dict = field(default_factory=lambda: {
        "point_size": 3,
        "line_width": 2,
        "opacity": 0.7,
        "colormap": "tab10"
    })
    snapshot_load_path: Optional[str] = None
    snapshot_save_path: Optional[str] = "outputs/snapshots/initial.npz"
    snapshot_auto_save: bool = True


@dataclass
class ReferenceVisualizationConfig:
    """Configuration for reference trajectory visualization (matplotlib plots)."""
    save_dir: str = "outputs/reference_viz"
    show_windows: bool = True
    plot_full_trajectory: bool = True
    plot_sliding_windows: bool = True
    window_timesteps: List[int] = field(default_factory=lambda: [0, 10, 20, 30])


@dataclass
class RolloutVisualizationConfig:
    """Configuration for PyBullet GUI rollout visualization."""
    frame_skip: int = 1
    playback_speed: float = 1.0
    keep_gui_open: bool = True


@dataclass
class VideoConfig:
    """Configuration for video recording."""
    enabled: bool = False
    save_path: str = "outputs/videos"
    fps: int = 30
    codec: str = "mp4v"


@dataclass
class VisualizationConfig:
    """
    Master configuration for all visualization modes.

    This replaces the need for separate visualization scripts by providing
    a unified config-driven interface for all visualization types.

    Attributes:
        render: Enable PyBullet GUI rendering (from rollout_reference.py)
        cameras: Enable camera feed visualization (from visualize_cameras.py)
        trajectory_3d: Enable multi-rollout 3D trajectory analysis (from visualize_trajectories.py)
        reference_plot: Enable reference trajectory matplotlib plots (from visualize_reference.py)
    """

    # Main visualization mode toggles
    render: bool = False
    cameras: bool = False
    trajectory_3d: bool = False
    reference_plot: bool = False

    # Sub-configurations for each mode
    camera: CameraVisualizationConfig = field(default_factory=CameraVisualizationConfig)
    trajectory: TrajectoryVisualizationConfig = field(default_factory=TrajectoryVisualizationConfig)
    reference: ReferenceVisualizationConfig = field(default_factory=ReferenceVisualizationConfig)
    rollout: RolloutVisualizationConfig = field(default_factory=RolloutVisualizationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'VisualizationConfig':
        """Create VisualizationConfig from nested dictionary (e.g., from Hydra)."""
        # Extract main toggles
        render = config_dict.get('render', False)
        cameras = config_dict.get('cameras', False)
        trajectory_3d = config_dict.get('trajectory_3d', False)
        reference_plot = config_dict.get('reference_plot', False)

        # Extract sub-configs
        camera_config = CameraVisualizationConfig(**config_dict.get('camera', {}))
        trajectory_config = TrajectoryVisualizationConfig(**config_dict.get('trajectory', {}))
        reference_config = ReferenceVisualizationConfig(**config_dict.get('reference', {}))
        rollout_config = RolloutVisualizationConfig(**config_dict.get('rollout', {}))
        video_config = VideoConfig(**config_dict.get('video', {}))

        return cls(
            render=render,
            cameras=cameras,
            trajectory_3d=trajectory_3d,
            reference_plot=reference_plot,
            camera=camera_config,
            trajectory=trajectory_config,
            reference=reference_config,
            rollout=rollout_config,
            video=video_config
        )

    def is_any_enabled(self) -> bool:
        """Check if any visualization mode is enabled."""
        return self.render or self.cameras or self.trajectory_3d or self.reference_plot
