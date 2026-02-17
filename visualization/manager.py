"""Visualization Manager - orchestrates all visualization modes.

This is the main entry point for the unified visualization system, replacing
the need for separate visualization scripts.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .config import VisualizationConfig
from .renderers import (
    CameraRenderer,
    PyBulletRenderer,
    MatplotlibRenderer,
    PlotlyRenderer
)

logger = logging.getLogger(__name__)


class VisualizationManager:
    """
    Central coordinator for all visualization modes.

    This class replaces 4 separate scripts:
    - scripts/rollout_reference.py → render mode
    - scripts/visualize_cameras.py → cameras mode
    - scripts/visualize_trajectories.py → trajectory_3d mode
    - scripts/visualize_reference.py → reference_plot mode

    Usage:
        config = VisualizationConfig.from_dict(cfg.visualization)
        viz_manager = VisualizationManager(config)

        # During experiment:
        for episode in episodes:
            result = run_episode(...)
            viz_manager.visualize_episode(result, env)
    """

    def __init__(self, config: VisualizationConfig):
        """
        Initialize visualization manager.

        Args:
            config: VisualizationConfig instance
        """
        self.config = config

        # Initialize renderers based on enabled modes
        self.camera_renderer = None
        self.pybullet_renderer = None
        self.matplotlib_renderer = None
        self.plotly_renderer = None

        if config.cameras:
            logger.info("Initializing camera renderer")
            self.camera_renderer = CameraRenderer(config.camera)

        if config.render:
            logger.info("Initializing PyBullet renderer")
            self.pybullet_renderer = PyBulletRenderer(config.rollout)

        if config.reference_plot:
            logger.info("Initializing matplotlib renderer")
            self.matplotlib_renderer = MatplotlibRenderer(config.reference)

        if config.trajectory_3d:
            logger.info("Initializing Plotly renderer")
            self.plotly_renderer = PlotlyRenderer(config.trajectory)

    def visualize_episode(
        self,
        env,
        episode_result: Optional[Any] = None,
        actions: Optional[Any] = None,
        calvin_obs: Optional[Dict] = None,
        step: Optional[int] = None
    ):
        """
        Visualize a single episode or step.

        Args:
            env: Environment instance
            episode_result: EpisodeResult from episode_runner (for trajectory_3d mode)
            actions: Action sequence for PyBullet rendering
            calvin_obs: CALVIN observation dict for camera rendering
            step: Step number for camera rendering
        """
        # Camera rendering
        if self.config.cameras and calvin_obs is not None:
            self.camera_renderer.render_step(calvin_obs, step)

        # PyBullet rendering (for reference rollout)
        if self.config.render and actions is not None:
            self.pybullet_renderer.render_episode(env, actions)

    def visualize_reference_trajectory(
        self,
        actions,
        task_name: str = "task",
        horizon: int = 16
    ):
        """
        Visualize reference trajectory using matplotlib.

        Args:
            actions: Reference action sequence (T, 7)
            task_name: Task name for titles/filenames
            horizon: Prediction horizon for sliding windows
        """
        if self.config.reference_plot and self.matplotlib_renderer:
            self.matplotlib_renderer.render_reference_trajectory(
                actions, task_name, horizon
            )

    def visualize_multi_rollout(
        self,
        env,
        policy,
        steering=None,
        snapshot=None
    ):
        """
        Visualize multiple rollouts with 3D trajectory analysis.

        This mode runs N rollouts from the same initial state and generates
        interactive Plotly visualizations.

        Args:
            env: Environment instance
            policy: Policy instance
            steering: Optional steering module
            snapshot: Optional pre-captured environment snapshot

        Returns:
            Dictionary with statistics and saved file paths
        """
        if self.config.trajectory_3d and self.plotly_renderer:
            return self.plotly_renderer.render_multi_rollout_trajectories(
                env, policy, steering, snapshot
            )
        return None

    def visualize_step(
        self,
        env,
        action=None,
        calvin_obs=None,
        step=None
    ):
        """
        Visualize a single environment step.

        Args:
            env: Environment instance
            action: Optional action to execute
            calvin_obs: Optional CALVIN observation for camera rendering
            step: Step number
        """
        # Camera rendering
        if self.config.cameras and calvin_obs is not None:
            self.camera_renderer.render_step(calvin_obs, step)

        # PyBullet step rendering
        if self.config.render and self.pybullet_renderer:
            self.pybullet_renderer.render_step(env, action)

    def reset(self):
        """Reset all renderers for new episode."""
        if self.camera_renderer:
            self.camera_renderer.reset()

    def is_enabled(self) -> bool:
        """Check if any visualization mode is enabled."""
        return self.config.is_any_enabled()

    def __repr__(self):
        """String representation showing enabled modes."""
        enabled_modes = []
        if self.config.render:
            enabled_modes.append("render")
        if self.config.cameras:
            enabled_modes.append("cameras")
        if self.config.trajectory_3d:
            enabled_modes.append("trajectory_3d")
        if self.config.reference_plot:
            enabled_modes.append("reference_plot")

        if not enabled_modes:
            return "VisualizationManager(no modes enabled)"

        return f"VisualizationManager(modes={enabled_modes})"
