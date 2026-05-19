"""Visualization Manager — Renderer Protocol dispatcher.

Iter 1 of Task 5: rewritten from toggle-keyed branching (272 LoC) to a thin
list-iteration dispatcher (~140 LoC including deprecated aliases). Each
enabled renderer is appended to `self._renderers`; the Manager fans
`update_state` / `tick` / `close` plus the optional `on_episode_start` /
`on_episode_end` / `on_waypoint` hooks out via `getattr(..., _noop)` so
strict 3-method Protocol implementers also work.

Deprecated aliases (kept through iter 2; removed in iter 3):
- `update_costmap(**kwargs)` → `update_state(state_dict)`
- `tick_costmap()`           → `tick()`
- `start_recording(eid)`     → `on_episode_start(eid)` (video only)
- `record_step(frames)`      → `on_waypoint(frames)` (video only)
- `stop_recording()`         → `on_episode_end()` (video only)
- `shutdown()`               → `close()`
- `reset()`                  → CameraRenderer.reset() (iter 2 strips)

The legacy `visualize_*` methods are kept verbatim for `run_experiment.py`'s
PyBullet / multi-rollout / reference-plot call sites; iter 2 strips them.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

from .base import Renderer
from .config import VisualizationConfig

logger = logging.getLogger(__name__)


def _noop(*args: Any, **kwargs: Any) -> None:
    """Default for missing optional Protocol hooks."""
    return None


class VisualizationManager:
    """Central coordinator dispatching the Renderer Protocol over enabled renderers."""

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config
        self._renderers: list[Renderer] = []

        # Camera renderer (handles video recording; also legacy per-step PNGs).
        # Iter 3 splits these — image-saving goes away, MP4 path stays.
        self._camera_renderer = None
        if config.cameras or config.video.enabled:
            from .renderers import CameraRenderer
            self._camera_renderer = CameraRenderer(config.camera)
            self._renderers.append(self._camera_renderer)
            logger.info("Initialized camera renderer (video + per-step PNGs)")

        # PyBullet renderer (legacy; iter 2 strips).
        self._pybullet_renderer = None
        if config.render:
            from .renderers import PyBulletRenderer
            self._pybullet_renderer = PyBulletRenderer(config.rollout)
            self._renderers.append(self._pybullet_renderer)
            logger.info("Initialized PyBullet renderer (legacy; iter 2 strips)")

        # Matplotlib reference-plot renderer (legacy; iter 2 strips).
        self._matplotlib_renderer = None
        if config.reference_plot:
            from .renderers import MatplotlibRenderer
            self._matplotlib_renderer = MatplotlibRenderer(config.reference)
            self._renderers.append(self._matplotlib_renderer)
            logger.info("Initialized matplotlib renderer (legacy; iter 2 strips)")

        # Plotly multi-rollout renderer (legacy; iter 2 strips).
        self._plotly_renderer = None
        if config.trajectory_3d:
            from .renderers import PlotlyRenderer
            self._plotly_renderer = PlotlyRenderer(config.trajectory)
            self._renderers.append(self._plotly_renderer)
            logger.info("Initialized Plotly renderer (legacy; iter 2 strips)")

        # Live tk costmap window — driven via Protocol dispatch.
        self._live_costmap = None
        if config.live_costmap:
            from .renderers import LiveCostmapWindow
            self._live_costmap = LiveCostmapWindow(
                refresh_interval=config.live_costmap_cfg.refresh_interval,
                downsample=config.live_costmap_cfg.downsample,
                point_threshold=config.live_costmap_cfg.point_threshold,
            )
            self._renderers.append(self._live_costmap)
            logger.info("Initialized live tk costmap window")

    # ------------------------------------------------------------------
    # Renderer Protocol dispatch
    # ------------------------------------------------------------------

    def update_state(self, state: dict[str, Any]) -> None:
        for r in self._renderers:
            getattr(r, "update_state", _noop)(state)

    def tick(self) -> None:
        for r in self._renderers:
            getattr(r, "tick", _noop)()

    def on_episode_start(self, episode_id: int) -> None:
        for r in self._renderers:
            # CameraRenderer needs the resolved VideoConfig; pass via kw.
            if r is self._camera_renderer:
                r.on_episode_start(episode_id, video_cfg=self.config.video)
            else:
                getattr(r, "on_episode_start", _noop)(episode_id)

    def on_episode_end(self) -> None:
        for r in self._renderers:
            getattr(r, "on_episode_end", _noop)()

    def on_waypoint(self, frames: dict[str, Any]) -> None:
        for r in self._renderers:
            getattr(r, "on_waypoint", _noop)(frames)

    def close(self) -> None:
        for r in self._renderers:
            getattr(r, "close", _noop)()

    def register(self, renderer: Renderer) -> None:
        """Programmatic extension point — append a custom renderer.

        Documented entry point for Task #7's `SceneImageRenderer` (VLM
        scene-image ingestion) and any future renderer not worth a
        config-level toggle.
        """
        self._renderers.append(renderer)

    def is_enabled(self) -> bool:
        """Check if any visualization mode is enabled."""
        return bool(self._renderers)

    # ------------------------------------------------------------------
    # Deprecated aliases (iter 1+2 carry-over; iter 3 removes)
    # ------------------------------------------------------------------

    def update_costmap(
        self,
        value_map: Any,
        ee_pos: Any,
        target: Any,
        objects: Optional[list],
        step: int,
        stage_idx: int,
        num_stages: int,
        instruction: str = "",
        primitive: Optional[str] = None,
        target_rotation: Any = None,
    ) -> None:
        """[DEPRECATED] Use `update_state(state)` with a snapshot dict."""
        warnings.warn(
            "VisualizationManager.update_costmap(**kwargs) is deprecated; "
            "use update_state(state_dict) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.update_state(
            {
                "value_map": value_map,
                "ee_pos": ee_pos,
                "target": target,
                "objects": objects,
                "step": step,
                "stage_idx": stage_idx,
                "num_stages": num_stages,
                "instruction": instruction,
                "primitive": primitive,
                "target_rotation": target_rotation,
            }
        )

    def tick_costmap(self) -> None:
        """[DEPRECATED] Use `tick()`."""
        warnings.warn(
            "VisualizationManager.tick_costmap() is deprecated; use tick().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.tick()

    def start_recording(self, episode_id: int) -> None:
        """[DEPRECATED] Use `on_episode_start(episode_id)`."""
        warnings.warn(
            "VisualizationManager.start_recording(eid) is deprecated; "
            "use on_episode_start(eid).",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._camera_renderer is not None and self.config.video.enabled:
            self._camera_renderer.on_episode_start(
                episode_id, video_cfg=self.config.video
            )

    def record_step(self, obs_rgb: dict) -> None:
        """[DEPRECATED] Use `on_waypoint(frames)`."""
        warnings.warn(
            "VisualizationManager.record_step(frames) is deprecated; "
            "use on_waypoint(frames).",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._camera_renderer is not None and self.config.video.enabled:
            self._camera_renderer.on_waypoint(obs_rgb)

    def stop_recording(self) -> None:
        """[DEPRECATED] Use `on_episode_end()`."""
        warnings.warn(
            "VisualizationManager.stop_recording() is deprecated; "
            "use on_episode_end().",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._camera_renderer is not None and self.config.video.enabled:
            self._camera_renderer.on_episode_end()

    def shutdown(self) -> None:
        """[DEPRECATED] Use `close()`."""
        warnings.warn(
            "VisualizationManager.shutdown() is deprecated; use close().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.close()

    def reset(self) -> None:
        """Reset CameraRenderer's per-episode step counter.

        Retained without deprecation warning — iter 2 strips the underlying
        step counter when image-saving is removed.
        """
        if self._camera_renderer is not None:
            self._camera_renderer.reset()

    # ------------------------------------------------------------------
    # Legacy convenience wrappers (kept through iter 2; iter 2 strips)
    # ------------------------------------------------------------------

    def visualize_episode(
        self,
        env,
        episode_result: Optional[Any] = None,
        actions: Optional[Any] = None,
        calvin_obs: Optional[dict] = None,
        step: Optional[int] = None,
    ) -> None:
        """Legacy entry point used by `run_experiment.py` for PyBullet + camera viz."""
        if self.config.cameras and calvin_obs is not None and self._camera_renderer is not None:
            self._camera_renderer.render_step(calvin_obs, step)
        if self.config.render and actions is not None and self._pybullet_renderer is not None:
            self._pybullet_renderer.render_episode(env, actions)

    def visualize_reference_trajectory(
        self, actions, task_name: str = "task", horizon: int = 16
    ) -> None:
        """Legacy matplotlib reference-trajectory plot. Iter 2 strips."""
        if self.config.reference_plot and self._matplotlib_renderer is not None:
            self._matplotlib_renderer.render_reference_trajectory(
                actions, task_name, horizon
            )

    def visualize_multi_rollout(self, env, policy, steering=None, snapshot=None):
        """Legacy Plotly multi-rollout 3D analysis. Iter 2 strips."""
        if self.config.trajectory_3d and self._plotly_renderer is not None:
            return self._plotly_renderer.render_multi_rollout_trajectories(
                env, policy, steering, snapshot
            )
        return None

    def visualize_step(self, env, action=None, calvin_obs=None, step=None) -> None:
        """Legacy per-step camera + PyBullet entry point. Iter 2 strips."""
        if self.config.cameras and calvin_obs is not None and self._camera_renderer is not None:
            self._camera_renderer.render_step(calvin_obs, step)
        if self.config.render and self._pybullet_renderer is not None:
            self._pybullet_renderer.render_step(env, action)

    def __repr__(self) -> str:
        enabled = []
        if self.config.render:
            enabled.append("render")
        if self.config.cameras:
            enabled.append("cameras")
        if self.config.trajectory_3d:
            enabled.append("trajectory_3d")
        if self.config.reference_plot:
            enabled.append("reference_plot")
        if self.config.video.enabled:
            enabled.append("video")
        if self.config.live_costmap:
            enabled.append("live_costmap")
        if not enabled:
            return "VisualizationManager(no modes enabled)"
        return f"VisualizationManager(modes={enabled})"
