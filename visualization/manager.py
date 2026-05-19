"""Visualization Manager — Renderer Protocol dispatcher.

Iter 3 of Task 5: dropped the 5 deprecated aliases (`update_costmap` /
`tick_costmap` / `start_recording` / `stop_recording` / `record_step` /
`shutdown`). Callers now use the Protocol methods directly. `VideoRecorder`
takes its `VideoConfig` at construction, so `on_episode_start(episode_id)`
no longer needs a per-call config kwarg.

Dispatch shape:
    update_state(state) / tick() / close()    — required Protocol methods
    on_episode_start(eid) / on_episode_end()  — optional lifecycle hooks
    on_waypoint(frames)                       — optional sub-step hook
    register(renderer)                        — programmatic extension point
                                                (Task #7 SceneImageRenderer)
"""

from __future__ import annotations

import logging
from typing import Any

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

        # Stage HTML renderer (Plotly 3D, one HTML per stage activation).
        if config.html.enabled:
            from .renderers import StageHtmlRenderer
            self._renderers.append(
                StageHtmlRenderer(
                    save_dir=config.html.save_dir,
                    quality=config.html.quality,
                )
            )
            logger.info("Initialized stage HTML renderer")

        # Live tk costmap window (headless-safe; silently disables on
        # TclError so headless eval hosts don't crash).
        if config.live_costmap.enabled:
            from .renderers import LiveCostmapTkRenderer
            self._renderers.append(
                LiveCostmapTkRenderer(
                    refresh_interval=config.live_costmap.refresh_interval,
                    downsample=config.live_costmap.downsample,
                    point_threshold=config.live_costmap.point_threshold,
                )
            )
            logger.info("Initialized live tk costmap renderer")

        # MP4 video recorder.
        if config.video.enabled:
            from .renderers import VideoRecorder
            self._renderers.append(VideoRecorder(config.video))
            logger.info("Initialized video recorder")

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

    def __repr__(self) -> str:
        enabled = []
        if self.config.html.enabled:
            enabled.append("html")
        if self.config.live_costmap.enabled:
            enabled.append("live_costmap")
        if self.config.video.enabled:
            enabled.append("video")
        if not enabled:
            return "VisualizationManager(no modes enabled)"
        return f"VisualizationManager(modes={enabled})"
