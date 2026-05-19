"""Visualization Manager — Renderer Protocol dispatcher.

Iter 2 of Task 5: dropped PyBullet / Matplotlib / Plotly init branches and
the legacy `visualize_*` convenience wrappers — the corresponding renderers
no longer exist. The manager now dispatches exclusively over the three
surviving renderers (`StageHtmlRenderer`, `LiveCostmapWindow`,
`CameraRenderer`).

Deprecated aliases (kept through iter 2 for `run_experiment.py`; iter 3
removes them):
- `update_costmap(**kwargs)` → `update_state(state_dict)`
- `tick_costmap()`           → `tick()`
- `start_recording(eid)`     → `on_episode_start(eid)` (video only)
- `record_step(frames)`      → `on_waypoint(frames)` (video only)
- `stop_recording()`         → `on_episode_end()` (video only)
- `shutdown()`               → `close()`
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

        # Stage HTML renderer (Protocol-only, opt-in via `cfg.html.enabled`).
        if config.html.enabled:
            from .renderers import StageHtmlRenderer
            self._renderers.append(
                StageHtmlRenderer(
                    save_dir=config.html.save_dir,
                    quality=config.html.quality,
                )
            )
            logger.info("Initialized stage HTML renderer")

        # Live tk costmap window.
        self._live_costmap = None
        if config.live_costmap.enabled:
            from .renderers import LiveCostmapWindow
            self._live_costmap = LiveCostmapWindow(
                refresh_interval=config.live_costmap.refresh_interval,
                downsample=config.live_costmap.downsample,
                point_threshold=config.live_costmap.point_threshold,
            )
            self._renderers.append(self._live_costmap)
            logger.info("Initialized live tk costmap window")

        # MP4 video recorder (still `CameraRenderer`; iter 3 renames).
        self._camera_renderer = None
        if config.video.enabled:
            from .renderers import CameraRenderer
            self._camera_renderer = CameraRenderer(config.video)
            self._renderers.append(self._camera_renderer)
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
    # Deprecated aliases (iter 2 carry-over; iter 3 removes)
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
