"""Renderer Protocol adapter for `voxposer.visualizer.ValueMapVisualizer`.

Routes stage-activation snapshots to the existing Plotly HTML renderer. The
adapter is purely additive: `ValueMapVisualizer`'s internals and call contract
stay untouched (offline scripts like `scripts/test_voxposer.py` keep working).

Behavior contract: the adapter only emits HTML on stage-index change, not
every tick. `_activate_stage` in `steering/stage_manager.py` already calls
`ValueMapVisualizer.visualize(...)` directly today; iter 3 of Task 5 will
remove that direct call site so the Manager owns all dispatch.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StageHtmlRenderer:
    """Protocol adapter: routes stage-activation snapshots to ValueMapVisualizer."""

    def __init__(self, save_dir: Optional[str] = None, quality: str = "low") -> None:
        # Lazy import — `voxposer.visualizer` pulls in plotly which we don't want
        # at module-import time for headless smoke tests.
        from voxposer.visualizer import ValueMapVisualizer

        self._impl = ValueMapVisualizer(
            {
                "visualization_save_dir": save_dir,
                "visualization_quality": quality,
            }
        )
        self._state: dict[str, Any] = {}
        self._last_stage_idx: Optional[int] = None
        logger.info(
            f"StageHtmlRenderer initialized (save_dir={save_dir}, quality={quality})"
        )

    def update_state(self, state: dict[str, Any]) -> None:
        self._state = state

    def tick(self) -> None:
        s = self._state
        vm = s.get("value_map")
        if vm is None:
            return
        stage_idx = s.get("stage_idx")
        # Only emit on stage activation (stage_idx change), not every step.
        if stage_idx == self._last_stage_idx:
            return
        self._last_stage_idx = stage_idx
        self._impl.visualize(
            vm,
            ee_pos_world=s.get("ee_pos"),
            objects=s.get("objects"),
        )

    def close(self) -> None:
        # ValueMapVisualizer writes HTML files synchronously; nothing to release.
        pass
