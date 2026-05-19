"""Renderer Protocol for the LangSteer visualization suite.

A Renderer is an observation-only viz hook that receives state updates from
`VisualizationManager` and produces an artifact (HTML file, live window, MP4,
etc.). Lifecycle: created at construction, updated per step via
`update_state` + `tick`, torn down on `close`.

The Protocol is intentionally minimal — three core methods plus three optional
lifecycle hooks. This matches the "observation-only, no rich shared state"
property of every viz path in this repo. Concrete renderers may carry their
own state (open writers, mutable artists, save dirs) but the Manager never
reads it.

Optional hooks are documented as keyword-only; the Manager dispatches each
hook via `getattr(renderer, '<hook>', _noop)` so a minimal 3-method Protocol
implementer is fully Manager-compatible.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Renderer(Protocol):
    """Observation-only viz hook for the steering pipeline.

    Implementations should be idempotent on `update_state` / `tick` and safe
    to call after `close`. The Manager does not enforce ordering beyond
    `update_state` → `tick` → `close`.

    Core methods (required):
        update_state(state): stash the latest steering snapshot.
        tick():              refresh the artifact for the current state.
        close():             release resources.

    Optional lifecycle hooks (dispatched via getattr; safe to omit):
        on_episode_start(episode_id): per-episode setup (e.g. open video).
        on_episode_end():             per-episode teardown (e.g. flush video).
        on_waypoint(frames):          sub-step granularity (used by video).
    """

    def update_state(self, state: dict[str, Any]) -> None:
        """Stash the latest steering snapshot. Should be cheap.

        `state` is the dict returned by `VoxPoserSteering.get_costmap_state()`
        plus any keys the Manager merges in (e.g. `episode_id`,
        `target_rotation`). Concrete renderers pick the keys they care
        about.
        """
        ...

    def tick(self) -> None:
        """Produce or refresh the artifact for the current state.

        May be a no-op (e.g., the stage HTML renderer only acts on stage
        activations, not every step).
        """
        ...

    def close(self) -> None:
        """Release resources (file handles, tk window, video writers).

        Must be safe to call multiple times.
        """
        ...
