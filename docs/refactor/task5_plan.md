# Task 5 — Visualization cleanup

**Owner:** planner → refactorer (after team-lead approval)
**Branch:** `refactoring`
**Task ID:** `#5` (status: in_progress — planning phase)
**Scope:** Strip dead visualization infrastructure, formalize a `Renderer` Protocol, keep the three live artifacts (stage HTML, live tk window, MP4 video), wire the live tk window into `run_evaluation.py` (currently dead in eval), and surface a clean extension point for Task #7 (VLM scene-image ingestion).

Smaller than 3b. Bounded to viz code; no edits to inference path, no behavioral changes to steering/policies. Expected work: 4 iters across one session. Implementation gated on Task #4 shipping (per team-lead) — plan can be reviewed/approved in parallel.

---

## 0. Task-status check (no split needed)

Task #5 in the task list covers all four sub-goals from the user brief in one bundle. The work is observation-only by definition and the four sub-goals are tightly coupled (the renderer Protocol shape determines the Manager dispatch shape determines what gets stripped). Splitting would force two reviews of effectively one design.

| Task ID | Subject | Status |
|---|---|---|
| #4 | Loop-back tuning + diagnose underperforming/regressed tasks | 🟡 in_progress |
| #5 | **Task 5 — Visualization cleanup** | 🟡 in_progress (this plan) |
| #7 | Task 7 — VLM scene-image ingestion for value-map construction | ⬜ pending — blocked by #4/#5. §9 leaves the extension point. |

**Note on Task #4 dependency:** team-lead deferred Task 5 implementation until #4 ships so the live tk window's diagnostic value can be measured against the post-#4 dwell behavior. Plan-drafting and team-lead review happen now; refactorer kickoff waits for #4 close.

---

## 1. Original brief excerpt (quoted verbatim for self-contained doc)

User kickoff (`refactoring_tasks.md`, viz paragraph):

> "We need to clean up the visualization utils. We want to retain the stage specific HTMLs that can be viewed and we want to have a 3D interactive render in a simple tkinter app that shows the HTML style value maps that can be rotated/inspected and the object + action primitive that the base model is conditioning on for a specific stage live (or any other lightweight visualization software that can be iterated on). All other visualization infrastructure should be eliminated but the visualization suite should still remain extensible for other types of rendering."

Four sub-goals decoded:
1. **Keep** stage-specific HTML rendering (per-stage value map snapshots).
2. **Add** a 3D interactive viewer in tkinter (or similar lightweight) showing live value-map + current stage's primitive + object.
3. **Strip** all other visualization infrastructure.
4. The suite should stay **extensible** for future render types.

**Critical finding from survey:** sub-goal #2 is *already implemented* as `visualization/renderers/costmap_tk.py::LiveCostmapWindow` (420L tkinter+matplotlib). The user's brief was written before this existed. Task 5's centerpiece is therefore **enable/extend/wire-into-eval**, not from-scratch build. Specifically:
- Already shows: rotatable 3D value map (drag/scroll), affordance/avoidance scatters, OBBs, gripper, target, PRIMITIVE label (color-coded by motion type), STAGE i/N, STEP, INSTRUCTION.
- Missing: OBJECT label (user explicitly asked for object + primitive).
- Missing: wiring into `run_evaluation.py` (only `run_experiment.py` instantiates `VisualizationManager` today).

---

## 2. Surveyed surface (KEEP / STRIP / NEW)

Two parallel viz packages exist today (`visualization/` and `utils/visualization/`) plus an in-package renderer (`voxposer/visualizer.py`). Full inventory:

| Path | LoC | Role | Verdict |
|------|-----|------|---------|
| `voxposer/visualizer.py::ValueMapVisualizer` | 332 | Plotly HTML stage renders (3D value-map + scene PCD + OBBs + EE + targets). Saves `HH_MM_SS.html` + `latest.html` per stage activation. | **KEEP** — adapted to Protocol via thin wrapper. Internals unchanged. |
| `visualization/renderers/costmap_tk.py::LiveCostmapWindow` | 420 | tkinter + matplotlib 3D live window (rotatable, drag/scroll), per-tick artist mutation, side panel with primitive/stage/step/instruction. | **KEEP + EXTEND** — add OBJECT label; adopt Protocol method names. |
| `visualization/renderers/camera_renderer.py::CameraRenderer` | 248 | (1) per-frame RGB image saves (PNG) + matplotlib display, (2) MP4 video writers (`start_video`/`write_frame`/`stop_video`). | **SLIM to ~80 LoC** — strip image-saving + matplotlib-display path (per Q1 (b)); keep MP4 video path; rename to `VideoRecorder` for clarity. |
| `visualization/manager.py::VisualizationManager` | 272 | Toggle-keyed dispatcher to 5 renderers (render/cameras/trajectory_3d/reference_plot/live_costmap + video). | **REWRITE as thin dispatcher** — iterate a list of `Renderer` instances; ~80 LoC. |
| `visualization/config.py` | 151 | 6 dataclasses (Camera/Trajectory/Reference/Rollout/Video/LiveCostmap) + master config. | **SLIM to 3** — keep LiveCostmap + Video + master toggles; drop the 3 dead ones. |
| `visualization/renderers/plotly_renderer.py` | 206 | Multi-rollout 3D trajectory analysis HTML (old `visualize_trajectories.py`). | **STRIP** |
| `visualization/renderers/matplotlib_renderer.py` | 124 | Reference trajectory matplotlib plots (old `visualize_reference.py`). | **STRIP** |
| `visualization/renderers/pybullet_renderer.py` | 98 | PyBullet GUI playback (old `rollout_reference.py`). | **STRIP** |
| `visualization/collectors/trajectory_collector.py` | 213 | Multi-rollout collector feeding plotly_renderer. | **STRIP** (delete `collectors/` package). |
| `visualization/collectors/__init__.py` | 8 | Reexports. | **STRIP** |
| `utils/visualization/trajectory_viz.py` | 397 | `TrajectoryVisualizer` plot helpers feeding plotly_renderer. | **STRIP** |
| `utils/visualization/plotly_viz.py` | 236 | Orphan — zero callers in prod code. | **STRIP** |
| `utils/visualization/__init__.py` | 1 | Empty. | **STRIP** (delete `utils/visualization/` dir). |
| `conf/visualization/base.yaml` | 68 | Hydra config with 6 mode toggles + sub-configs. | **REWRITE to ~25 lines** (live_costmap + video only). |
| `conf/visualization/multistage.yaml` | 58 | Video-recording profile for multi-stage task capture. | **TRIM** to ~20 lines (video block + headers only). |
| `tmp/denoise_visualizer/{capture,render,visualize}.py` | — | Standalone denoise-step exploration; depends on `tmp/bbox_annotator`. | **STRIP** — `tmp/` is not git-tracked (verified: `git ls-files tmp/` returns 0). Plain `rm -rf tmp/denoise_visualizer/`. |
| `scripts/test_voxposer.py`, `scripts/calibrate_voxposer_objects.py` | — | Offline calibration tools using `ValueMapVisualizer` standalone. | **KEEP as-is** (out of scope per Q4 decision). May need a one-line adjustment if `ValueMapVisualizer.__init__` signature changes for Protocol adoption; otherwise zero-touch. |

**Net delta:** roughly -1700 LoC of viz code (-3 dead renderers, -1 dead collector, -1 dead utils/viz dir, -1 orphan file, -1 stripped tmp dir, -manager rewrite, -config slim). +1 Protocol file (~40 LoC). +1 adapter for HTML (~30 LoC). +1 OBJECT-label patch (~10 LoC). +1 eval-wiring patch (~30 LoC). Net ≈ -1500 LoC.

---

## 3. Target file layout post-refactor

```
visualization/
├── __init__.py             # exports Renderer, VisualizationManager, VisualizationConfig
├── base.py                 # NEW — Renderer Protocol (~40 LoC)
├── config.py               # SLIM — LiveCostmap + Video + master toggles (~60 LoC)
├── manager.py              # REWRITE — thin dispatcher over List[Renderer] (~80 LoC)
└── renderers/
    ├── __init__.py         # SLIM — exports the three concrete renderers
    ├── stage_html.py       # NEW (~30 LoC) — thin Protocol adapter around voxposer/visualizer.py
    ├── live_costmap_tk.py  # RENAMED from costmap_tk.py + OBJECT label patch (~430 LoC)
    └── video_recorder.py   # RENAMED + SLIMMED from camera_renderer.py — MP4 only (~90 LoC)

voxposer/
└── visualizer.py           # UNCHANGED — internals stay; stage_html.py wraps it.

conf/visualization/
├── base.yaml               # REWRITE (~25 LoC) — live_costmap + video defaults only
└── multistage.yaml         # TRIM (~20 LoC) — video block only

scripts/
├── run_evaluation.py       # PATCH (~30 LoC added) — VisualizationManager wiring + step_callback tick hook
└── run_experiment.py       # PATCH (~15 LoC simplified) — same shape, just uses Protocol dispatch via Manager

# STRIPPED:
# visualization/collectors/          (entire package)
# visualization/renderers/plotly_renderer.py
# visualization/renderers/matplotlib_renderer.py
# visualization/renderers/pybullet_renderer.py
# utils/visualization/                (entire package, including plotly_viz.py + trajectory_viz.py)
# tmp/denoise_visualizer/             (untracked; plain rm -rf)
```

**Why `voxposer/visualizer.py` stays untouched:** `scripts/test_voxposer.py` and `scripts/calibrate_voxposer_objects.py` instantiate it directly (Q4 decision: keep as-is). Wrapping it in an adapter at `visualization/renderers/stage_html.py` lets the new Protocol dispatcher own all renderer wiring without breaking those scripts.

**Why rename `costmap_tk.py` → `live_costmap_tk.py`:** clearer intent. `costmap` was an old name; the rest of the codebase says "value map." Renaming matches the public terminology + makes the file purpose obvious.

**Why rename `camera_renderer.py` → `video_recorder.py`:** post-slim, the class only writes MP4. Camera-image-saving is gone; matplotlib-display is gone; "CameraRenderer" no longer describes the responsibility.

---

## 4. Renderer Protocol — spec + three concrete implementations

### 4.1 The Protocol (`visualization/base.py`)

```python
"""Renderer Protocol for the LangSteer visualization suite.

A Renderer is an observation-only viz hook that receives state updates from
VisualizationManager and produces an artifact (HTML file, live window, MP4, etc.).
Lifecycle: created at episode start, updated per step, closed at episode end.

The Protocol is intentionally minimal — three methods, no shared state, no rich
base class. This matches the "observation-only, no rich shared state" property
of every viz path in this repo. Concrete renderers may carry their own state
(open writers, mutable artists, save dirs) but the Manager never reads it.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Any


@runtime_checkable
class Renderer(Protocol):
    """Observation-only viz hook for the steering pipeline.

    Implementations should be idempotent on `update_state`/`tick` and safe to
    call after `close`. The Manager does not enforce ordering beyond
    `update_state` → `tick` → `close`.
    """

    def update_state(self, state: dict[str, Any]) -> None:
        """Stash the latest steering snapshot. Should be cheap.

        `state` is the dict returned by `VoxPoserSteering.get_costmap_state()`
        plus any keys merged by the Manager (e.g. `episode_id`). Concrete
        renderers pick the keys they care about.
        """
        ...

    def tick(self) -> None:
        """Produce/refresh the artifact for the current state. May be a no-op
        (e.g., HTML renderer only acts on stage activations, not every step)."""
        ...

    def close(self) -> None:
        """Release resources (file handles, tk window, video writers).
        Must be safe to call multiple times."""
        ...
```

**Why Protocol over ABC:** matches `typing.Protocol`'s structural-subtyping philosophy. No inheritance constraint, no `super().__init__()` boilerplate, and `@runtime_checkable` gives us `isinstance(x, Renderer)` for the Manager's registration validation. The three viz paths share zero implementation logic, so an ABC's common-base sharing buys nothing here.

**On `state: dict[str, Any]`:** the Manager passes through whatever `steering.get_costmap_state()` returns (today: `value_map`, `ee_pos`, `target`, `target_rotation`, `objects`, `stage_idx`, `num_stages`, `instruction`, `primitive`, `step`). Per Q3, this dict will gain `object` (current stage's object slot from `StageManager._stages[idx].object`). The Manager adds two top-level fields: `episode_id` (int) and `obs_rgb` (dict of camera frames, only when video is enabled — so video doesn't have to look up frames itself).

### 4.2 Concrete renderer #1 — `StageHtmlRenderer` (`visualization/renderers/stage_html.py`)

Thin adapter around `voxposer/visualizer.py::ValueMapVisualizer`. ~30 LoC:

```python
class StageHtmlRenderer:
    """Protocol adapter: routes stage-activation snapshots to ValueMapVisualizer."""

    def __init__(self, save_dir: str, quality: str = 'low'):
        from voxposer.visualizer import ValueMapVisualizer
        self._impl = ValueMapVisualizer({
            'visualization_save_dir': save_dir,
            'visualization_quality': quality,
        })
        self._last_stage_idx = None

    def update_state(self, state):
        self._state = state

    def tick(self):
        # Only emit on stage activation (stage_idx change), not every step.
        s = self._state
        if s.get('stage_idx') == self._last_stage_idx:
            return
        self._last_stage_idx = s.get('stage_idx')
        self._impl.visualize(
            s['value_map'],
            ee_pos_world=s.get('ee_pos'),
            objects=s.get('objects'),
        )

    def close(self):
        pass  # nothing to release
```

**Behavior preservation:** `stage_manager.py::_activate_stage` already calls `ValueMapVisualizer.visualize(...)` directly today. We *leave that call site alone* (Q4 told us not to refactor offline-script use of ValueMapVisualizer; the cleanest interpretation is "don't change ValueMapVisualizer's call contract"). The adapter is purely *additive* — when the Manager is enabled, it triggers a second render through the adapter on stage transitions. This produces redundant HTML files in two dirs (the steering-side direct path → `cfg.steering.visualization_save_dir`, the manager-side path → `cfg.visualization.html.save_dir`).

**To avoid duplication:** the cleanest approach is to *remove* the direct call from `stage_manager.py:425-430` and route exclusively through the Manager. This is a tiny edit (~6 lines deleted) and respects the "Manager owns all dispatch" principle from team-lead's Q5. Refactorer should:
1. Add the adapter and route through Manager.
2. Delete the direct `stage_manager.py:425-430` call site.
3. Keep `ValueMapVisualizer` itself unchanged — `scripts/test_voxposer.py` etc. still work.

### 4.3 Concrete renderer #2 — `LiveCostmapTkRenderer` (`visualization/renderers/live_costmap_tk.py`)

Renamed from `costmap_tk.py`. Existing `LiveCostmapWindow` class becomes `LiveCostmapTkRenderer`. Per Q3, plumb OBJECT label. Changes:

1. **Add `current_object` parameter to `update_state(...)`** — pulled from steering snapshot's new `object` key.
2. **Add OBJECT label in side panel** — mirror the PRIMITIVE block: tk.Label `OBJECT` header + `object_var` StringVar + label with monospace bold font. Place between PRIMITIVE and STAGE for visual hierarchy.
3. **Rename methods to match Protocol:** `update_state` already exists with matching name; add `tick()` (already exists); add `close()` (already exists). Just thin wrappers over the existing 420-line class — no behavioral change beyond the OBJECT label addition. The `update_costmap`/`tick_costmap` names on `VisualizationManager` go away in favor of Protocol-uniform `update_state`/`tick`.

Also: extract the OBB-edge constants and `_OBJ_COLORS`/`_PRIMITIVE_COLORS` to module-level (already there). No deep refactor inside.

### 4.4 Concrete renderer #3 — `VideoRecorder` (`visualization/renderers/video_recorder.py`)

Slimmed from `camera_renderer.py`. Keep:
- `__init__` (stash config)
- video state init (`_video_writers`, `_video_writer_sizes`, etc.)
- `start_video` (now folded into `update_state` first-call lazy init OR exposed via Manager `start_episode(episode_id)` hook — see §4.5)
- `_open_writer`
- `write_frame` (called `record_step` today; promote to part of `tick()` logic OR keep as a side-channel API)
- `stop_video`
- `_normalize_image`

Strip:
- `render_step` (per-frame PNG save, matplotlib display) — entire method gone.
- `display_cameras` — gone.
- `reset` step-counter machinery — gone (video is per-episode, not per-step).
- `save_images` config field — gone.
- `display_mode`, `show_live` — gone.

**Design question — does video fit cleanly into `tick()`?** Video needs per-waypoint frames (`env.set_waypoint_render_fn`-driven, finer granularity than the step callback). Today this is a side channel: `viz_manager.record_step(frames)` called from inside the waypoint callback. Two options for refactor:
- (a) **Keep the side channel.** Manager exposes a `record_waypoint(frames)` method that fans out to all renderers via a new optional Protocol method `on_waypoint(frames)` (default no-op). Cleanest mapping to current behavior.
- (b) **Fold into `tick()`.** Manager's `tick()` polls `env` for current frame and passes via `state['obs_rgb']`. Forces every renderer to deal with the frame data on every tick.

**Recommend (a)** — video is the only renderer that needs sub-step granularity, so adding an optional `on_waypoint` hook (or just an `on_frame(frames)` method that only `VideoRecorder` implements) is honest about that. The Protocol stays minimal (three core methods); `on_waypoint` is documented as an optional extension hook. Refactorer to add it as a separate optional Protocol or a `__post_tick_hook__` mechanism — surface a final decision in the iteration plan §5 iter 2.

Similarly, `start_recording(episode_id)` and `stop_recording()` map to a per-episode lifecycle. Two clean options:
- Add `on_episode_start(episode_id)` / `on_episode_end()` to the Protocol as default-no-op methods.
- Use `update_state(state)` with `state['lifecycle'] == 'episode_start'/'episode_end'`.

**Recommend the former** — explicit beats magic-string-keyed state.

**Net Protocol shape (final):**

```python
class Renderer(Protocol):
    def update_state(self, state): ...
    def tick(self): ...
    def close(self): ...
    # Optional hooks — default no-op via Protocol-compatible base class or
    # just `getattr(r, 'on_episode_start', lambda *_, **__: None)`.
    def on_episode_start(self, episode_id: int) -> None: ...
    def on_episode_end(self) -> None: ...
    def on_waypoint(self, frames: dict) -> None: ...
```

Manager iterates each method over its renderer list; renderers that don't implement an optional hook get a no-op (Manager uses `getattr(..., default=_noop)` so a strict 3-method Protocol implementer also works).

### 4.5 `VisualizationManager` rewrite (`visualization/manager.py`)

Goes from 272 LoC of toggle-keyed branching to ~80 LoC of list iteration:

```python
class VisualizationManager:
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self._renderers: list[Renderer] = []
        # Build the renderer list from config toggles (kept thin — no per-mode logic)
        if config.html.enabled:
            from .renderers.stage_html import StageHtmlRenderer
            self._renderers.append(StageHtmlRenderer(**config.html.kwargs()))
        if config.live_costmap.enabled:
            from .renderers.live_costmap_tk import LiveCostmapTkRenderer
            self._renderers.append(LiveCostmapTkRenderer(**config.live_costmap.kwargs()))
        if config.video.enabled:
            from .renderers.video_recorder import VideoRecorder
            self._renderers.append(VideoRecorder(**config.video.kwargs()))

    def update_state(self, state):
        for r in self._renderers:
            r.update_state(state)

    def tick(self):
        for r in self._renderers:
            r.tick()

    def on_episode_start(self, episode_id):
        for r in self._renderers:
            getattr(r, 'on_episode_start', _noop)(episode_id)

    def on_episode_end(self):
        for r in self._renderers:
            getattr(r, 'on_episode_end', _noop)()

    def on_waypoint(self, frames):
        for r in self._renderers:
            getattr(r, 'on_waypoint', _noop)(frames)

    def close(self):
        for r in self._renderers:
            r.close()

    def register(self, renderer: Renderer):
        """Programmatic extension point — append a custom renderer."""
        self._renderers.append(renderer)

    def is_enabled(self) -> bool:
        return bool(self._renderers)
```

**Note:** `register()` is the documented extension point for Task #7 (VLM scene-image renderer) and any future renderer that isn't worth a config-level toggle.

---

## 5. Iteration plan (4 iters, refactorer to execute step-by-step)

Each iter must end in a working repo (eval-runnable, no test regressions). Refactorer commits per iter; team-lead reviews before next iter.

### Iter 1 — Renderer Protocol + Manager rewrite (no behavior change)

**Scope:** introduce the new abstractions without removing anything yet.

1. Add `visualization/base.py` with the `Renderer` Protocol per §4.1.
2. Add `visualization/renderers/stage_html.py` adapter per §4.2 (with the dedup in `stage_manager.py` deferred to iter 3).
3. Rewrite `visualization/manager.py` per §4.5. Keep `update_costmap`/`tick_costmap`/`start_recording`/`stop_recording`/`record_step` as deprecated aliases routing to `update_state`/`tick`/`on_episode_start`/`on_episode_end`/`on_waypoint` so existing call sites in `run_experiment.py` keep working.
4. Adapt existing `LiveCostmapWindow` and `CameraRenderer` to implement Protocol method names. Camera_renderer keeps its old methods as aliases for now.
5. `from .renderers import ...` switches to import the to-be-renamed classes lazily (import-on-use inside Manager constructor).
6. **Validation:** existing `run_experiment.py` smoke test (1 task, 1 ep, no steering or with voxposer steering) produces an HTML + tk window. No code deletion yet.

**Iter 1 acceptance:** repo still runs identically; Protocol exists; Manager is the new dispatcher; nothing deleted.

### Iter 2 — Strip dead infrastructure (3 renderers + collectors + utils + tmp)

**Scope:** delete files. No new logic.

1. Delete `visualization/renderers/plotly_renderer.py`.
2. Delete `visualization/renderers/matplotlib_renderer.py`.
3. Delete `visualization/renderers/pybullet_renderer.py`.
4. Delete `visualization/collectors/` (entire dir).
5. Delete `utils/visualization/` (entire dir, including `trajectory_viz.py`, `plotly_viz.py`, `__init__.py`).
6. `rm -rf tmp/denoise_visualizer/` (untracked).
7. Update `visualization/renderers/__init__.py` — drop the 3 deleted imports + the to-be-renamed ones.
8. Slim `visualization/config.py` — drop `CameraVisualizationConfig`, `TrajectoryVisualizationConfig`, `ReferenceVisualizationConfig`, `RolloutVisualizationConfig`. Keep `LiveCostmapConfig`, `VideoConfig`, master `VisualizationConfig` (toggles: `html`, `live_costmap`, `video`).
9. Rewrite `conf/visualization/base.yaml` to ~25 lines (html + live_costmap + video sections).
10. Trim `conf/visualization/multistage.yaml` to ~20 lines (video block only).
11. Grep for stranded imports (`from utils.visualization`, `from .collectors`, etc.) and fix any remaining call sites.
12. **Validation:** `uv run python scripts/run_experiment.py steering=voxposer env.task=open_drawer num_episodes=1` succeeds; `ruff check .` passes.

**Iter 2 acceptance:** ~1700 LoC of viz code gone; the three target renderers still work via Manager; `ruff` clean.

### Iter 3 — Rename + extend live tk + dedup HTML

**Scope:** the user-facing feature work.

1. Rename `visualization/renderers/costmap_tk.py` → `live_costmap_tk.py`. Rename class `LiveCostmapWindow` → `LiveCostmapTkRenderer`. Update imports.
2. Rename `visualization/renderers/camera_renderer.py` → `video_recorder.py`. Rename class `CameraRenderer` → `VideoRecorder`. Strip image-saving + display methods per §4.4.
3. **Add OBJECT label to `LiveCostmapTkRenderer`** (per Q3):
   - Add `current_object: Optional[str] = None` to `update_state(...)` signature.
   - Add `object_var = tk.StringVar(value='—')` + a side-panel block analogous to PRIMITIVE (header label + value label, monospace bold).
   - Wire `state.get('object')` from the snapshot dict.
4. **Extend `StageManager.snapshot(...)`** in `steering/stage_manager.py` to include `"object": self._stages[self._current_stage_idx].object` in its return dict. ~2-line edit.
5. **Remove direct `ValueMapVisualizer.visualize(...)` call from `stage_manager.py::_activate_stage` (lines 425-430).** Route exclusively through the Manager's `StageHtmlRenderer`. Same call to ValueMapVisualizer happens; the call site moves from `stage_manager.py` to `stage_html.py`'s `tick()`. The `_visualizer` field + `_visualize` toggle on `StageManager` become dead — remove them too (and the `cfg.steering.visualize` flag on the steering config; replace with `cfg.visualization.html.enabled`).
6. Remove the deprecated Manager aliases (`update_costmap`, `tick_costmap`, `start_recording`, `stop_recording`, `record_step`) — update `run_experiment.py` to call the new Protocol method names directly.
7. **Validation:** voxposer smoke run produces (a) HTML files in `outputs/.../visualizations/`, (b) tk window with OBJECT label populated and PRIMITIVE label color-coded, (c) MP4s under `multistage.yaml`.

**Iter 3 acceptance:** OBJECT label visible; HTML is generated exactly once per stage activation via the Manager; legacy aliases gone.

### Iter 4 — Wire `run_evaluation.py` + final smoke

**Scope:** close the eval gap.

1. Patch `scripts/run_evaluation.py` to mirror `run_experiment.py`'s viz wiring:
   - Copy the `VisualizationConfig.from_dict` block (~7 lines) into the setup region.
   - Inside the existing `step_callback` (line 336-354), add: `if viz_manager is not None and steering is not None and hasattr(steering, 'get_costmap_state'): state = steering.get_costmap_state(obs.ee_pose[:3]); if state: viz_manager.update_state(state); viz_manager.tick()`.
   - Add `viz_manager.on_episode_start(ep_idx)` before the rollout and `viz_manager.on_episode_end()` after; wire the waypoint hook for video.
   - Call `viz_manager.close()` once at the end of the outer task loop.
2. Verify `conf/evaluation/langsteer_primitive_object.yaml` (or wherever the eval default config lives) doesn't accidentally enable a dead viz path.
3. **Validation — final smoke (the §6 protocol).**

**Iter 4 acceptance:** running `uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer` with `visualization=base visualization.live_costmap.enabled=true visualization.html.enabled=true visualization.video.enabled=true` produces all three artifacts.

---

## 6. Validation — smoke test producing all three artifacts

Single smoke run. Conditions:

```bash
DISPLAY=:0 uv run python scripts/run_evaluation.py \
    policy=diffuser_actor_primitive_object \
    steering=voxposer \
    visualization=base \
    visualization.live_costmap.enabled=true \
    visualization.html.enabled=true \
    visualization.video.enabled=true \
    num_episodes=1 \
    task_order=[open_drawer]  # 1 task × 1 ep, ~30s
```

**Pass criteria (all must hold):**

| Artifact | Check | Pass if… |
|---|---|---|
| Stage HTMLs | `ls outputs/<hydra-dir>/html/*.html` | ≥ 1 file per stage activation (i.e., 1 file for single-stage `open_drawer`, ≥2 for `place_in_slider`); each opens in browser and shows a 3D Plotly figure with the value-map volume + OBBs. |
| Live tk window | Visual inspection during run | Window opens, value map rotates on drag, scrolls on zoom, side panel shows PRIMITIVE (color-coded) + OBJECT (per Q3) + STAGE i/N + STEP + INSTRUCTION. |
| MP4 video | `ls outputs/<hydra-dir>/videos/episode_0000_*.mp4` | 1 file per camera (static + gripper); both play back at configured fps; static at requested resolution (default native, overridden in multistage.yaml). |
| No regressions | Compare success/failure on `open_drawer` vs baseline | Same outcome (no viz code should affect inference). |
| `ruff check .` | Lint cleanup | Passes. |
| Import surface | `grep -r "from utils.visualization\|from visualization.collectors" .` | Zero matches outside `.venv/`, `__pycache__/`, `tmp/`. |

**On headless servers** (no `DISPLAY`): `visualization.live_costmap.enabled=false` should be safe and the HTML + video artifacts should still produce. Test this fallback as part of the smoke run by adding a second invocation: `unset DISPLAY; uv run ... visualization.live_costmap.enabled=false` — must succeed without trying to instantiate Tk.

**Regression coverage:** because this task only touches viz code, the 28×5 canonical eval isn't required. A 1-task smoke is sufficient. Optional: refactorer runs a 5-task × 3-ep mini-eval comparing pre- and post-refactor success-rate to confirm zero behavioral drift, but this is gold-plating; we accept the smoke as sufficient given the no-inference-touch constraint.

---

## 7. Risks & gotchas

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Tkinter on headless GPU servers** — CALVIN often runs on remote nodes without X11. `tk.Tk()` crashes immediately with `TclError: no display name`. | High | (a) `LiveCostmapTkRenderer.__init__` wraps `tk.Tk()` in try/except, logs a clear warning + falls back to a no-op renderer. (b) The default `visualization.live_costmap.enabled=false` keeps the window off unless explicitly toggled. (c) Document in `README.md` (out of plan-scope; refactorer to write a one-paragraph note). |
| **X11 forwarding latency** — over SSH `-X`, the matplotlib redraw at every step can balloon eval wall-clock by 5-20%. | Medium | `refresh_interval: int = 1` in `LiveCostmapConfig` (already exists) lets the user throttle. Document the perf knob. |
| **Duplicate HTML generation during iter 1-2 transition.** | Low — iter 3 explicitly removes the direct call site. | Iter 3 acceptance check #5 requires "exactly once per stage activation." |
| **`run_evaluation.py` step_callback refactor breaks the steering update_dash legacy path** (lines 350-354 of run_evaluation.py have a `hasattr(steering, "update_dash")` block that looks like dead code). | Low | Iter 4 grep `update_dash` to confirm it's vestigial; if so, delete those lines as a bonus cleanup. Otherwise leave it alone. |
| **`ValueMapVisualizer` config dict format change** breaks `scripts/test_voxposer.py` / `calibrate_voxposer_objects.py`. | Very low | StageHtmlRenderer wraps the same constructor signature (`{'visualization_save_dir': ..., 'visualization_quality': ...}`). No change to ValueMapVisualizer itself. Q4 boundary respected. |
| **`stage_manager.py::_visualizer` removal interacts with Task 4 work-in-flight.** | Medium | Coordinate timing with team-lead — refactorer should rebase onto post-#4 main before starting iter 3. The `_visualizer` field is currently only touched in `_activate_stage` and `_init_lmp_system`; conflict surface should be small. |
| **Deprecated aliases on Manager confuse reviewers in iter 1.** | Low | Aliases are explicitly removed in iter 3; their lifespan is two iters. Refactorer to add a one-line `DeprecationWarning` when each alias is hit to make the deletion obvious. |

---

## 8. Out of scope (explicit)

- `voxposer/visualizer.py::ValueMapVisualizer` internals. Wrapped, not refactored (Q4).
- `scripts/test_voxposer.py`, `scripts/calibrate_voxposer_objects.py` — offline tools, zero edits (Q4).
- Adding new viz capabilities beyond OBJECT label (e.g., target_rotation 3D arrow rendering — `LiveCostmapWindow` already accepts the field but explicitly notes "not yet rendered"). Future work, separate plan.
- VLM scene-image rendering — Task #7 (see §9).
- Multi-rollout 3D trajectory analysis (the deleted `plotly_renderer.py` feature). If anyone needs this back, it returns as a new `Renderer` implementation through `Manager.register(...)`, not by un-deleting the dead code.
- PyBullet GUI playback (the deleted `pybullet_renderer.py` feature). CALVIN's own `enable_gui=true` flag still works for in-process rendering; the external playback script doesn't.
- Reference trajectory matplotlib plots (the deleted `matplotlib_renderer.py` feature). If needed, re-add as a `ReferenceTrajectoryRenderer` through `Manager.register(...)`.
- Per-frame PNG image saving. Dead capability; not coming back.

---

## 9. Future-work hook — Task #7 (VLM scene-image ingestion)

Team-lead noted Task #7 absorbs the scene-grounding work that came out of Task 3b's P4 perturbation deferral. The viz overlap is: a VLM-based composer ingests an overhead camera image at composer-query time and uses it to disambiguate scene-info-dependent instructions ("Turn on the light", "Pick up the block from the drawer").

**Extension point left in Task 5:**

1. **`VideoRecorder` already captures the static overhead camera** at high resolution. Task #7 can reuse `env.render_high_res_static(...)` (the same code path video uses for hi-res recording) to grab the composer-query frame. No new hook needed — but documenting the contract in `video_recorder.py`'s docstring makes the dependency discoverable.

2. **The `Renderer` Protocol is the natural shape for a future `SceneImageRenderer`** — one that observes the overhead frame at episode start, dumps it to `outputs/.../scene_images/`, and (optionally) annotates the VLM's bounding-box outputs. Task #7's composer module would register one via `Manager.register(SceneImageRenderer(...))` rather than baking image dumps into the LMP pipeline directly.

3. **`state['obs_rgb']`** on episode start is already part of the state dict in the proposed shape — the VLM renderer can consume it without any Manager change.

**What this plan does NOT decide for Task #7:**
- Whether the VLM composer is invoked at episode-start only, on every stage, or with retry.
- The image-resolution / FOV requirements for VLM grounding.
- Whether bounding-box overlays on the static frame become a runtime artifact or a debug-only artifact.

All three are Task #7 territory. Task 5 leaves the `register(...)` hook + the Protocol shape; nothing more.

---

## 10. Approval checklist (team-lead before refactorer kickoff)

- [ ] §0 task-status check matches the team-lead's mental model — no split.
- [ ] §1 brief excerpt is verbatim from `refactoring_tasks.md`.
- [ ] §2 surface inventory matches the survey table sent before plan-draft.
- [ ] §3 file layout matches the intended post-refactor shape.
- [ ] §4 Protocol shape adopts Q5 (a) verbatim (with the optional `on_episode_*` + `on_waypoint` hooks documented).
- [ ] §5 iteration boundaries are sequenced correctly (no iter ships a half-state).
- [ ] §6 smoke test produces all three artifacts on a 1-task × 1-ep run.
- [ ] §7 headless-tk risk handled before refactorer touches `LiveCostmapTkRenderer.__init__`.
- [ ] §8 out-of-scope items match Q1-Q4 decisions.
- [ ] §9 Task #7 hook is a stub (no design imported from Task #7).

On approval → ScheduleWakeup for Task #4 completion; refactorer kickoff message includes a pointer to this plan + the iteration boundary it should commit at.
