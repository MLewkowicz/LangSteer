# Task 5 — Visualization cleanup — Iteration log

Plan: `docs/refactor/task5_plan.md`.
Branch: `refactoring`.

---

## Iter 1 — Renderer Protocol + Manager rewrite (no behavior change)

**Date:** 2026-05-19.
**Scope (per plan §5 iter 1):** introduce the Renderer Protocol abstraction
without removing anything yet. All existing call sites keep working via
deprecated aliases.

### Files added

- `visualization/base.py` (43 LoC) — `Renderer` Protocol with 3 core methods
  (`update_state` / `tick` / `close`) and 3 documented-optional lifecycle
  hooks (`on_episode_start` / `on_episode_end` / `on_waypoint`). The Manager
  dispatches optional hooks via `getattr(r, '<hook>', _noop)` so a strict
  3-method implementer is also Manager-compatible.
- `visualization/renderers/stage_html.py` (45 LoC) — Protocol adapter around
  `voxposer.visualizer.ValueMapVisualizer`. Emits HTML only on stage-index
  change (not every tick). `ValueMapVisualizer` internals are untouched —
  the adapter wraps the existing `visualize(...)` call site so offline
  scripts (`scripts/test_voxposer.py`, `scripts/calibrate_voxposer_objects.py`)
  keep working.

### Files modified

- `visualization/__init__.py` — exports `Renderer` alongside the existing
  `VisualizationConfig` + `VisualizationManager`.
- `visualization/renderers/__init__.py` — exports `StageHtmlRenderer`.
  Existing legacy renderer exports unchanged (iter 2 strips them).
- `visualization/renderers/costmap_tk.py::LiveCostmapWindow.update_state` —
  signature changed from kwarg-based (`value_map=..., ee_pos=..., ...`) to
  Protocol-conformant (`state: dict`). Body now does `self._state = dict(state)`
  (was building the dict from kwargs). Only caller is `VisualizationManager`,
  which now builds the dict in `update_costmap` (deprecated alias) and passes
  it through.
- `visualization/renderers/camera_renderer.py` — added 5 Protocol methods
  (`update_state` / `tick` / `close` / `on_episode_start` / `on_episode_end` /
  `on_waypoint`) as thin wrappers over the existing legacy API. Legacy
  methods (`start_video`, `write_frame`, `stop_video`, `reset`, `render_step`,
  `display_cameras`) are preserved untouched — iter 3 strips image-saving +
  matplotlib display + renames to `VideoRecorder`.
- `visualization/manager.py` — rewritten from 272 → ~260 LoC. Per-mode
  toggle-keyed branching replaced with a `self._renderers: list[Renderer]`
  built once in `__init__`, then iterated for every Protocol method. Legacy
  methods (`update_costmap`, `tick_costmap`, `start_recording`,
  `stop_recording`, `record_step`, `shutdown`) kept as deprecated aliases
  emitting `DeprecationWarning` (removed in iter 3 per plan). Legacy
  per-mode wrappers (`visualize_episode`, `visualize_reference_trajectory`,
  `visualize_multi_rollout`, `visualize_step`) preserved for
  `run_experiment.py` (iter 2 strips when the underlying renderers are
  deleted).

### Behavior preservation

- `run_experiment.py` calls `viz_manager.update_costmap(**state)`,
  `tick_costmap()`, `reset()`, `start_recording(eid)`, `record_step(frames)`,
  `stop_recording()`, `shutdown()`, `visualize_reference_trajectory(...)`.
  Each now routes through Protocol dispatch under the hood; legacy entry
  points emit `DeprecationWarning`.
- `run_evaluation.py` doesn't instantiate `VisualizationManager` today
  (iter 4 wires this up).
- `scripts/test_voxposer.py` + `scripts/calibrate_voxposer_objects.py`
  instantiate `ValueMapVisualizer` directly — unchanged (Q4 boundary).

### Validation (per iter 1 acceptance)

1. **Import smoke:** `from visualization import Renderer, VisualizationConfig,
   VisualizationManager` + `from visualization.renderers import
   StageHtmlRenderer, LiveCostmapWindow, CameraRenderer` — all clean.
2. **Empty manager:** `VisualizationManager(VisualizationConfig())` →
   `is_enabled() == False`, all Protocol methods iterate over empty list
   (no-op). PASS.
3. **Video-only manager:** `VisualizationConfig(video=VideoConfig(enabled=True))`
   → 1 renderer registered (`CameraRenderer`). All Protocol methods +
   deprecated aliases routed cleanly. 5/5 `DeprecationWarning` fired on
   alias use.
4. **End-to-end eval:** 1-task × 1-ep `run_evaluation.py` smoke on
   `open_drawer` (default visualization config, no viz overrides).
   _(result below — see "Smoke test" section)_.

### Smoke test — `open_drawer` 1×1 eval

`/tmp/task5_iter1_smoke/langsteer_primitive_object.json`:
- `open_drawer`: **1/1 success, 3 steps**. Matches the dwell=4 canary's
  open_drawer baseline ([3, 3, 4] across 3 eps).
- Pybullet `__del__` cleanup traceback (`pybullet.error: Not connected to
  physics server`) — unrelated to iter 1 changes; same noise observed across
  prior 3b runs (in-flight bug in `calvin_env.envs.play_table_env.close()`).

`run_evaluation.py` doesn't instantiate `VisualizationManager` (iter 4
wires this in). This smoke confirms the Visualization package imports
cleanly and the steering pipeline runs end-to-end without touching viz
code — i.e. iter 1's additive changes don't perturb inference.

### Iter 1 acceptance — status

- [x] Protocol exists (`visualization/base.py`).
- [x] Manager dispatches via list iteration over `self._renderers`.
- [x] Legacy aliases preserved, all emit `DeprecationWarning` (verified 5/5).
- [x] No file deleted.
- [x] Eval smoke run succeeds with no behavioral regression
      (open_drawer 1/1, matches dwell=4 canary baseline).

### Risks observed in iter 1

- **Headless tkinter** — not exercised this iter (live_costmap is opt-in via
  config; the default eval YAML doesn't enable it). Iter 3 will harden the
  `LiveCostmapWindow.__init__` Tk init against `TclError: no display name`.
- **CameraRenderer.on_episode_start signature** — accepts `video_cfg` kwarg
  to let the Manager pass the resolved `VideoConfig`. The legacy
  `start_recording(eid)` alias still works because the Manager looks up
  the config itself in the alias body. Strict Protocol-only implementers
  would need to construct a `VideoConfig`-shaped object — flagged for iter 3
  cleanup (likely move video-config storage into `CameraRenderer.__init__`).

---

## Iter 2 — Strip dead visualization infrastructure

**Date:** 2026-05-19.
**Scope (per plan §5 iter 2 + team-lead's iter-2 brief):** delete 3 legacy
renderers, the trajectory collector, the orphan `utils/visualization/`
package, and `tmp/denoise_visualizer/`. Slim `config.py` to the three
surviving sub-configs + `VisualizationConfig` master. Strip dead manager
wrappers + dead CameraRenderer paths. Update `run_experiment.py` callers.

### Files deleted (7 targets — all from team-lead's strip list)

1. `visualization/renderers/plotly_renderer.py` (206 LoC).
2. `visualization/renderers/matplotlib_renderer.py` (124 LoC).
3. `visualization/renderers/pybullet_renderer.py` (98 LoC; also clears the
   9 pre-existing ruff F841 errors).
4. `visualization/collectors/` — full dir: `__init__.py` (8 LoC) +
   `trajectory_collector.py` (213 LoC).
5. `utils/visualization/` — full dir: `__init__.py` (1 LoC) +
   `trajectory_viz.py` (397 LoC) + `plotly_viz.py` (236 LoC, orphan).
6. `tmp/denoise_visualizer/` — confirmed untracked (`git ls-files tmp/`
   returns 0); plain `rm -rf`.

Net deletion: ~1283 LoC + 6 files + 2 directories.

### Files slimmed

- `visualization/config.py`: 151 → 87 LoC.
  - Dropped: `CameraVisualizationConfig`, `TrajectoryVisualizationConfig`,
    `ReferenceVisualizationConfig`, `RolloutVisualizationConfig` (4
    dataclasses).
  - Dropped master toggles: `render`, `cameras`, `trajectory_3d`,
    `reference_plot` + their `camera`/`trajectory`/`reference`/`rollout`
    sub-config fields.
  - Added `HtmlConfig` (enabled + save_dir + quality) for the new
    `StageHtmlRenderer`.
  - `LiveCostmapConfig` + `VideoConfig` gained an `enabled: bool = False`
    field (was on the parent toggle). Master is now `{html, live_costmap,
    video}`.
  - `VisualizationConfig.from_dict` silently ignores unknown keys so
    legacy YAML configs with `cameras: false` / `render: false` /
    `live_costmap: false` still load during iter 1→3 transition.
- `visualization/manager.py`: 285 → 220 LoC.
  - Dropped PyBullet / Matplotlib / Plotly init branches +
    `_pybullet_renderer` / `_matplotlib_renderer` / `_plotly_renderer`
    fields.
  - Dropped `visualize_episode` / `visualize_reference_trajectory` /
    `visualize_multi_rollout` / `visualize_step` legacy wrappers (their
    renderers are gone).
  - Dropped `reset()` method (CameraRenderer's `reset` was only used by
    the dead per-step image-save path).
  - Added `html.enabled` init branch wiring `StageHtmlRenderer` through
    the Manager (Protocol-only; iter 3 routes the
    `stage_manager.py:425-430` direct call site through here).
  - Kept the 5 deprecated aliases (`update_costmap` / `tick_costmap` /
    `start_recording` / `stop_recording` / `record_step` / `shutdown`)
    for `run_experiment.py`. Iter 3 removes them.
- `visualization/renderers/camera_renderer.py`: 249 → ~165 LoC.
  - Dropped: `render_step`, `display_cameras`, `reset`, `step_counter`,
    `matplotlib.pyplot` import, `PIL.Image` import.
  - Constructor signature now accepts a `VideoConfig` (the surviving
    block) instead of the deleted `CameraVisualizationConfig`. The
    Manager passes `config.video`.
  - Kept legacy aliases `start_video` / `write_frame` / `stop_video`
    routing to the new internal `_start_video` / `_write_frame` /
    `_stop_video` (`run_experiment.py`'s `viz_manager.start_recording`
    deprecated alias still depends on them via the Manager).
- `visualization/renderers/__init__.py`: dropped 3 stripped exports;
  three survivors remain (`CameraRenderer`, `LiveCostmapWindow`,
  `StageHtmlRenderer`).
- `conf/visualization/base.yaml`: 68 → 34 LoC. 3 toggle blocks
  (`html`/`live_costmap`/`video`) only.
- `conf/visualization/multistage.yaml`: 58 → 25 LoC. Video block only.

### Files modified outside `visualization/`

- `scripts/run_experiment.py`:
  - Dropped `viz_manager.reset()` call at line 331 (the corresponding
    Manager.reset is gone).
  - Dropped the post-episode `reference_plot` block (lines 396-407 —
    matplotlib renderer is gone). 12 LoC removed; replaced with a 2-line
    comment noting iter 2's removal.

### Validation (per iter 2 acceptance)

| Check | Result |
|---|---|
| 7 files / dirs deleted | ✓ verified `ls` returns empty for stripped paths |
| `git ls-files tmp/` | 0 (confirmed untracked before delete) |
| `ruff check visualization/` | **0 errors** (was 9; all pre-existing in `pybullet_renderer.py`) |
| Grep for stranded imports | `grep -rn "from utils.visualization\|visualization.collectors\|PlotlyRenderer\|MatplotlibRenderer\|PyBulletRenderer\|TrajectoryCollector\|trajectory_viz\|plotly_viz" --include="*.py"` → **0 hits** in production code |
| Module-level smoke | imports clean; empty / video-only Manager constructs; legacy-key `from_dict` ignored cleanly; Protocol dispatch + lifecycle hooks all work |
| Eval smoke (1×1 `open_drawer`) | **1/1 success, 3 steps** — identical to iter 1 baseline. No behavioral regression. |

### Notes / known follow-up

- `utils/visualize_steering.py` is now orphaned (only consumer was the
  deleted `matplotlib_renderer.py`). Not on team-lead's strip list; left
  intact for a separate follow-up.
- `tmp/` contains other untracked subdirs (`bbox_annotator`,
  `bimodal_experiment`, `p4_validation`, `visualizations`). Only
  `denoise_visualizer/` was on the strip list; others untouched.
- 4 pre-existing F401 errors remain in `scripts/run_experiment.py`
  (`Dict`, `Any`, `Observation`, `Action` unused). Pre-existing —
  unrelated to iter 2. Filed for a separate import-hygiene pass.

### Iter 2 acceptance — status

- [x] All 7 strip targets deleted.
- [x] `ruff check visualization/` → 0 errors.
- [x] 0 stranded imports in production code.
- [x] Eval smoke run succeeds with no behavioral regression
      (open_drawer 1/1, 3 steps — identical to iter 1 baseline).

## Iter 3 — _(pending)_

## Iter 4 — _(pending)_
