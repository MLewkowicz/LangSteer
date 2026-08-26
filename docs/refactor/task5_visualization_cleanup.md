# Task 5 — Visualization Cleanup

**Status:** ✅ done  
**Plan:** [task5_plan.md](task5_plan.md)  
**Iteration log:** [task5_log.md](task5_log.md)  
**Commits:** `16a7571` (iter 1) · `5407942` (iter 2) · `27e224f` (iter 3) · `5c44b80` (iter 4)  
**Branch:** `refactoring`

---

## Goal

The visualization package had grown to five renderers across two parallel packages (`visualization/` and `utils/visualization/`), plus a dead collector, scattered orphan scripts, and an untracked `tmp/denoise_visualizer/`. Three of the five renderers (Plotly multi-rollout, Matplotlib reference-plot, PyBullet GUI playback) had no live callers. The live tkinter window existed but was only wired into `run_experiment.py`, not the dominant `run_evaluation.py` path. HTML stage output was emitted by two independent code paths (direct `ValueMapVisualizer` call in `stage_manager.py` + Manager-routed `StageHtmlRenderer`). Task 5 stripped the dead infrastructure, formalized a `Renderer` Protocol as the single extension point, completed the live tk window (OBJECT label, headless safety), and wired the Manager into `run_evaluation.py`.

**Net result: 5 → 3 renderers, ~1700 LoC deleted, all three live artifacts confirmed end-to-end.**

---

## Plan summary

Full plan: [task5_plan.md](task5_plan.md)

- **Keep:** stage-specific HTML rendering, live tkinter 3D window, MP4 video. These directly satisfy the user brief.
- **Add:** `Renderer` Protocol (3 core methods + 3 optional lifecycle hooks) as the formal extension point. New renderers register via `Manager.register(renderer)` — no Manager internals change needed.
- **Strip:** plotly/matplotlib/pybullet renderers, trajectory collectors, `utils/visualization/`, `utils/visualize_steering.py`, `tmp/denoise_visualizer/`.
- **Fix:** HTML duplicate-emit via single call site through `StageHtmlRenderer`; headless tkinter guard; OBJECT label in side panel.
- **Wire:** `VisualizationManager` into `run_evaluation.py` with a `visualization_overrides` mechanism mirroring the existing `steering_overrides` pattern.
- **4-iter cadence:** additive only → strip dead code → renames + cleanup → wire eval. Each iter committed individually; eval smoke after every iter.

---

## Implementation

### Files added

| File | Lines | Role |
|------|-------|------|
| `visualization/base.py` | 43 | `Renderer` Protocol — 3 core methods (`update_state(state)`, `tick()`, `close()`) + 3 optional lifecycle hooks (`on_episode_start(ep_id)`, `on_episode_end()`, `on_waypoint(frames)`). Manager dispatches optional hooks via `getattr(r, hook, _noop)` so strict 3-method implementers are also Manager-compatible. |
| `visualization/renderers/stage_html.py` | 45 | `StageHtmlRenderer` — thin Protocol adapter around `voxposer.visualizer.ValueMapVisualizer`. Emits HTML only on stage-index change (not every tick). `ValueMapVisualizer` internals untouched — offline scripts keep working. |

### Files deleted (~1700 LoC net)

| Path | LoC | Reason |
|------|-----|--------|
| `visualization/renderers/plotly_renderer.py` | 206 | Multi-rollout 3D trajectory analysis — 0 callers |
| `visualization/renderers/matplotlib_renderer.py` | 124 | Reference-trajectory plots — 0 callers |
| `visualization/renderers/pybullet_renderer.py` | 98 | PyBullet GUI playback — 0 callers; was also the source of 9 pre-existing ruff F841 errors |
| `visualization/collectors/` (2 files) | 221 | Trajectory collector feeding the deleted Plotly renderer |
| `utils/visualization/` (3 files) | 634 | `trajectory_viz.py` + `plotly_viz.py` + `__init__.py` — orphan package |
| `utils/visualize_steering.py` | ~80 | Only consumer was the deleted `matplotlib_renderer.py` |
| `tmp/denoise_visualizer/` | — | Untracked scratch dir; confirmed via `git ls-files tmp/` before delete |

### Files renamed (via `git mv`)

| Before | After |
|--------|-------|
| `visualization/renderers/costmap_tk.py` · `LiveCostmapWindow` | `visualization/renderers/live_costmap_tk.py` · `LiveCostmapTkRenderer` |
| `visualization/renderers/camera_renderer.py` · `CameraRenderer` | `visualization/renderers/video_recorder.py` · `VideoRecorder` |

### Files modified (key changes)

| File | Change summary |
|------|----------------|
| `visualization/manager.py` | 272 → ~220 lines. Per-mode toggle-keyed branches → `self._renderers: list[Renderer]` built once in `__init__`, iterated uniformly. Dropped PyBullet/Matplotlib/Plotly init branches + 4 dead wrapper methods (`visualize_episode`, `visualize_reference_trajectory`, `visualize_multi_rollout`, `visualize_step`). Added `html.enabled` init branch wiring `StageHtmlRenderer`. Deprecated aliases added (iter 1) then dropped (iter 3) — net zero. |
| `visualization/config.py` | 151 → 87 lines. Dropped 4 dead sub-configs (`Camera`, `Trajectory`, `Reference`, `Rollout`). Added `HtmlConfig`. Master `VisualizationConfig` now: `{html, live_costmap, video}`. `from_dict` silently ignores unknown keys for legacy YAML compat. |
| `visualization/renderers/live_costmap_tk.py` | OBJECT label added: `object_var = StringVar()` + side-panel block (above PRIMITIVE, 18pt monospace bold, color-coded via `_OBJ_COLORS`). Headless hardening: `__init__` wraps `_build_window()` in `try/except tk.TclError`; sets `self._disabled = True` on failure; all Protocol methods early-return when disabled. |
| `visualization/renderers/video_recorder.py` | `VideoConfig` moved to `__init__` (was per-call kwarg). Dropped: `render_step`, `display_cameras`, `reset`, `matplotlib`, `PIL.Image`. Stripped `start_video`/`write_frame`/`stop_video` legacy aliases; internal methods renamed `_open_writer`/`_write_frame`/`_stop_video`. ~165 LoC (was 249). |
| `steering/stage_manager.py` | Dropped `visualize: bool` constructor param, `self._visualize`, `self._visualizer`, `ValueMapVisualizer` import, and the direct `self._visualizer.visualize(...)` call in `_activate_stage`. Added `"object"` key to `snapshot()` dict alongside existing `"primitive"`. |
| `steering/voxposer_steering.py` | Removed `visualize=cfg.get("visualize", False)` kwarg from `StageManager(...)` constructor call. |
| `conf/steering/voxposer.yaml` | Removed `visualize`, `visualization_quality`, `visualization_save_dir` keys (now at `cfg.visualization.html.*`). |
| `conf/visualization/base.yaml` | 68 → 34 lines. 3 toggle blocks only (`html`/`live_costmap`/`video`). |
| `conf/visualization/multistage.yaml` | 58 → 25 lines. Video block only. |
| `scripts/run_experiment.py` | Migrated from deprecated Manager aliases to Protocol methods. Dropped `viz_manager.reset()` call + dead `reference_plot` block (12 LoC). Updated `cfg.steering.visualize*` wiring → `cfg.visualization.html.*`. |
| `scripts/run_evaluation.py` | Manager instantiated in `run_condition`. `step_callback` drops dead `update_dash` block; adds `update_state(...)` + `tick()`. Episode loop: `on_episode_start`/`on_episode_end` + per-waypoint `on_waypoint(frames)`. Helper `_make_waypoint_render(...)` factored to module-level (same shape as `run_experiment.py`'s closure). `visualization_overrides` mechanism added for eval YAML opt-in. `viz_manager.close()` before `env.close()`. |

### Iteration map

| Iter | Commit | Change | Validation |
|------|--------|--------|------------|
| 1 | `16a7571` | Protocol + Manager rewrite. No deletes; all deprecated aliases preserved. `StageHtmlRenderer` added. | Import smoke ✅; empty/video Manager ✅; 5/5 `DeprecationWarning` fired ✅; eval 1×1 open_drawer 1/1 ✅ |
| 2 | `5407942` | Strip: 3 dead renderers, collectors dir, `utils/visualization/`, `tmp/denoise_visualizer/`. Slim `config.py`, slim `camera_renderer.py` (→ drop image-save + matplotlib). | ruff 0 errors ✅; 0 stranded imports ✅; eval 1×1 1/1 ✅ |
| 3 | `27e224f` | Renames (`costmap_tk` → `live_costmap_tk`, `camera_renderer` → `video_recorder`). OBJECT label. HTML dedup (remove direct `stage_manager.py` call). Drop 6 deprecated Manager aliases. Headless tk guard. Strip `utils/visualize_steering.py`. | ruff 0 ✅; `git grep` 0 hits for renamed symbols ✅; headless tk no-crash ✅; OBJECT plumbing end-to-end ✅; eval 1×1 1/1 ✅ |
| 4 | `5c44b80` | Wire `run_evaluation.py`. Drop `update_dash` vestige. Final end-to-end smoke (all 3 artifacts). | ruff 0 ✅; 3 HTML files + 2 MP4 files produced ✅; live tk opened ✅; headless fallback no-crash ✅; eval 1×1 1/1 ✅ |

---

## Key design choices

1. **Protocol over ABC.** `Renderer` is a `typing.Protocol` — no inheritance required; any duck-typed object with 3 methods is a valid renderer. Optional lifecycle hooks use `getattr(r, hook, _noop)` dispatch so the 3-method minimum never breaks.
2. **HTML routing exclusively through Manager.** The direct `ValueMapVisualizer.visualize(...)` call in `stage_manager.py` (pre-Task-5 HTML source) is removed. `StageHtmlRenderer` is now the single HTML call site, routed via Manager. `ValueMapVisualizer`'s own API is preserved unchanged for offline scripts (`scripts/test_voxposer.py`, `scripts/calibrate_voxposer_objects.py`).
3. **Tk window was already built.** `LiveCostmapWindow` pre-existed Task 5 with the rotatable 3D view, PRIMITIVE label, stage/step/instruction. Task 5 added OBJECT label + headless safety + renamed to `LiveCostmapTkRenderer` — extend rather than build from scratch.
4. **Video kept as the third renderer.** Slimmed `VideoRecorder` (~165 LoC, was 249) keeps MP4 production for demos/papers and demonstrates the Protocol extensibility property — new renderers register via `Manager.register(renderer)`.
5. **Per-iter commits** — first use of per-iter commits in this refactor. Each iter leaves a runnable repo with a clean rollback point.

---

## Behavior preserved / removed / relocated

| Category | Item | Notes |
|----------|------|-------|
| **Preserved** | Stage HTML output | Now exclusively via `StageHtmlRenderer` ↔ Manager dispatch |
| **Preserved** | Live tk 3D costmap window | Extended: OBJECT label, headless guard |
| **Preserved** | MP4 video recording | `VideoRecorder` (~165 LoC, slimmed) |
| **Preserved** | `voxposer.visualizer.ValueMapVisualizer` public API | Unchanged; offline scripts keep working |
| **Preserved** | `run_experiment.py` visualization behavior | Migrated to Protocol methods; behavior identical |
| **Removed** | Plotly multi-rollout renderer | 0 callers; deleted |
| **Removed** | Matplotlib reference-trajectory renderer | 0 callers; deleted |
| **Removed** | PyBullet GUI playback renderer | 0 callers; deleted (also cleared 9 pre-existing ruff errors) |
| **Removed** | `visualization/collectors/` | Feeding the deleted Plotly renderer |
| **Removed** | `utils/visualization/` (634 LoC) | Orphan package |
| **Removed** | Direct `ValueMapVisualizer.visualize()` in `stage_manager.py` | HTML dedup — single call site now |
| **Removed** | `update_dash` in `run_evaluation.py` step_callback | Dead vestige; 0 callers |
| **Removed** | `visualize` / `visualization_save_dir` from `conf/steering/voxposer.yaml` | Now at `cfg.visualization.html.*` |
| **Relocated** | 6 Manager deprecated aliases | Added iter 1, dropped iter 3 — net zero |
| **Added** | `visualization/base.py::Renderer` Protocol | Extension point for Task #7 + future renderers |
| **Added** | OBJECT label in live tk side panel | `StageManager.snapshot()` now emits `"object"` key |
| **Added** | `VisualizationManager` in `run_evaluation.py` | Was experiment-only; eval is now first-class |
| **Added** | `visualization_overrides` in eval YAML schema | Mirrors `steering_overrides` pattern |

---

## Smoke tests / validation

All four iters produced `open_drawer 1/1 success, 3 steps` — identical across every iter. No behavioral regression introduced.

**Final end-to-end smoke (iter 4, all 3 artifacts):**

| Artifact | Result |
|----------|--------|
| HTML | 3 files: `22_50_15.html`, `22_50_20.html`, `latest.html` (1 per stage activation + `latest.html` equivalent) |
| MP4 | 2 files: `episode_0000_static.mp4`, `episode_0000_gripper.mp4` (both opened at 200×200; frames via `on_waypoint`) |
| Live tk | Opened on host with `$DISPLAY=:0`; ran end-to-end without crash |
| Headless tk | `DISPLAY=` unset → warning + `_disabled=True` + all Protocol methods no-op; no crash |
| `ruff check` | 0 errors across `visualization/`, `steering/stage_manager.py`, `scripts/run_evaluation.py` |
| `git grep update_dash` | 0 live callers (only iter-4 comment) |

**Bundling note:** iter 3's commit (`27e224f`) accidentally pulled in pre-existing modified-state from prior tasks (Task 1's `voxposer_steering.py` split, `stage_manager.py` changes, Task 4's dwell=4 tuning). These were already correct on the branch; the commit captured them rather than introducing regressions. The user can sort hunks at PR review.

---

## Open items

- **`conf/evaluation/_task5_iter4_smoke.yaml`** — dev-only smoke YAML with all 3 viz toggles on; marked with underscore prefix. Safe to delete after PR.
- **`utils/visualize_steering.py` follow-up** — was left in iter 2 (not on the strip list); deleted in iter 3. Clean.
- **`tmp/` other subdirs** (`bbox_annotator`, `bimodal_experiment`, `p4_validation`, `visualizations`) — only `denoise_visualizer/` was on the strip list; others left untouched.
- **`scripts/run_experiment.py` F401 imports** (`Dict`, `Any`, `Observation`, `Action` unused) — 4 pre-existing, unrelated to Task 5. Flagged for a separate import-hygiene pass.
- **Task #7 extension point** — `Renderer` Protocol is the intended hook for a VLM scene-image renderer. New renderer registers via `Manager.register(...)`, implements `update_state(state)` (reads camera frames from `state["rgb"]`), and no Manager internals change.
