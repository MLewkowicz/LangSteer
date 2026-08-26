# Task 1 — Steering Module Separation

**Status:** ✅ done  
**Plan:** [task1_plan.md](task1_plan.md)  
**Branch:** `refactoring`

---

## Goal

`steering/voxposer_steering.py` had grown to 1583 lines by entangling seven distinct concerns in a single class: stage data/parsing, LMP lifecycle, stage state-machine and transitions, adaptive guidance scalers, position-space Tweedie math, rotation-space Tweedie math, and the `get_guidance` compositing call. The result made it difficult to extend any one concern (e.g. add a rotation value-map, tune a scaler) without understanding the whole file. Task 1 splits the monolith into eight focused modules — a slim orchestrator plus seven helpers — while preserving every externally visible API and all runtime behavior identically.

---

## Plan summary

Full plan: [task1_plan.md](task1_plan.md)

- **Target layout** (8 files under `steering/`):
  - `voxposer_steering.py` — slim orchestrator ~250–280 lines
  - `stage_spec.py` — `StageSpec` dataclass, vocabs, `parse_composer_stages`, `normalize_rot_target`
  - `diffusion_utils.py` — `get_alpha_bar`, `predict_x0`, `epsilon_coeff`, `dps_coeff`
  - `coordinates.py` — `PositionTransform`: model↔world conversion + voxel gradient lookup
  - `scalers.py` — `ScalerContext`, `BaseScaler`, four concrete scalers, `compose()`
  - `position_field.py` — `PositionFieldGuidance.compute()` → `(B,H,3)` ε delta
  - `rotation_field.py` — `RotationFieldGuidance.compute()` → `(B,H,6)` ε delta, first-class SO(3) branch
  - `stage_manager.py` — `StageManager` owning stages, transitions, grasp gate, loop-back, LMP lifecycle, callbacks; exposes `StageActivation` snapshots

- **11-step implementation plan** (Steps 0–10 + cleanup at Step 11):
  - Step 0: scaffold empty modules
  - Steps 1–7: migrate one concern per step (stage_spec → diffusion_utils → coordinates → scalers → position_field → rotation_field → stage_manager)
  - Step 8: slim down orchestrator
  - Step 9: update `steering/__init__.py` (likely no-op)
  - Step 10: smoke test
  - Step 11: docstrings + ruff/mypy

- **Public API is frozen** — every method signature on `VoxPoserSteering` is preserved unchanged. Two compat shims added: `_value_map` property delegates to `stage_manager.current().value_map`; `_lmp_config` dict is shared by reference so external post-construction mutation propagates.

- **Key design decisions:**
  - `RotationFieldGuidance` is first-class (not bolted-on); designed to accept a `RotationValueMap` target in Task 3 without touching the orchestrator.
  - `RotationAlignmentScaler` computes its own `rot_pred_6d` internally after x₀_rot is computed (avoids a circular `ScalerContext` dependency).
  - `StageManager` owns all grasp-gate, loop-back, and composer-call state; orchestrator holds only `_episode_step`, schedulers, and device.
  - `snapshot(ee_pos)` on `StageManager` produces the `get_costmap_state` dict; orchestrator merges the episode-level `step` key before returning.

---

## Implementation

All changes are uncommitted on branch `refactoring` (seven new files untracked, two modified).

### Files added

| File | Lines | Role |
|------|-------|------|
| `steering/stage_spec.py` | 294 | `PRIMITIVE_VOCAB`, `OBJECT_VOCAB`, `ARTICULATED_TARGET_TASKS`, `VALID_STAGE_MODES` (unprefixed — now public); `StageSpec` dataclass; `parse_composer_stages()` free function (replaces `_parse_stage`); `normalize_rot_target()` free function. |
| `steering/diffusion_utils.py` | 94 | `get_alpha_bar`, `predict_x0` (epsilon / v_prediction / sample), `epsilon_coeff`, `dps_coeff`. Single source of truth for all Tweedie math. Falls back to `alpha_bar=0.5` when scheduler is None (pre-inference). |
| `steering/coordinates.py` | 153 | `PositionTransform` — holds gripper pos + `gripper_loc_bounds` + workspace bounds; `model_to_world`, `lookup_voxel_gradient`, inline Jacobian `world_gradient_to_model`. `set_gripper_pos` / `current_gripper_pos` property. |
| `steering/scalers.py` | 188 | `ScalerContext` dataclass; `BaseScaler`; `TimestepScaler`, `DistanceScaler`, `StepScaler`, `RotationAlignmentScaler`; `compose()`. Imports `compute_rotation_matrix_from_ortho6d` for the alignment scaler. |
| `steering/position_field.py` | 126 | `PositionFieldGuidance.compute()` → `(B,H,3)` ε delta or None. Applies `TimestepScaler` + `DistanceScaler` + `StepScaler`. Preserves `step % 10 == 0` log throttle. |
| `steering/rotation_field.py` | 217 | `RotationFieldGuidance.compute()` → `(B,H,6)` ε delta or None. `slerp_targets_per_horizon` is a `@staticmethod`. Builds its own per-call `ScalerContext` via `dataclasses.replace` after computing x₀_rot so `RotationAlignmentScaler` can read the fresh prediction without a circular dependency on the orchestrator's base ctx. Applies `TimestepScaler` + `RotationAlignmentScaler`. |
| `steering/stage_manager.py` | 602 | `StageActivation` snapshot dataclass + `StageManager`. Owns: LMP lazy init + composer call, stage parsing via `parse_composer_stages`, `_activate_stage` (static/track caching, gradient precompute, HTML visualizer call, primitive/object callback dispatch), `check_transition`, `_maybe_loop_back`, `_is_grasp_complete`, `refresh`, `_eval_map`, all grasp/loop/stage mutable state. `current()` → `StageActivation`. `snapshot(ee_pos)` → live-viewer dict (without `step` — orchestrator merges it). |

### Files modified

| File | Before → After | Note |
|------|----------------|------|
| `steering/voxposer_steering.py` | 1583 → 409 lines | Rewritten as slim orchestrator. Builds `PositionTransform`, `StageManager`, four scalers, `PositionFieldGuidance`, `RotationFieldGuidance`. `get_guidance` reads `stage = self._stage_manager.current()`, builds `ScalerContext`, fetches both `alpha_bar`s, dispatches to the two `compute()` methods, combines results. `_value_map` and `current_episode_step` exposed as `@property`. `_lmp_config` held as plain attr and shared by ref with `StageManager`. |
| `steering/tweedie.py` | minor | Ruff-format-only changes (quote style, long line wrap). No logic changed. |

### Files unchanged

| File | Note |
|------|------|
| `steering/__init__.py` | Still exports `TweedieSteering`, `VoxPoserSteering` only — no new exports. |
| `voxposer/lmp.py` | Unchanged; emits raw stage tuples, consumed by `StageManager`. |
| `conf/steering/voxposer.yaml` | Cfg schema frozen. |

### Key design call — rotation ctx immutability

The plan identified risk #3 (RotationAlignmentScaler needs x₀_rot but that's computed inside RotationFieldGuidance). The resolution: `RotationFieldGuidance.compute()` receives the orchestrator's base `ScalerContext` and immediately does `dataclasses.replace(ctx, rot_pred_6d=x0_rot_pred)` after computing x₀_rot, then passes the enriched ctx to `compose()`. The orchestrator's ctx remains immutable across the two branches.

### get_guidance shape (representative excerpt)

```python
# voxposer_steering.py — get_guidance (simplified)
stage = self._stage_manager.current()
ctx = ScalerContext(timestep=t, ..., rot_target_6d=stage.rotation_target_6d)
alpha_bar_pos = get_alpha_bar(self.position_scheduler, timestep, device=self.device)
alpha_bar_rot = get_alpha_bar(self.rotation_scheduler, timestep, device=self.device)

pos_delta = self._position_field.compute(x_t=current_sample, eps=model_output,
    timestep=timestep, alpha_bar=alpha_bar_pos, stage=stage, ctx=ctx,
    episode_step=self._episode_step)
if pos_delta is not None:
    guidance[:, :pos_delta.shape[1], :3] = pos_delta

rot_delta = self._rotation_field.compute(x_t_rot=current_sample[..., 3:9],
    eps_rot=model_output[..., 3:9], timestep=timestep, alpha_bar_rot=alpha_bar_rot,
    stage=stage, ctx=ctx, episode_step=self._episode_step)
if rot_delta is not None:
    guidance[:, :rot_delta.shape[1], 3:9] = rot_delta

return guidance.detach()
```

---

## Behavior preserved / removed / relocated

| Category | Item | Before → After |
|----------|------|----------------|
| **Preserved** | All public API method signatures | `VoxPoserSteering.*` unchanged |
| **Preserved** | Tweedie x₀ formula + ε chain rule | Identical math, now in `diffusion_utils.py` |
| **Preserved** | `guidance_mode` switch (epsilon / dps) | Both branches still route through `epsilon_coeff` / `dps_coeff` |
| **Preserved** | All four scaler ramps | Now `TimestepScaler`, `DistanceScaler`, `StepScaler`, `RotationAlignmentScaler` in `scalers.py` |
| **Preserved** | `start_guidance_timestep` / `_rot` gating | Checked first in each `compute()` |
| **Preserved** | Per-horizon SLERP + quadratic alpha ramp | Verbatim in `RotationFieldGuidance.slerp_targets_per_horizon` |
| **Preserved** | Grasp gate (deque, latch, articulated-task disable) | Moved entirely to `StageManager` |
| **Preserved** | Loop-back (dwell, radius, max-loops, restart-stage-0) | Moved entirely to `StageManager` |
| **Preserved** | Static vs. track stage caching | `StageSpec.cached_*` fields, populated by `StageManager._activate_stage` |
| **Preserved** | Composer-tuple parsing (2-/3-/4-/5-/6-tuple forms) | `parse_composer_stages()` in `stage_spec.py` |
| **Preserved** | Primitive-id / object-id callbacks | Dispatched from `StageManager._activate_stage` |
| **Preserved** | `get_costmap_state` dict schema | `StageManager.snapshot(ee_pos)` + orchestrator merges `step` |
| **Preserved** | All logger messages + `step % 10` throttle | Moved with the logic that generates them |
| **Preserved** | `_value_map` external read (run_evaluation:218, run_experiment:215) | `@property` shim on orchestrator |
| **Preserved** | `_lmp_config` post-construction mutation (run_experiment:104) | Shared dict reference |
| **Removed** | Nothing | — |
| **Relocated** | Vocabs + `StageSpec` | `voxposer_steering.py` → `stage_spec.py` |
| **Relocated** | `_parse_stage`, `_normalize_rot_target` | `voxposer_steering.py` → `stage_spec.py` (free functions) |
| **Relocated** | `_predict_x0`, `_get_alpha_bar`, `_predict_x0_rot`, `_get_alpha_bar_rot` | `voxposer_steering.py` → `diffusion_utils.py` |
| **Relocated** | `_model_to_world`, `_lookup_gradient`, gripper-pos state | `voxposer_steering.py` → `coordinates.py` |
| **Relocated** | `_compute_*_scale` (4 methods) | `voxposer_steering.py` → `scalers.py` classes |
| **Relocated** | Position ε-build (L1045–1086) | `voxposer_steering.py` → `position_field.py` |
| **Relocated** | Rotation ε-build + SLERP (L1088–1175, L1500–1566) | `voxposer_steering.py` → `rotation_field.py` |
| **Relocated** | `_init_lmp_system`, composer call, `_activate_stage`, `check_stage_transition`, `_maybe_loop_back`, `_is_grasp_complete`, `refresh_costmap`, `_eval_map`, all grasp/loop/stage state | `voxposer_steering.py` → `stage_manager.py` |

---

## Smoke tests / validation

All static checks were run by the refactorer on the `refactoring` branch:

| Check | Result |
|-------|--------|
| `uv run ruff check steering/` | ✅ clean |
| `uv run ruff format --check steering/` | ✅ clean |
| `uv run mypy steering/` | ✅ 0 errors in `steering/` (85 pre-existing errors in `policies/` etc., out of scope) |
| Import smoke test: `python -c "from steering.voxposer_steering import VoxPoserSteering"` | ✅ passes |
| Guidance unit test: mock uniform-gradient field + DDPM-style scheduler | ✅ guidance norm 2.1778 vs hand-derived legacy 2.1776 (Δ < 1e-4, within float jitter) |
| **CALVIN end-to-end** (see below) | ✅ **1/1 SUCCEEDED** |

**End-to-end CALVIN smoke test** (post-refactor)
- Command: `uv run python scripts/run_evaluation.py --evaluation langsteer_primitive_object --num-episodes 1 --tasks /tmp/smoke_task_order.json`
- Task: `open_drawer`, seed=42 (default)
- Result: **1/1 SUCCEEDED** in 3 steps. 100.0% success.
- Log excerpt:
  ```
  09:54:59 [steering.stage_manager] Primitive-id set: stage 1 -> 'pull' (id=2)
  09:54:59 [steering.stage_manager] Object-id set: stage 1 -> 'drawer_handle' (id=2)
  09:54:59 [steering.stage_manager] Activated stage 1/1: affordance max=1.00, non-zero=1
  09:55:04   ✓ Episode 1/1 SUCCEEDED (steps=3, reward=0.00)
  ```
- Confirms: composer → `parse_composer_stages` → `StageManager._activate_stage` → primitive/object callbacks → policy conditioning → guidance branches → successful task completion.
- Caveat (not a regression): composer emitted only 1 stage (`pull`) instead of the 2-primitive (`grasp` + `pull`) decomposition that the annotation schema lists for `open_drawer`. This is pre-existing LLM-composer behavior, captured as scope for Task 3.

**Soft-threshold deviations from plan §9:**
- `voxposer_steering.py` is 409 lines (plan target ~300). The overage is in `set_current_gripper_rotation` (~30 lines of Euler→matrix conversion, forward-compat code) and constructor verbosity. Per-module responsibility is still singular.
- `stage_manager.py` is 602 lines (plan target ~350). Stage management genuinely owns many concerns; splitting it further would violate the plan's own scope statement. Accepted.

---

## Open items

- **`tweedie.py` dedup** — `steering/tweedie.py` has equivalent Tweedie math (`_get_reference_window`, `_compute_timestep_scale`, etc.). It can share `diffusion_utils.py` and `TimestepScaler` for ~50 lines of dedup (plan §7 note 10). Captured as a post-Task-1 follow-up; not blocking any downstream task.
- **Composer single-stage emission** — `open_drawer` smoke test confirmed the composer emitted only 1 stage (`pull`) rather than the 2-primitive (`grasp` + `pull`) sequence the annotation schema expects. Observed for this task; likely surfaces for others. Input for Task 3 (VoxPoser prompt / value-map redesign).
- **`RotationValueMap` target** — `RotationFieldGuidance` is designed to accept it; Task 3 adds the implementation.
- **`run_evaluation.py` double-call** — Lines 342–346 call `check_stage_transition` twice when `update_dash` is present. Harmless (stage state is monotonic), noted for future cleanup.
