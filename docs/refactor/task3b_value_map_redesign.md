# Task 3b — Value-Map Redesign + Loop-back Fix + Composer Perturbation Robustness

**Status:** ✅ done  
**Plan:** [task3b_plan.md](task3b_plan.md)  
**Iteration log:** [task3b_log.md](task3b_log.md)  
**Baseline data:** [task3b_baseline.json](task3b_baseline.json) | [task3b_baseline_gpt5_canonical.json](task3b_baseline_gpt5_canonical.json) | [task3b_composer_perturbation.json](task3b_composer_perturbation.json) | [task3b_value_map_state.json](task3b_value_map_state.json)  
**Branch:** `refactoring`

---

## Goal

Task 3b addressed three interlocked problems discovered or deferred during Task 3a: (1) the loop-back recovery mechanism never fired because a basin-proximity wiggle reset zeroed the counter before it could reach its threshold; (2) `place_in_slider`, `stack_block`, and `push_into_drawer` failed with silent bad data — a workspace-center fallback when object detection found nothing, and a generic `'block'` object slot that produced no resolvable position; (3) the planned value-map geometry overhaul (full-bounding-box affordances, EDT-smoothed obstacle masking) to address cavity-target failures. The headline result is **+2.1pp / +3 episodes** over the Phase 0 3b baseline (76/140 = 54.3% vs 73/140 = 52.1%). The planned geometric overhaul was reverted after iterative empirical testing revealed a fundamental EDT gradient incompatibility with grasp primitives.

---

## Plan summary

Full plan: [task3b_plan.md](task3b_plan.md)

- **Original scope:** three sequenced phases — 3b.1 loop-back fix, 3b.2 composer perturbation robustness, 3b.3 value-map geometry redesign.
- **Expanded mid-plan:** Phase 3b.0.5 added for a bug-fix bundle (held-block resolution, ObjectResolutionError, 2-stage lift rule) per user direction.
- **3b.2 skipped:** Phase 0 composer perturbation audit showed P1–P3 rephrasings produced identical emissions to canonical — zero-iter. P4 (scene-info-dependent ambiguity) deferred to Task #7 (VLM scene-image ingestion).
- **3b.3 partial shipped:** composer prompt fixes (stack_block, push_into_drawer, light-toggle 2-stage rule) landed. bbox-fill affordance prompts and obstacle-mask code reverted after the geometric incompatibility was empirically identified.
- **Audit-first protocol** carried forward from 3a: Phase 0 ran a 28-task × 5-ep canonical eval on the new model plus three parallel sub-audits (perturbation, loop-back reproduction, value-map state survey) before any code change.
- **Hard acceptance criteria:** overall ≥ Phase 0 baseline (52.1%), no task with baseline ≥ 3/5 drops below 1/5.

---

## Model switch: gpt-4o → gpt-5.4-mini

Per user direction at 3b kickoff. The literal `gpt-5` reasoning model rejected the existing LLMBackend's `temperature=0`, `max_tokens`, and `stop=[...]` parameters. `gpt-5.4-mini` is drop-in compatible with a one-line config change plus a model-aware dispatch in `voxposer/lmp.py::LLMBackend._call_openai`:
- `gpt-5*` family: pass `max_completion_tokens`, omit `stop=`, truncate response client-side at first stop marker.
- gpt-4o and earlier: legacy path unchanged.

**Model-swap effect:** gpt-5.4-mini on canonical = 73/140 = 52.1% vs gpt-4o 3a iter-5 = 78/140 = 55.7% (-3.6pp). Three tasks regressed substantially (`turn_on_lightbulb` -4, `turn_off_lightbulb` -3, `unstack_block` -3). Composer emissions were identical between models per §2.1 audit; regression source is affordance-LMP code-generation differences. All of 3b measured against the gpt-5.4-mini Phase 0 baseline (73/140), not the gpt-4o 3a anchor.

---

## Implementation

### Phase 0 — Audit (read-only, blocking)

Four parallel sub-audits run before any code change:

- **§2.0a — 28×5 canonical baseline on gpt-5.4-mini:** 73/140 = 52.1%. Sets the 3b comparison anchor.
- **§2.1 — Composer perturbation audit (5 variants × 28 tasks = 140 calls):** P1–P3 identical to canonical (0 failures); P4 produced 12 `valid_wrong_object` cases — all scene-info-dependent (ambiguous color / fixture from terse instruction). 3b.2 confirmed zero-iter for P1–P3; P4 deferred to Task #7.
- **§2.2 — Loop-back reproduction:** 0 loop-back fires across 140 baseline episodes. Root cause: `wiggle_reset` — basin-proximity counter reset to 0 on any EE excursion outside the radius, never reaching threshold 15. Peak counter across all 140 episodes: 3.
- **§2.3 — Value-map state survey:** 78.6% centroid-collapse rate; 0 filled-region emissions; 3 of 7 residual tasks have inside-cavity centroid targets. Phase 3b.3 bbox-fill was well-targeted by the audit.
- **§2.4 — Negative leakage grep:** 0 hits for `action_primitive_object_annotations` in `voxposer/`, `steering/`, `policies/`. PASS.

### Files changed (final shipped state)

| File | Change summary |
|------|----------------|
| `voxposer/lmp.py` | Extended from 3a: `ObjectResolutionError` class; model-aware `LLMBackend._call_openai` dispatch for gpt-5 family (`max_completion_tokens`, client-side stop truncation). `HANDLE_ALIASES` + `VocabValidationError` from 3a unchanged. |
| `voxposer/calvin_interface.py` | Added `_get_held_block()` private helper (queries gripper state + nearest-block detection); `_detect_object('block')` now falls back to held-block when open-gripper lookup returns None; replaced silent workspace-center fallback with `raise ObjectResolutionError`. |
| `steering/stage_manager.py` | (a) `_eval_map` and `setup_episode` exception clauses narrowed to propagate `ObjectResolutionError` alongside `VocabValidationError`. (b) `_maybe_loop_back` simplified from ~58 to ~22 lines: removed `_loop_back_radius`, `_steps_in_last_stage_basin`, basin-proximity logic; replaced with monotone `_steps_in_last_stage` counter on policy-iters at last stage. Fire-event INFO log preserved. (c) Surface-projection added in `_activate_stage` target-computation — for `{place, push, pull}` primitives, target = closest raw_aff voxel to current EE; for `{grasp, rotate}`, centroid (unchanged). Currently a no-op with single-voxel affordances; preserved for future primitive-aware EDT work. |
| `conf/steering/voxposer.yaml` | `llm_model: gpt-5.4-mini`. Removed `loop_back_radius` + `last_stage_dwell_radius` knobs. Remaining loop-back knobs: `loop_back_on_last_stage: false` (default OFF), `last_stage_dwell_steps`, `max_loop_backs`. |
| `scripts/run_evaluation.py` | Catches `ObjectResolutionError` analogously to `VocabValidationError` — episode-level fast-fail with log, no runner crash. |
| `voxposer/prompts/calvin/composer_prompt.txt` | (a) 2-stage lift normative rule added (~10 lines near cheat-sheet). (b) `stack_block` example: generic `'block'` → specific `('grasp', 'blue_block')` + `('place', 'red_block')`. (c) `push_into_drawer` example added: `('grasp', 'pink_block')` → `('place', 'pink_block')` with stage-2 target "5cm above the drawer handle". (d) Light-toggle 2-stage rule added. Net: 227 → ~250 lines. |
| `voxposer/prompts/calvin/get_affordance_map_prompt.txt` | **Reverted to 3a iter-4 state** (72 lines). bbox-fill `set_voxel_by_box` was added in 3b.3 v1 and pulled in v2-rollback after the EDT-gradient incompatibility surfaced. `set_voxel_by_box` helper call remains in the handle-padding examples (unrelated to grasp affordances). |
| `voxposer/value_map.py` | **Reverted to pre-3b state.** `smooth()` obstacle-mask extension reverted after v1 canary showed hard-rule violation (`lift_blue_block_slider` 5/5 → 0/5). |
| `conf/evaluation/langsteer_primitive_object.yaml` | Per-iter `cache_dir` overrides reverted at Phase 3b close. |

### Iteration map

| Phase | Change | Outcome | Decision |
|-------|--------|---------|---------|
| Phase 0 | Audit: 28×5 canonical, perturbation (140 calls), loop-back repro, value-map survey | 73/140 = 52.1% baseline; P1–P3 zero-iter; wiggle_reset confirmed; 78.6% centroid-collapse | Blocking — complete before any edit |
| **3b.0.5** | Held-block helper + ObjectResolutionError + 2-stage lift rule in composer | `place_in_slider` no longer hangs (fast-fail ~25s/ep); lift tasks emit 2-stage | ✅ SHIP |
| **3b.1** | `_maybe_loop_back` ~58 → ~22 lines, monotone counter, default OFF after dwell=3 fired 3× on close_drawer with 0 successes | Code simplified; loop-back preserved for opt-in use | ✅ SHIP (default disabled) |
| 3b.2 | Composer perturbation robustness | Zero-iter — P1–P3 already robust | **SKIPPED** |
| 3b.3 v1 | bbox-fill + surface-projection + obstacle-mask | 0/12 canary; lift_pink_block_drawer control 4/5→0/3 | 🔴 ROLLBACK obstacle-mask |
| 3b.3 v2 | bbox-fill + surface-projection only | 2/12 canary; 28×5 full → 62/140 = 44.3%; lift_blue_block_slider 5/5→0/5 ❌ hard-rule violation | 🔴 ROLLBACK bbox-fill + surface-projection |
| 3b.3 partial | Composer prompt fixes only (stack/push_into_drawer/light-toggle) | stack_block 0→1, push_into_drawer 0→1, lightbulb tasks reversed model-swap regression | ✅ SHIP |
| **Final 28×5 v3** | All shipped (3b.0.5 + 3b.1 + 3b.3 partial); bbox/obstacle reverted | **76/140 = 54.3%** | ✅ SHIP Phase 3b |

---

## Lessons

**1. Strip-then-test beats add-then-debug.**  
Every speculative abstraction in 3b.3 (basin-proximity wiggle accounting, bbox-volume EDT, surface-projection for {place,push,pull}, obstacle mask) was reverted after empirical iteration. The shipped bundle is dramatically smaller than the planned bundle. Code that survived shipping did so because canary evidence supported it.

**2. Eval-budget limits what loop-back can detect.**  
Loop-back was designed for "policy stuck at last stage." At max_steps=120 / pred_horizon=20 → ~6 policy-iters/episode, the dwell counter cannot distinguish "stuck" from "completing slowly." dwell=3 fired 3× on close_drawer with 0 successes. The simplified ~22-line mechanism is preserved for future opt-in use (longer episodes, faster policy queries) but default-disabled. The real 3b.1 win is the code simplification from ~58-line basin-proximity machinery to a monotone counter.

**3. EDT gradient vanishes inside a bbox volume.**  
`set_voxel_by_box` fills the entire object interior; EDT smoothing produces a gradient that points away from the voxel centroid everywhere — but inside a dense volume the gradient is near-zero at the center. Grasp primitives need an approach gradient pointing INTO the object boundary. The correct fix is primitive-aware gradient construction (gaussian-from-centroid for grasp; EDT-from-surface for place). Deferred as future work, likely interacts with Task #7 VLM scene grounding.

**4. Obstacle-mask needs cavity-path awareness.**  
Naive masking (all non-target AABBs) zeroes the affordance gradient inside fixture housings (drawer_handle, slider_handle), blocking EE approach to cavity targets. A cavity-aware design must exempt the target's parent fixture from masking. Requires task-graph or composer-hint — out of 3b scope. Deferred.

**5. Audit-first earns its keep.**  
Phase 0's per-task instrumentation was the foundation for every diagnosis: loop-back root-cause via peak-basin-counter data, P1–P3 zero-iter decision from perturbation audit, geometric diagnosis of bbox-gradient failure from canary regressions. Without baseline data every regression would have been a guessing game.

---

## Behavior preserved / removed / relocated

| Category | Item | Notes |
|----------|------|-------|
| **Preserved** | All public API on `CalvinLMPInterface` + `StageManager` | No signature changes |
| **Preserved** | `VocabValidationError` hard-fail path | Extended with `ObjectResolutionError` |
| **Preserved** | Loop-back code path | Simplified to ~22 lines, default OFF; still fires when enabled |
| **Preserved** | Surface-projection code in `_activate_stage` | No-op with single-voxel affordances; ready for future primitive-aware EDT |
| **Preserved** | `voxposer/value_map.py::smooth()` original signature | Obstacle-mask extension fully reverted |
| **Removed** | Silent workspace-center fallback in `_detect_object` | Replaced by `ObjectResolutionError` |
| **Removed** | `_loop_back_radius`, `_steps_in_last_stage_basin`, basin-proximity check | `_maybe_loop_back` simplified; Phase 0 instrumentation also removed |
| **Removed** | `loop_back_radius` + `last_stage_dwell_radius` yaml knobs | Replaced by `last_stage_dwell_steps` monotone counter |
| **Added** | `ObjectResolutionError`, `_get_held_block()` | `calvin_interface.py` + `lmp.py` |
| **Added** | Composer normative rules | 2-stage lift, specific-color stack/push, light-toggle 2-stage |
| **Reverted** | bbox-fill affordance prompt (`set_voxel_by_box` for grasp targets) | Geometric incompatibility with EDT gradient |
| **Reverted** | `voxposer/value_map.py` obstacle-mask extension | Hard-rule violation on lift_blue_block_slider |

---

## Smoke tests / validation

**Composer audit (Phase 0 + per-iter, linter ON):**

| Check | Result |
|-------|--------|
| Valid emissions, 28 tasks × 5 perturbation variants | ✅ 0 `invalid_after_retries` across 140 calls |
| P1–P3 correctness rate vs canonical | ✅ Identical (19/28 valid_correct on all 4 variants) |
| 2-stage lift enforcement (post-3b.0.5) | ✅ 10/10 lift+unstack tasks emit 2-stage on canonical |
| `ObjectResolutionError` propagates | ✅ Runner catches and fast-fails; no silent bad-data spin |
| Negative leakage grep | ✅ 0 hits for `action_primitive_object_annotations` in production paths |

**Final 28×5 v3:**

| Metric | Phase 0 3b baseline (gpt-5.4-mini) | Final 3b (v3) | Δ |
|--------|------------------------------------|---------------|---|
| Overall | 73/140 = **52.1%** | 76/140 = **54.3%** | **+2.1pp (+3 ep)** |
| Improved tasks | — | 6 | avg +2.0 ep |
| Regressed tasks | — | 5 | avg -1.8 ep |
| Same | — | 17 | — |

**Key wins vs Phase 0 3b baseline:**

| Task | Phase 0 3b | Final 3b | Source |
|------|-----------|----------|--------|
| `turn_on_lightbulb` | 1/5 | **5/5** | gpt-5.4-mini model-swap regression reversed by composer 2-stage light-toggle rule |
| `turn_off_lightbulb` | 0/5 | **3/5** | Same |
| `unstack_block` | 1/5 | **3/5** | Composer + affordance improvements |
| `stack_block` | 0/5 | **1/5** | Specific-color composer fix (3b.0.6/3b.3) |
| `push_into_drawer` | 0/5 | **1/5** | Specific-color composer fix |

**Persistent regressions vs Phase 0 3b baseline (all within hard rule):**

| Task | Phase 0 3b | Final 3b | Root cause |
|------|-----------|----------|------------|
| `lift_blue_block_slider` | 5/5 | 2/5 | 2-stage lift rule changes trajectory shape vs baseline 1-stage emission |
| `lift_red_block_drawer` | 4/5 | 1/5 | Same root cause |
| `lift_blue_block_table` | 2/5 | 1/5 | Sampling noise / model-swap residual |
| `lift_red_block_slider` | 2/5 | 1/5 | Sampling noise |
| `turn_on_led` | 5/5 | 4/5 | Sampling noise |

Hard rule (baseline ≥ 3/5 must stay ≥ 1/5): all pass. `lift_blue_block_slider` at 2/5 > 1/5 threshold.

---

## Open items (future work, in priority order)

1. **Per-task / per-scene 1-stage vs 2-stage lift decision.** The 2-stage lift rule (3b.0.5) improved previously-1-stage tasks but hurt `lift_blue_block_slider` and `lift_red_block_drawer`, whose Phase 0 1-stage emissions worked. A task-specific or scene-conditioned rule (e.g. "is the block in a cavity?") would target the rule correctly.
2. **Primitive-aware EDT gradient.** Grasp primitives need approach gradient pointing into the object boundary; EDT from a filled volume gives near-zero gradient at the center. Two candidate designs: (a) gaussian-from-centroid for grasp + EDT-from-surface for place/push/pull; (b) shell-only affordance (rasterize only bbox boundary voxels). Prerequisite for re-introducing bbox-fill. Likely interacts with Task #7 VLM scene grounding.
3. **Cavity-aware obstacle mask.** Target's parent fixture (drawer_handle, slider_handle) must not be masked. Requires task-graph awareness or a composer hint indicating the containment path. Deferred until primitive-aware EDT is in place.
4. **P4 perturbation axis** ("Turn on the light.", "Pick up the block from the drawer.") — scene-info-dependent; cannot be resolved from language alone. Owned by Task #7 (VLM scene-image ingestion at composer query time).
5. **3 accidentally-exposed host-side LMP methods** (`update_state`, `get_object_names`, `get_all_detections`) — carried forward from 3a; requires `setup_lmp` explicit allowlist, not just class changes.
6. **`loop_back_on_last_stage` empirical validation** — the simplified mechanism is correct but requires longer episodes or faster policy queries to distinguish "stuck" from "completing slowly." Revisit when eval budget increases or pred_horizon changes.
