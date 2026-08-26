# Task 3b — Iteration log

Diff-against: `docs/refactor/task3b_baseline.json` (Phase 0).
Plan: `docs/refactor/task3b_plan.md`.

---

## Phase 0 — Audit + gpt-5.4-mini baseline refresh

**Date:** 2026-05-18
**Wall-clock:** ~95 min (model-swap probe + LLMBackend patch + 4 sub-audits + 28×5 baseline).
**Composer model:** `gpt-5.4-mini` (1-line config swap from gpt-4o, per user direction; literal `gpt-5` reasoning model rejected the LLMBackend's `temperature=0` / `max_tokens` / `stop=[...]`; `gpt-5-chat-latest` worked drop-in but the user corrected to `gpt-5.4-mini`).
**Cache:** `/tmp/task3b_audit_cache_gpt5_4_mini/`.

### Model-swap incident notes

`gpt-5.4-mini` accepts `temperature=0` but rejects `max_tokens` (requires `max_completion_tokens`) and rejects `stop=[...]` sequences. `voxposer/lmp.py::LLMBackend._call_openai` patched with a model-name-based dispatch:
- For `gpt-5*` family: pass `max_completion_tokens`, omit `stop=`, truncate the response client-side at the first `stop` marker.
- For gpt-4o and earlier: legacy path unchanged.

This is a behavior-preserving change (post-truncation matches what the LLM would have emitted with server-side stop). Probed on a composer-shape query before launching the audit — clean Python output.

### §2.0a — 28×5 canonical baseline on gpt-5.4-mini

**Overall: 73/140 = 52.1%** (vs 3a iter-5 on gpt-4o: 78/140 = 55.7%; **-3.6 pp regression on model swap**).

| Task | 3a iter-5 (gpt-4o) | 3b baseline (gpt-5.4-mini) | Δ |
|---|---:|---:|---:|
| push_red_block_right | 3/5 | 3/5 | 0 |
| push_red_block_left | 3/5 | 3/5 | 0 |
| push_blue_block_right | 0/5 | 0/5 | 0 |
| push_blue_block_left | 2/5 | 2/5 | 0 |
| push_pink_block_right | 4/5 | 4/5 | 0 |
| push_pink_block_left | 4/5 | 4/5 | 0 |
| move_slider_left | 3/5 | 3/5 | 0 |
| move_slider_right | 3/5 | 3/5 | 0 |
| open_drawer | 5/5 | 5/5 | 0 |
| close_drawer | 2/5 | **3/5** | **+1** |
| lift_red_block_table | 2/5 | 2/5 | 0 |
| lift_blue_block_table | 1/5 | **2/5** | **+1** |
| lift_pink_block_table | 3/5 | 3/5 | 0 |
| lift_red_block_slider | 1/5 | **2/5** | **+1** |
| lift_blue_block_slider | 2/5 | **5/5** | **+3** ✓ |
| lift_pink_block_slider | 3/5 | 3/5 | 0 |
| lift_red_block_drawer | 3/5 | **4/5** | **+1** |
| lift_blue_block_drawer | 2/5 | 2/5 | 0 |
| lift_pink_block_drawer | 5/5 | 4/5 | **-1** |
| place_in_slider | 0/5 | 0/5 | 0 |
| place_in_drawer | 4/5 | 4/5 | 0 |
| push_into_drawer | 1/5 | 0/5 | **-1** |
| stack_block | 0/5 | 0/5 | 0 |
| unstack_block | 4/5 | **1/5** | **-3** ✗ |
| turn_on_lightbulb | 5/5 | **1/5** | **-4** ✗ MAJOR |
| turn_off_lightbulb | 3/5 | **0/5** | **-3** ✗ MAJOR |
| turn_on_led | 5/5 | 5/5 | 0 |
| turn_off_led | 5/5 | 5/5 | 0 |
| **OVERALL** | **78/140 = 55.7%** | **73/140 = 52.1%** | **-3.6 pp** |

**Notable model-swap effects:**
- `turn_*_lightbulb` and `unstack_block` regressed significantly. Suggests gpt-5.4-mini's composer behavior differs from gpt-4o on these specific tasks despite §2.1 audit showing identical (primitive, object) emissions. Likely spatial-offset wording differences in the affordance queries.
- `lift_blue_block_slider` improved 2→5/5 — surprising single-task improvement.
- Pushes / sliders / drawers / led / red-rotate all maintained per-task parity.

### §2.1 — Composer perturbation audit (140 LLM calls)

Source: `docs/refactor/task3b_composer_perturbation.json`.

| Variant | valid_correct | strategy_unified¹ | valid_wrong_object | invalid_after_retries | structurally_broken |
|---|---:|---:|---:|---:|---:|
| canonical | 19/28 | 9/28 | 0 | **0** | 0 |
| P1 | 19/28 | 9/28 | 0 | **0** | 0 |
| P2 | 19/28 | 9/28 | 0 | **0** | 0 |
| P3 | 19/28 | 9/28 | 0 | **0** | 0 |
| P4 | 7/28 | 9/28 | 12 | **0** | 0 |
| **Total** | **83/140** | **45/140** | **12/140** | **0/140** | **0/140** |

¹ The `valid_wrong_primitive` classifier output was relabeled `strategy_unified` per team-lead's guidance — the 9 cases are tasks where the composer emits the 3a iter-2 grasp+place strategy that diverges from `action_primitive_object_annotations.json`'s symbolic primitive labels but works empirically (e.g., `push_*_block_*` emits `(grasp, place)` instead of GT `(push)`; 3a iter-5 showed 60-80% success). The annotation-GT diff is by design.

**Key findings:**
- **0 `invalid_after_retries` across all 140 audits** on gpt-5.4-mini. The vocab linter holds; the prompt's existing example coverage is sufficient.
- **P1, P2, P3 valid_correct rates = canonical rate (19/28).** LLM open-world knowledge already handles terse/verbose/synonym rephrasings without prompt help.
- **P4 is the only axis that degrades** — 12 `valid_wrong_object` cases, all on tasks where instruction strips disambiguating info:
  - 9× `lift_*_block_*` P4 = "Lift a block from the table." / "Lift the block in the cabinet." / "Pick up the block from the drawer." → emits generic `block` (correct given the ambiguous query — no color info).
  - 2× `turn_on_led` / `turn_off_led` P4 = "Turn on the light." / "Turn off the light." → emits `lightbulb_switch`. The SAME P4 instruction text is also used for `turn_on_lightbulb` / `turn_off_lightbulb`, where `lightbulb_switch` IS the correct answer. Composer cannot disambiguate from language alone.
  - 1× `turn_off_lightbulb` P4 = "Turn off the light." → emits `lightbulb_switch` (matches GT for this task; would be miscounted as wrong on P4 audit for `turn_off_led` whose GT is `led_button`).

**3b.2 implication:** the plan §3.2 perturbation work is **zero-iter** for P1-P3 — canonical handling is identical. P4 cases are scene-info-dependent (Task #7 territory per plan §3.2 scope boundary).

### §2.2 — Loop-back reproduction (extracted from §2.0a baseline)

Per planner's no-cost-bonus authorization, loop-back data was extracted from the §2.0a baseline log (28 tasks × 5 ep × full instrumentation) instead of running the 3-task × 3-ep targeted repro separately.

**Hard finding: 0 loop-back fires across 140 episodes.**

**Three-step classification across 28 tasks:**

| Classification | Count | Tasks |
|---|---:|---|
| `dwell_not_met_wiggle_reset` (a) | **16** | close_drawer, lift_blue_block_table/drawer, move_slider_*, open_drawer, place_in_slider, push_blue_block_*, push_into_drawer, push_pink_block_*, push_red_block_*, turn_on_lightbulb, turn_off_lightbulb |
| `policy_never_in_basin` | 7 | lift_blue_block_slider, lift_pink_block_drawer, place_in_drawer, stack_block, turn_on_led, turn_off_led, unstack_block — mostly high-success or non-multi-stage tasks where loop-back is N/A |
| `loop_back_excluded` (c) | 5 | lift_red_block_table/slider/drawer, lift_pink_block_table/slider — 1-stage composer emissions blocked at the `num_stages < 2` guard |
| `dwell_met_no_fire` (sub-case of d) | 0 | (none — basin counter never reached threshold 15 anywhere) |
| `loop_back_fired_no_help` | 0 | (loop-back never fired at all) |

**Primary loop-back bug confirmed: Q3 hypothesis (a) `wiggle_reset`.** Peak basin counter across all 140 episodes is **3** (vs threshold 15). Counter resets to 0 on radius-exit before reaching threshold.

**Per-task peak basin counters (5-ep arrays):**

| Task | Peaks | Max |
|---|---|---:|
| close_drawer | [2, 0, 0, 0, 1] | 2 |
| lift_blue_block_drawer | [3, 0, 1, 0, 0] | 3 |
| lift_blue_block_table | [0, 0, 1, 0, 0] | 1 |
| move_slider_left | [0, 0, 0, 1, 0] | 1 |
| move_slider_right | [2, 0, 0, 0, 3] | 3 |
| open_drawer | [0, 0, 0, 1, 0] | 1 |
| place_in_slider | [3, 0, 0, 0, 0] | 3 |
| push_blue_block_left | [2, 2, 1, 0, 3] | 3 |
| push_blue_block_right | [0, 1, 0, 0, 0] | 1 |
| push_into_drawer | [0, 1, 1, 0, 0] | 1 |
| push_pink_block_left | [3, 0, 0, 1, 1] | 3 |
| push_pink_block_right | [1, 0, 0, 0, 0] | 1 |
| push_red_block_left | [0, 0, 0, 0, 1] | 1 |
| push_red_block_right | [0, 1, 0, 0, 1] | 1 |
| turn_off_lightbulb | [0, 1, 3, 1, 1] | 3 |
| turn_on_lightbulb | [0, 1, 0, 0, 0] | 1 |

**Secondary bug confirmed: Q3 hypothesis (c) `loop_back_excluded` for 1-stage tasks.** 206 `guard_block` log lines across 5 lift tasks where gpt-5.4-mini composer emits a single `(grasp, X_block)` tuple. The iter 2 composer prompt has a 2-stage lift example, but gpt-5.4-mini regressed: §2.1 audit confirms 8/28 canonical tasks emit 1-stage (6 lift variants + turn_on/off_led, where 1-stage IS correct for vertical poke).

**Fix recommendations per plan §3.1:**
- Primary (a): change `_maybe_loop_back` reset to `_steps_in_last_stage_basin = max(0, x - 1)` (decay-not-reset) — addresses 16/28 tasks.
- Secondary (c): composer prompt rule "lifts MUST be 2-stage" — addresses 5/28 tasks. Team-lead approved this for Phase 3b.0.5 bundle.

### §2.3 — Value-map current-state survey

Source: `docs/refactor/task3b_value_map_state.json`.

**Centroid-collapse rate: 11/14 = 78.6%** across 7 residual tasks.

| Task | Stage 0 voxels | Stage 1 voxels | Cavity placement |
|---|---:|---:|---|
| push_blue_block_right | 1 | 1 | No |
| push_blue_block_left | 1 | 1 | No |
| stack_block | 1 | 1 | Audit-artifact (no scene state) |
| close_drawer | 1 | 1 | No (handle, not cavity) |
| push_into_drawer | _aff_err | 1 | **Yes** — destination `[0.18, -0.009, 0.356]` inside drawer AABB |
| place_in_slider | _aff_err | 1 | **Yes** — destination `[0.18, -0.009, 0.356]` inside drawer AABB (composer confused slider for drawer) |
| place_in_drawer | _aff_err | 1 | **Yes** — destination `[0.18, -0.085, 0.361]` inside drawer AABB |

**Key insight:** `place_in_drawer` at 80% success has cavity-target placement similar to `place_in_slider` at 0% — the 80% margin is loose-bbox-compliant luck (y-offset 7cm closer to drawer's front opening), not fundamentally correct placement. Validates Phase 3b.3 surface-projection hypothesis.

**0 stages emit filled-region (set_voxel_by_box) affordances.** All single-voxel collapses. Phase 3b.3 bbox-affordance rewrite is well-targeted.

### §2.4 — Negative leakage grep

```bash
$ grep -rn "action_primitive_object_annotations" --include="*.py" -- voxposer/ steering/ policies/
(no hits)
```

**PASS.** Audit script's ground-truth load is the only inference-side reference; production composer path is clean.

## Phase 3b.0.5 — Bug-fix bundle (3 fixes)

**Date:** 2026-05-18
**Cache:** `/tmp/task3b_audit_cache_iter3b05/` (fresh).
**Files touched:**
- `voxposer/prompts/calvin/composer_prompt.txt` — added LIFTS MUST EMIT 2 STAGES normative paragraph near the cheat-sheet (lines ~22-30). 227 → 236 lines.
- `voxposer/calvin_interface.py` — added `ObjectResolutionError` class + `_get_held_block()` private helper + `_detect_object('block')` held-block fallback + replaced silent workspace-center fallback with `raise ObjectResolutionError`.
- `steering/stage_manager.py` — narrowed `_eval_map`'s catch (propagate `ObjectResolutionError`); narrowed `setup_episode`'s catch to also propagate `ObjectResolutionError` alongside `VocabValidationError`.
- `conf/evaluation/langsteer_primitive_object.yaml` — cache_dir → `/tmp/task3b_audit_cache_iter3b05`.

**Fix (a) — 2-stage lift normative rule:** PASS. Composer audit on iter 3b.0.5 cache: 10/10 lift_*+unstack_block tasks now emit 2-stage on canonical (vs Phase 0: 6/10 1-stage). Only correctly-1-stage emissions remain (turn_on_led, turn_off_led — vertical pokes, not lifts).

**Fix (b) — held-block resolution helper:** FUNCTIONAL. 12 fires during the canary run, resolving to `pink_block` / `red_block` based on which block is held. No false positives on transport-context tasks within the canary scope. Unit tests pass (gripper-open → None, gripper-closed-far → None, gripper-closed-near → block name, no-state → None).

**Fix (c) — infinite-loop cap / ObjectResolutionError:** SHIPPED. Replaced silent workspace-center fallback in `_detect_object` with explicit `ObjectResolutionError`. Per audit, no explicit retry loop existed; the "infinite-loop" risk was the silent-bad-data-with-policy-spins pattern that the fallback enabled. ObjectResolutionError surfaces failures to the runner instead.

**Canary (lift_red_block_table + place_in_slider × 3 ep = 6 ep):**

| Task | Phase 0 (5 ep) | Iter 3b.0.5 (3 ep) | Pass criterion |
|---|---:|---:|---|
| lift_red_block_table | 2/5 = 40% | **1/3 = 33%** | ✓ within variance; 2-stage emission verified |
| place_in_slider | 0/5 = 0% | **0/3 = 0%**, no hangs (avg 25s/ep fast-fail) | ✓ no longer hangs |
| **Canary total** | — | **1/6 = 16.7%** | ✓ both criteria met |

**Stop signal check (planner's spec):**
- ✓ SHIP: place_in_slider no longer hangs (fast-fail, ~25s/ep vs the planner's worst-case "hangs to 360-step timeout"). Held-block helper resolving correctly.
- ✓ SHIP: lift_red_block_table emits 2-stage (audit + runtime confirmed).
- N/A ROLLBACK: held-block helper false-positive on transport tasks not observed within canary scope. Wider spot-check pending in 3b.1 canary (which covers close_drawer, stack_block, open_drawer — close_drawer is the closest analog to a transport-context task with closed gripper).
- N/A PING-PLANNER: clear pass on both ship criteria.

**Decision: SHIP iter 3b.0.5.**

**ROI vs predicted:** the place_in_slider non-hang behavior + held-block firing is the targeted-fix evidence. The episode-level pass-rate didn't change because the policy still can't execute the place trajectory once given a held-block target — that's value-map / Phase 3b.3 territory. Iter 3b.1 (wiggle_reset) targets the loop-back side; iter 3b.3 targets value-map shape.

**Lift improvement expectation:** the 2-stage lift rule should benefit lift-from-cavity tasks (lift_*_slider, lift_*_drawer). Phase 0 baseline had several at 1-3/5; will retest at 3b.1 ship.

---

### Summary for planner sign-off gate

**Loop-back primary root cause:** `wiggle_reset` (Q3 hypothesis a). Confirmed across 16/28 tasks with peak basin counter ≤ 3 (threshold 15). Fix: plan §3.1 candidate (a) — `_steps_in_last_stage_basin = max(0, x - 1)` on radius-exit.

**Loop-back secondary root cause:** `loop_back_excluded` (Q3 hypothesis c) for 5 lift tasks. Fix: composer prompt enforces 2-stage lifts (team-lead pre-approved for Phase 3b.0.5).

**Composer P-axis tractability:**
- P1-P3: 0 perturbation failures. **3b.2 is zero-iter on the language-tractable axes.**
- P4: 12/28 fail due to scene-info-dependent ambiguity. **Defers to Task #7 per plan §3.2.**

**Value-map current state:** 78.6% centroid-collapse, 0 filled-region. 3 of 7 residuals have inside-cavity centroid placements. **Phase 3b.3 bbox-affordance rewrite + surface-projection fix is well-targeted.**

**Model-swap effect:** gpt-5.4-mini regressed -3.6pp vs gpt-4o on canonical eval. Three tasks dropped significantly (turn_on/off_lightbulb, unstack_block). The composer emissions appear identical per §2.1 audit, so the regression source is policy-side variance / value-map placement diff. Worth flagging for Phase 3b.1 canary watch.

**Phase 3b.0.5 fix bundle (pending team-lead final sign-off):**
1. Composer 2-stage lift normative rule (composer prompt edit, ~10 lines).
2. Held-block resolution in `calvin_interface.py` for `place_in_slider` (per planner spec).
3. Infinite-loop cap in parse_query_obj retry path.

Wiggle_reset fix not in 3b.0.5 — Phase 3b.1 territory per the plan.

---

## Phase 3b.1 — final fix shape (2026-05-18)

**Stripped the basin-proximity check entirely** (radius, wiggle accounting, decay-not-reset, Phase 0 instrumentation). Replaced with a monotone counter on policy-iters at the last stage.

**Lesson.** Phase 0's "wiggle_reset" classification was an artifact of the over-engineered basin abstraction. The simpler model — "count policy-iters at the last stage" — was structurally correct all along. The success oracle handles the "actually succeeded" case; we don't need a proximity proxy.

**Diff summary.**
- `steering/stage_manager.py`: `_maybe_loop_back` collapsed from ~58 lines to ~22; removed `_loop_back_radius`, `_steps_in_last_stage_basin`; renamed counter to `_steps_in_last_stage`. Removed `_task3b_peak_basin_count`, `_task3b_check_transition_calls`, `_task3b_loop_back_enabled_at_setup` instrumentation and the three `[task3b_phase0/*]` INFO log sites. `check_transition` no longer passes `ee_pos` to loop-back.
- `conf/steering/voxposer.yaml`: removed `loop_back_radius` + `last_stage_dwell_radius` knobs. Now 3 loop-back knobs total: `loop_back_on_last_stage`, `last_stage_dwell_steps`, `max_loop_backs`. ~30 lines of explanatory comment trimmed.
- Public-facing fire-event INFO log ("Last-stage dwell ≥ N policy-iters — looping back") preserved.

**Canary v6 launching with:** `last_stage_dwell_steps=3`, `max_loop_backs=3`, cache `/tmp/task3b_iter3b1_cache_v6_simple/`. Same 4-task set (push_pink_block_left, turn_off_lightbulb, push_blue_block_left, open_drawer × 3 ep).

---

## Phase 3b.3 — value-map redesign + folded-in 3b.0.6 composer fix (2026-05-18)

Three coupled changes shipped as one landing per team-lead spec.

### 1. Composer prompt fix (3b.0.6 folded in)

`voxposer/prompts/calvin/composer_prompt.txt`:
- `stack_block` example: object slot was generic `'block'` ("colour is incidental"). Now requires specific colors — `('grasp', 'blue_block')` then `('place', 'red_block')`. Generic `'block'` at stage 0 had no resolvable position → ObjectResolutionError. Phase 3a's held-block fallback covered stage-1 lookups but stage 0's grasp target is OPEN-gripper → no held block to fall back to.
- Added a `push_into_drawer` example: 2-stage `('grasp', 'pink_block')` → `('place', 'pink_block')` with stage-2 target at "5cm above the drawer handle". Same specific-color requirement.

### 2. bbox-fill affordances

`voxposer/prompts/calvin/get_affordance_map_prompt.txt`:
- "center of red block" example switched from `affordance_map[x,y,z] = 1` (single voxel, 5mm cube of 1) to `set_voxel_by_box(affordance_map, red_block, value=1)`. The EDT-smoothed gradient field now radiates from the whole block volume, not a single interior voxel.
- Offset examples ("5cm above X", "10cm to the right of X", "8cm in front of X") kept as single-voxel — those queries describe a point in space, not a region.

### 3. Surface-projection target

`steering/stage_manager.py` `_activate_stage` target-computation block (lines ~330–360):
- For `spec.primitive ∈ {place, push, pull}`: target = closest raw_aff voxel to current EE (`_robot_obs[:3]`). With a bbox-filled affordance, this picks the nearest reachable face instead of the (interior) centroid.
- For `spec.primitive ∈ {grasp, rotate}`: target = centroid (unchanged). We want to drive *into* the object to grasp/rotate.
- No behavior change for single-voxel affordances (closest-to-EE == centroid == single voxel).

### 4. Obstacle-aware EDT masking

- `voxposer/value_map.py` `smooth()` extended with `obstacle_mask: Optional[np.ndarray] = None`. When set, zeros out smoothed affordance INSIDE obstacle voxels so the cost-map gradient can't pull the EE through them. Default `None` preserves prior behavior.
- `steering/stage_manager.py` adds `_build_obstacle_mask(target_object)`: rasterizes every detected object's AABB via `self._lmp_interface.set_voxel_by_box(buf, obj, value=1)` except (a) the current stage's target object, (b) the table (workspace floor), (c) the gripper (self). Falls back to `None` on any setup gap so smooth() degrades gracefully.

### Canary v6 (3b.3 bundle)

4 tasks × 3 episodes = 12 episodes. Cache `/tmp/task3b_iter3b3_cache/`. Same `--max-steps 120` as 3b.1 v6 for parity.

Tasks: `place_in_slider` (primary cavity target), `stack_block` (composer fix + cavity-place), `push_into_drawer` (composer fix + value-map), `lift_pink_block_drawer` (4/5 in 3a iter 5 — control for "don't regress existing wins").

Pass criteria per planner:
1. `place_in_slider` ≥ 1/3
2. `stack_block` ≥ 1/3
3. `lift_pink_block_drawer` ≥ 2/3 (no regression)
4. `push_into_drawer` ≥ baseline (currently 0%)

Canary running PID 203968, log `/tmp/task3b_iter3b3_canary.log`.

---

### v1 → v2: obstacle-mask pulled

**v1 (all 3 changes):** 0/12 = 0%. lift_pink_block_drawer control regressed from 4/5 → 0/3. Hypothesis: obstacle-mask was zeroing the affordance gradient inside drawer_handle (and other fixture handles), blocking EE approach to in-drawer / in-slider targets.

**v2 (obstacle-mask pulled, kept bbox-fill + surface-projection + composer-prompt fix):**
```
place_in_slider              0/3      0%
stack_block                  0/3      0%
push_into_drawer             0/3      0%   (1 ObjectResolutionError from composer stochasticity)
lift_pink_block_drawer       2/3     67%   ← control RECOVERED from 0/3
OVERALL                      2/12   16.7%
```

**Diagnosis confirmed.** Obstacle-mask is the regression source. Lift control's composer emits two `grasp` stages → surface-projection NOT applied (only fires for {place, push, pull}). Bbox-fill alone is not regressive. The 0% cavity-target residuals persist without obstacle-mask, which means bbox-fill + surface-projection ALONE doesn't unblock those tasks either — they need scene-info that the current value-map pipeline doesn't have (Task #7 / VLM territory).

**Reverted code:**
- `voxposer/value_map.py::smooth()` back to original signature `(self, obstacle_sigma: float = 3.0)`.
- `steering/stage_manager.py::_activate_stage` calls `smooth()` without `obstacle_mask=`.
- Removed `_build_obstacle_mask` helper.

**Shipped bundle (3b.3 v2):**
1. Composer prompt: stack_block emits specific colors (no generic 'block'); push_into_drawer example added.
2. get_affordance_map_prompt: "center of X" uses `set_voxel_by_box(map, obj)` instead of single voxel.
3. stage_manager: for {place, push, pull} primitives, target = closest raw_aff voxel to EE (surface point); for {grasp, rotate}, target = centroid (unchanged).

**Documented follow-up (deferred):** Obstacle-mask v2 design — target-containing fixtures (drawer_handle for in-drawer blocks, slider_handle for in-slider blocks) must NOT mask. Only mask obstacles that aren't on the path to target. Requires task-graph awareness or a "passthrough hint" from the composer.

Awaiting team-lead ship signal + go-ahead for final 28×5 eval.

---

### 3b.3 SHIP — obstacle-mask rationale + Task #7 hand-off

**Shipped:** bbox-fill prompts + surface-projection + composer-prompt fix for stack/push generic-block + 2-stage rule reaffirmed.

**Obstacle-mask follow-up (deferred, not 3b deliverable):**
> Naive obstacle-mask (all non-target AABBs masked) breaks cavity-target tasks because the target's containing fixture (drawer_handle, slider_handle) becomes "masked" and blocks the EE's approach gradient. A cavity-aware design — exempting the target's parent containment from masking — needs more thought. Filed as a follow-up; not a 3b deliverable. Bbox-fill + surface-projection + composer-prompt fix are the deployable bundle.

**Retrospective hook (post-3b):** Cavity-target tasks (place_in_slider, stack_block) didn't move with the value-map redesign. Open question: are these genuinely value-map-shape issues (needing the cavity-aware obstacle-mask), or do they require scene-image grounding (Task #7 / VLM)? Hand off to Task #7 for now.

## Phase 3b acceptance gate — final 28×5 eval

28 tasks × 5 episodes = 140 episodes. Cache `/tmp/task3b_final_28x5_cache/`. All 3b changes shipped (3b.0.5 + 3b.1 + 3b.3).

Pass criteria:
1. Overall ≥ 52.1% (Phase 0 baseline = 73/140 = 52.1%). Hard floor: no overall regression.
2. No task with new-baseline ≥ 3/5 drops below 1/5.
3. Key targets show meaningful lift (close_drawer ≥3/5; lift_*_drawer/slider ≥3/5; 5 previously-1-stage lifts now multi-stage with loop-back recovery).

---

## 3b.3 v2 minimal-rollback (2026-05-18)

**28×5 final result with bbox-fill bundle: 62/140 = 44.3%** vs Phase 0 baseline 73/140 = 52.1%. Acceptance gate failed:
- Criterion 1 (Overall ≥ 52.1%): 44.3% ❌
- Criterion 2 (no baseline-≥3/5 task drops <1/5): **lift_blue_block_slider 5/5 → 0/5** ❌ HARD-RULE VIOLATION
- Criterion 3 (close_drawer ≥ 3/5): 1/5 ❌

12 tasks regressed (7 of 9 lift tasks); 6 improved; 10 same.

**Geometric diagnosis (verbatim from team-lead):**
> 3b.3 v2 minimal-rollback: bbox-fill + surface-projection reverted because EDT gradient → 0 inside bbox volume, breaking grasp primitives (which need approach gradient pointing into object boundary, not vanishing inside). Lift_blue_block_slider 5/5 → 0/5 was the hard-rule violation that surfaced this. The fix needs primitive-aware EDT (different gradient construction for grasp vs push vs place) — out of scope for 3b. Filed as future work.

**Reverted:**
- `voxposer/prompts/calvin/get_affordance_map_prompt.txt` — "center of red block" back to single-voxel `affordance_map[x,y,z] = 1`. The `set_voxel_by_box` helper remains in the prompt for the handle-padding examples that don't drive grasp affordances.

**Kept (the actually-deployed bundle for 3b):**
- **3b.0.5:** composer/LMP bug bundle (held_block fallback + ObjectResolutionError + 2-stage lift rule in composer prompt).
- **3b.1:** simplified loop-back (~22 line `_maybe_loop_back`, `last_stage_dwell_steps: 3`, no basin-proximity check).
- **3b.3 partial:** composer prompt fixes (stack_block + push_into_drawer specific-color emissions; light-toggle 2-stage rule). These were the orthogonal 0→1 wins on stack_block / push_into_drawer in the 28×5; preserved.

**Note:** Surface-projection logic in `stage_manager._activate_stage` is preserved as a no-op (single-voxel raw_aff → closest-voxel == centroid). Keeps the option open to re-introduce bbox affordances with primitive-aware EDT later, without further code churn.

**Future work (out of 3b scope):**
- Primitive-aware gradient construction. The EDT-based smooth gradient that's correct for "approach the centroid" (single voxel) breaks for "approach the bbox surface" (volume). Needs either (a) different smoothing per primitive (gaussian-from-centroid for grasp; EDT-from-surface for place), or (b) shell-only affordance (rasterize only the bbox boundary voxels, not the interior). Likely interacts with Task #7 VLM scene grounding.

### v2-rollback canary

6 tasks × 3 ep = 18 episodes. Cache `/tmp/task3b_v2_rollback_canary/`. Pass criteria:
- `lift_blue_block_slider` ≥ 3/3 (hard-rule violation must clear)
- `close_drawer` ≥ 2/3
- `lift_red_block_drawer` ≥ 2/3
- `lift_pink_block_drawer` ≥ 2/3
- `open_drawer` 3/3 (control)
- `stack_block` ≥ 0/3 (don't lose the composer-fix win)

If canary passes → relaunch final 28×5 on cache `/tmp/task3b_final_28x5_v2/` to lock 3b's acceptance number.

---

## 3b.1 outcome refinement (2026-05-18)

Loop-back code shipped (simplified `_maybe_loop_back`, ~22 lines, monotone counter), but default flipped back to `false`. Empirically, loop-back at the current eval budget (max_steps=120, pred_horizon=20 → 6 policy-iters/episode) cannot distinguish "policy stuck" from "policy completing slowly" — dwell=3 fires during legitimate last-stage execution (close_drawer 3 fires / 0 successes in v2-rollback canary; close_drawer 3 → 1 in 28×5 final). The mechanism is preserved for future opt-in use under different eval params (longer episodes, faster policy queries) but not default-enabled. The strip-down from the original wiggle_reset machinery to a monotone counter is the real 3b.1 win — simpler code, opt-in default, no behavior surface in production until empirical evidence supports enabling it.

**Reverted:**
- `conf/steering/voxposer.yaml`: `loop_back_on_last_stage: true → false`. Comment updated explaining the structural rationale.
- `conf/evaluation/langsteer_primitive_object.yaml::steering_overrides`: removed `loop_back_on_last_stage: true` and `last_stage_dwell_steps: 3`. Only `cache_dir` remains.

**Kept:** the entire simplified loop-back code path in `stage_manager.py` (22-line `_maybe_loop_back`, `_steps_in_last_stage` counter, fire-event INFO log).

### v3 canary

Same 6-task 18-ep set as v2-rollback, now with loop-back disabled by default:
- lift_blue_block_slider (v2 0/3 → ≥2/3 expected)
- close_drawer (v2 0/3 → ≥2/3 expected)
- lift_red_block_drawer (v2 2/3 stable)
- lift_pink_block_drawer (v2 3/3 stable)
- open_drawer control (v2 3/3 stable)
- stack_block (v2 0/3 stable)

Cache `/tmp/task3b_v3_canary_loopback_off/`. If pass → final 28×5.

---

### v3 canary results + follow-up flag

v3 canary 18-ep (loop-back off): 9/18 = 50%, **0 fires across all 18 episodes** (vs 6 fires in v2-rollback). close_drawer recovered 0/3 → 1/3 (eliminating the catastrophic 3-fires-per-episode reset). open_drawer control 3/3 stable. lift_pink_block_drawer 3/3 stable. Other tasks within N=3 sample noise of v2-rollback.

**Follow-up flag (verbatim from team-lead):**
> lift_blue_block_slider canary 1/3 (vs Phase 0 baseline 5/5) on v3 — likely culprit is the 3b.0.5 2-stage lift rule changing trajectory shape from the baseline's 1-stage emission. The 2-stage rule was added to make loop-back useful on lift tasks; with loop-back now disabled by default, the rule's value is unclear. If 28×5 shows broad lift regression, consider reverting the 2-stage lift rule in a follow-up. Don't change scope mid-final-eval.

### Final 28×5 v3

Cache `/tmp/task3b_final_28x5_v3/`, full 28-task × 5-ep eval. Hard-stop policy: ship whatever 28×5 says, no further 3b iteration regardless of outcome.

---

## Phase 3b SHIPPED — Final 28×5 v3 (2026-05-18)

**76/140 = 54.3%** vs Phase 0 baseline 73/140 = 52.1% → **+3 episodes / +2.1pp**. Hard-rule clean.

| | Tasks | Avg Δ |
|---|---|---|
| Improved | 6 | +2.0 |
| Regressed | 5 | −1.8 |
| Same | 17 | 0 |

**Key wins:** turn_on_lightbulb +4, turn_off_lightbulb +3 (gpt-5.4-mini regression reversed), unstack_block +2, composer prompt fixes +2 (stack_block, push_into_drawer).

**Persistent regressions:** lift_blue_block_slider 5→2, lift_red_block_drawer 4→1. **Future work:** 2-stage lift rule is unfit for these specific tasks (their Phase 0 1-stage emissions worked). Consider task-specific or scene-conditioned 1-stage vs 2-stage decision in a follow-up.

**Final 3b bundle:**
- **3b.0.5:** vocab linter, HANDLE_ALIASES, ObjectResolutionError discipline, held-block fallback, composer 2-stage lift rule, runner-side ObjectResolutionError catch.
- **3b.1:** simplified `_maybe_loop_back` (~22 lines, monotone counter). Default off.
- **3b.3 partial:** composer prompt fixes (stack_block + push_into_drawer specific colors, light-toggle 2-stage rule). bbox-fill / obstacle-mask / surface-projection reverted as future work needing primitive-aware EDT.

**Wall-clock:** ~49 min for final eval. Total 3b runtime ~6 hours including all iterations.

Task #6 closed.

---

## Post-3b config tweak: loop-back re-enabled per user direction (2026-05-18)

User pushback on the loop-back-disabled state: observed lift_red_block_drawer oscillating in basin without recovery (14+ steps in final-stage basin, no reset). Re-enabled `loop_back_on_last_stage: true` in `conf/evaluation/langsteer_primitive_object.yaml::steering_overrides`. Base config default remains `false` for opt-out safety; eval YAML opts in.

**Trade-off acknowledged:**
- Without loop-back (3b final v3): close_drawer recovered to 3/5 (matched baseline), some lift tasks oscillated without recovery (lift_blue_block_slider 2/5, lift_red_block_drawer 1/5).
- With loop-back re-enabled: close_drawer regression returns (~1/5 in earlier canary, 3 fires / 0 successes pattern), but lift oscillation should recover.

User-directed decision. Verified mechanism wires cleanly: 1-ep canary on lift_blue_block_slider produced expected fire log ("Last-stage dwell ≥ 3 policy-iters without success — looping back to stage 0 (loop 1/3)").

**Future work:** per-primitive or per-task dwell threshold could resolve the trade-off — longer dwell for `push`/`pull` primitives (where the policy needs time to complete the contact motion), shorter for `grasp`/`lift` (where stuck = stuck). This would let close_drawer keep its 3/5 baseline while still recovering oscillating lifts.

Task #6 status unchanged — this is a post-ship config tweak documented for traceability, not a re-opening of the work.

---

## Post-3b structural fix: max_steps=360 + loop-back enabled (2026-05-18)

User pushback resolved via structural fix: increased --max-steps from 120 → 360 (default in `conf/env/calvin.yaml`) AND re-enabled `loop_back_on_last_stage: true` in eval YAML override. The 6-policy-iter budget at --max-steps=120 was the underlying problem — with 18 policy-iters, dwell=3 fires only on genuinely stuck cases (3 iters of no progress at last stage), preserving close_drawer's legitimate slow execution while recovering oscillation cases like lift_red_block_drawer.

**Verification (2 eps lift_blue_block_slider with --max-steps=360, loop-back enabled):**
- Max steps: 360 ✓ (18 policy-iter budget per episode)
- 2 fires across 2 episodes — both stuck cases triggered at iter ~6 (5 iters reaching stage 1 + 3 iters dwell)
- Recovery budget: 12 iters available post-fire (out of 18)
- Episodes still failed because lift_blue_block_slider has the orthogonal 2-stage-lift-rule regression, but the loop-back mechanism + recovery budget worked exactly as designed

**Future eval runs:** do not override --max-steps below the 360 default for production evals. The 28×5 wall-clock cost (~150 min vs ~50 min at 120) is one-time and the headroom is structurally necessary.

Task #6 status unchanged.

---
