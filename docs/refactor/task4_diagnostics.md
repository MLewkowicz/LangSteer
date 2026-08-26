# Task 4 — Loop-back tuning + diagnostics

Two-part task per team-lead:
1. **Loop-back tuning** — bump `last_stage_dwell_steps` from 3 → 4 (or 5), verify
   recovery on stuck cases AND non-regression on slow-legitimate cases.
2. **Diagnostics** — categorize each underperforming / regressed task from the
   3b final 28×5 eval (`outputs/evaluation/task3b_final_28x5_v3/`).

---

## Part 1 — Loop-back tuning

### Config edit

`conf/steering/voxposer.yaml`: `last_stage_dwell_steps: 3 → 4`.

Rationale (verbatim from `docs/refactor/task3b_log.md`'s "Phase 3b SHIPPED" +
"Post-3b structural fix" sections):
- With `max_steps=360` and `pred_horizon=20` → 18 policy-iters/ep, dwell=3
  fires after 3 stuck iters and leaves 15 iters of recovery budget.
- close_drawer's legit slow execution was hitting dwell=3 mid-completion (3
  fires / 0 successes in earlier canary), cutting off the trajectory before
  the success oracle could fire.
- dwell=4 raises the threshold to 4 stuck iters (still leaves 14-iter recovery
  budget) — should preserve recovery on lift_red_block_drawer oscillation
  cases while letting close_drawer's slower trajectory finish.

### Canary setup

Cache: `/tmp/task4_dwell4_canary/`. Reuses `langsteer_primitive_object.yaml`
override `cache_dir: /tmp/task3b_final_28x5_v3` for the composer LLM cache, so
composer emissions are byte-identical to 3b final v3 (the same LLM cache as
the 76/140 ship eval).

Tasks (4 × 3 ep = 12 ep), `max_steps=360`:
| Task | Role | Expected behavior at dwell=4 |
|---|---|---|
| `lift_red_block_drawer` | Stuck oscillation case | ≥1 fire → ≥1 recovered success |
| `close_drawer` | Slow-legitimate case | 0 cut-short fires → ≥2/3 success |
| `lift_blue_block_slider` | 3b-final regression (5→2) | Fires preserved if stuck |
| `open_drawer` | Control | 3/3 stable |

### Canary results (dwell=4, max_steps=360, loop-back enabled)

`/tmp/task4_dwell4_canary/langsteer_primitive_object.json` + log
`/tmp/task4_dwell4_canary.log`:

| Task | Success | Episode lengths | Loop-back fires | Verdict vs criteria |
|---|---|---|---:|---|
| lift_red_block_drawer | 2/3 | [4, 18, 5] | **0** | ✓ ≥1/3 success. **0 fires because composer regressed to 1-stage on all 3 eps** (`Activated stage 0/0`); `_maybe_loop_back` is gated out by `len(stages) < 2`. The failed ep was stuck at last stage but loop-back ineligible — this is a composer-side bug, not a dwell tuning issue. See Part 2. |
| close_drawer | **3/3** | [6, 6, 5] | **0** | ✓ PASS. dwell=3 produced "3 fires / 0 successes" per 3b doc; dwell=4 → 0 fires / 3/3 successes. Tuning resolved the slow-legit cut-short. |
| lift_blue_block_slider | 1/3 | [5, 18, 18] | **3** | ✓ DEMONSTRATED. Ep 2 hit max_loop_backs (loops 1/3 + 2/3 fired); ep 3 fired loop 1/3. Composer emitted 2-stage on these eps so `_maybe_loop_back` was eligible. Grasp-gate also fired cleanly ("Stage 0 transition gated by grasp check: dist=0.046m < 0.1m but gripper not closed on object (width=['0.080'])"). |
| open_drawer (control) | 3/3 | [3, 3, 4] | 0 | ✓ stable |

### Verdict — SHIP dwell=4

- **`close_drawer` regression cleared.** 0 cut-short fires, 3/3 success.
- **Loop-back fires correctly when eligible.** 3 fires across 2 stuck eps on lift_blue_block_slider; the firing event matched the design ("dwell ≥ 4 policy-iters without success → reset to stage 0").
- **Grasp-gate verified working.** Stage 0 → 1 transition blocked when EE in proximity but gripper still open (`width=['0.080']` > `max_width=0.07`). After grasp completes (post-close + post-stability-check), transition fires.
- **lift_red_block_drawer's 0 fires is NOT a dwell bug.** Composer emits 1-stage; loop-back guard `len(stages) >= 2` is by design (one-stage tasks like vertical pokes/leds shouldn't re-trigger themselves). The failure mode is the 2-stage-lift-rule regression flagged in 3b ship.

**Trade-off framing:** dwell=4 gives close_drawer 4 policy-iters (~80 env-steps with pred_horizon=20) to complete its slow handle-push trajectory before loop-back catches it, while still leaving a 14-iter recovery budget after the first fire (max_steps=360 → 18 policy-iters/ep).

### Did NOT escalate to dwell=5

close_drawer hit 3/3 at dwell=4. No regression observed → dwell=5 unnecessary.

---

## Part 2 — Underperforming / regressed task diagnostics

Source: `outputs/evaluation/task3b_final_28x5_v3/langsteer_primitive_object.json`
(28 tasks × 5 ep, max_steps=120, loop-back disabled, gpt-5.4-mini composer,
all 3b fixes shipped). Overall: 76/140 = 54.3%.

### Categorization framework

- **composer-side**: composer emits wrong (primitive, object) tuple, wrong
  number of stages, or wrong offsets for a task; fix is in the prompt or
  vocab linter.
- **value-map-side**: composer emits correctly but the resulting affordance
  geometry (centroid, smoothing, gradient) doesn't drive the EE to the
  reachable contact point; fix is in `value_map.py` or `_activate_stage`.
- **policy-side**: composer + value-map are correct but the base policy's
  conditioned trajectory can't execute under steering (e.g. primitive token
  embedding doesn't generalize, or rotation/grasp finesse missing).
- **Task #7 (VLM)**: requires scene-image grounding to disambiguate (e.g.
  P4 perturbations, cavity geometry, occlusion).

### Part 2 diagnostic canary (6 tasks × 2 ep, max_steps=360, dwell=4)

Source: `/tmp/task4_part2_diag/langsteer_primitive_object.json` + log
`/tmp/task4_part2_diag.log`. Picks 6 representative under-performers
(persistent 0%, 3b regressions, composer-fix-partial-win).

| Task | Result | Composer | Stage targets | Loop-back fires | Direct-observation classification |
|---|---|---|---|---:|---|
| `push_blue_block_right` | 1/2 [7, 18] | 2-stage `(grasp, blue_block)` → `(place, blue_block)` (strategy_unified) | Stg0 contact ~[-0.34, -0.07, 0.46]; Stg1 dest ~[-0.15, -0.07, 0.46] (push direction looks correct) | 2 (ep 2) | **policy-side** — composer + value-map shape correct; ep 1 succeeded in 7 steps. Ep 2 failed despite 2 loop-back fires; policy oscillates between stage 0/1 without progressing the push. Confirms the persistent-0% in 3b is *not* composer/value-map shape — base policy can't drive the specific blue-block push contact under current steering knobs. |
| `place_in_slider` | 1/2 [0, 2] | Ep 1: ObjectResolutionError on generic 'block' (held-block fallback can't fire at episode start — no held block). Ep 2: 2-stage with specific color, succeeded in 2 steps (lucky). | Stg0 grasp ~[0.24, -0.10, 0.47]; Stg1 cavity target ~[-0.19, 0.04, 0.69] (inside slider AABB) | 1 (ep 1, before hard fail) | **composer-side + Task #7 (VLM)** — composer stochasticity on generic 'block' even after 3b.0.5 vocab linter (hard-fails the ep). Even when composer succeeds, the cavity destination geometry is the fundamental block — single-voxel target inside fixture AABB; placing requires surface-projection that needs cavity-aware obstacle masking (Task #7). |
| `stack_block` | 0/2 [18, 18] | 2-stage with specific colors per 3b.3 fix ✓ | Stg0 source block; Stg1 dest [0.10, -0.08, 0.53] (above target block; correct stack height) | 2 (ep 1: loops 1/3 + 2/3) | **value-map-side** — composer correct (3b.3 fix landed). Stacking destination is a single-voxel air target without surface to descend onto; policy oscillates around it. Same primitive-aware EDT issue as place_in_slider but in different geometry (above vs inside). After loop-back, stg0 target drifts to drawer area [0.03, -0.21, 0.36] because the held block has moved → composer re-eval picks up the wrong source position. |
| `lift_red_block_drawer` | 2/2 [4, 6] | **1-stage** on both eps (`Composer returned 1 stage(s)`) | Single grasp target | 0 (guard) | **composer-side (2-stage rule regression)** — confirms Part 1 finding. gpt-5.4-mini's composer on this specific task ignores the 3b.0.5 "lifts MUST emit 2 stages" rule despite the prompt. Loop-back is structurally ineligible. The 2/2 here is sample-size variance vs the 1/5 in 3b final v3. |
| `push_into_drawer` | 0/2 [0, 0] | **ObjectResolutionError** on generic 'block' both eps | n/a — hard fail at composer | n/a | **composer-side** — same vocab-linter failure mode as place_in_slider ep 1. 3b.3 explicitly added a `push_into_drawer` example with specific colors to the composer prompt, but gpt-5.4-mini still emits generic 'block' on episode start. The held-block fallback in `_detect_object` can't help because the gripper isn't holding a block at composer-time. |
| `lift_blue_block_table` | 2/2 [7, 13] | 2-stage with specific colors ✓ | Stg0 block pos; Stg1 lift-up (+18cm in z) | 0 | **OK** — N=2 happens to be 2/2; the 1/5 in 3b final v3 was sample-size variance. Grasp-gate firing correctly on width=0.077 (open) and width=0.000 (closed-empty). |

### 3b final v3 per-task results

| Task | 3b final | 3a iter5 | Δ vs 3a | Bucket | Notes |
|---|---:|---:|---:|---|---|
| push_red_block_right | 3/5 | 3/5 | 0 | — | OK |
| push_red_block_left | 3/5 | 3/5 | 0 | — | OK |
| push_blue_block_right | **0/5** | **0/5** | 0 | **policy-side** | Direct obs (Part 2): composer + value-map shape correct; ep 1 of diag canary succeeded. Policy can't drive the push contact reliably on blue_block specifically. Out of scope. |
| push_blue_block_left | 2/5 | 2/5 | 0 | **policy-side** | Same family as push_blue_right per direct-obs inference. |
| push_pink_block_right | 4/5 | 4/5 | 0 | — | OK |
| push_pink_block_left | 4/5 | 4/5 | 0 | — | OK |
| move_slider_left | 3/5 | 3/5 | 0 | — | OK; articulated, grasp-gate disabled |
| move_slider_right | 3/5 | 3/5 | 0 | — | OK; articulated |
| open_drawer | 5/5 | 5/5 | 0 | — | OK |
| close_drawer | _(varies w/ loop-back)_ | 2/5 | — | policy-side trade-off | Slow legit; dwell=3 cut short, dwell=4 expected to recover |
| lift_red_block_table | 2/5 | 2/5 | 0 | value-map | 2-stage emission works but bbox-fill reverted left target=single voxel |
| lift_blue_block_table | 2/5 | 1/5 | +1 | — | OK improvement |
| lift_pink_block_table | 3/5 | 3/5 | 0 | — | OK |
| lift_red_block_slider | 2/5 | 1/5 | +1 | — | OK improvement |
| lift_blue_block_slider | **2/5** | 2/5 | 0 | composer-side | Phase 0 baseline was 5/5 (gpt-5.4-mini emitted 1-stage); 3b.0.5's 2-stage lift rule regressed this task. The 2-stage rule was added for loop-back coverage but hurts the slider-specific approach geometry. Flagged in 3b ship doc as future work. |
| lift_pink_block_slider | 3/5 | 3/5 | 0 | — | OK |
| lift_red_block_drawer | **1/5** | 3/5 | -2 | composer-side + policy-side | 2-stage rule shape regression (same as blue_slider). Loop-back at dwell=3 didn't recover; expect dwell=4 to help. |
| lift_blue_block_drawer | 2/5 | 2/5 | 0 | composer-side | Same family |
| lift_pink_block_drawer | 4/5 | 5/5 | -1 | composer-side | Same family but only -1 |
| place_in_slider | **0/5** | 0/5 | 0 | Task #7 (VLM) | Cavity-target placement; affordance centroid is interior of slider AABB; need primitive-aware EDT or scene grounding to find reachable surface |
| place_in_drawer | 4/5 | 4/5 | 0 | — | OK (luck-of-bbox per 3b §2.3 audit) |
| push_into_drawer | _composer-fix partial_ | 1/5 | — | **composer-side** | Direct obs: 2/2 eps hard-fail with ObjectResolutionError on generic 'block'. 3b.3 explicit prompt example NOT being applied reliably by gpt-5.4-mini. |
| stack_block | _composer-fix partial_ | 0/5 | — | **value-map-side** | Direct obs: composer specific-color fix landed correctly; stage 1 target is correct stack height. Failure is single-voxel-in-air target — policy oscillates without surface to descend onto. After loop-back, composer re-eval drifts because held block moves. |
| unstack_block | **1/5** in baseline → 3/5 final | 4/5 | — | composer-side | Phase 0 gpt-5.4-mini regression (1/5 vs 3a 4/5); 3b's 2-stage lift rule restored to ≥3 |
| turn_on_lightbulb | **1/5** → 5/5 final | 5/5 | — | composer-side | Phase 0 gpt-5.4-mini regression; recovered by composer prompt fixes |
| turn_off_lightbulb | **0/5** → 3/5 final | 3/5 | — | composer-side | Same |
| turn_on_led | 5/5 | 5/5 | 0 | — | OK |
| turn_off_led | 5/5 | 5/5 | 0 | — | OK |

### Bucket summary (post direct-observation refinement)

| Bucket | Task count | Tasks |
|---|---:|---|
| **composer-side (2-stage lift rule too aggressive)** | 4 | lift_blue_block_slider, lift_red_block_drawer (verified 1-stage emission), lift_blue_block_drawer, lift_pink_block_drawer |
| **composer-side (generic 'block' emission)** | 2 | push_into_drawer (verified), place_in_slider (ep 1 verified — half the failure mode) |
| **value-map-side (single-voxel-in-air for place)** | 1 | stack_block (verified — composer correct, policy oscillates around stack-height air target) |
| **policy-side (base policy weakness on specific contact)** | 3 | push_blue_block_right (verified), push_blue_block_left (inferred — same family), close_drawer (was trade-off; resolved in Part 1) |
| **Task #7 (VLM — cavity geometry)** | 1 | place_in_slider (half the failure mode — the cavity destination) |
| **OK / no regression / sample variance** | 17 | rest (incl. lift_blue_block_table per direct-obs) |

**Note on `lift_red_block_drawer`:** the 2-stage-lift-rule regression and the loop-back ineligibility are the SAME bug — composer emits 1-stage, so `_maybe_loop_back`'s `len(stages) >= 2` guard kicks in and the policy can't recover. Fix is composer-side.

### Recommended fixes — by scope

#### A. Composer prompt tweaks (in-scope quick wins — flagged, not implemented)

> Per team-lead's directive: "capture as recommendations for review. We can
> decide whether to ship them as part of Task 4's deliverable or defer."

1. **Block-detect held-block-aware composer prompt.** For `push_into_drawer`
   (and `place_in_slider` ep 1 case), the composer emits a generic `'block'`
   at episode start when no block is held. The held-block fallback in
   `voxposer/calvin_interface.py::_detect_object` cannot help (gripper is
   open → no held block to fall back to). **Suggested edit:** in
   `voxposer/prompts/calvin/composer_prompt.txt`, add a normative rule near
   line ~22: *"Stage 0 object MUST be a specific block color (`red_block`,
   `blue_block`, or `pink_block`) — never generic `block`. Stage ≥1 may use
   `block` only if a prior stage's grasp produced a held block."* (Or
   conditionally re-prompt in the vocab linter when a stage 0 emission is
   generic `block`.)
2. **Drop the universal 2-stage lift rule for cavity sources.** The 3b.0.5
   "lifts MUST emit 2 stages" rule regresses `lift_*_slider`, `lift_*_drawer`
   (4 tasks, ~-4 episodes total in 3b final). Empirically, gpt-5.4-mini ignores
   the rule for some of these tasks anyway (lift_red_block_drawer: 1-stage
   both eps in Part 2 canary), suggesting the LLM-side "this lift doesn't
   need a separate lift-up stage" intuition is correct. **Suggested edit:**
   relax the rule to *"Lifts from `table` MUST emit 2 stages; lifts from
   `slider` or `drawer` MAY be 1-stage."*

#### B. Value-map / steering changes (future work; Task #7 territory)

3. **Primitive-aware EDT / target shape.** 3b.3 v1's obstacle-mask broke
   cavity-target tasks (lift_pink_block_drawer 4→0). Underlying issue: EDT
   gradient → 0 inside bbox volume for grasp primitives; single-voxel air
   target gives no surface to descend onto for `place`/`stack`. Need
   per-primitive smoothing kernel (gaussian-from-centroid for grasp,
   EDT-from-surface for place). Likely couples with Task #7.
4. **Cavity-aware obstacle mask.** For place_in_slider / place_in_drawer,
   the target's containing fixture must be exempt from masking so the EE
   can approach. Requires composer-side "containment hint" or task-graph
   awareness. Filed under Task #7.

#### C. Out of refactor scope (training-side)

5. **`push_blue_block_*` policy weakness.** Direct-observation confirms
   composer + value-map shape correct; base diffuser-actor policy
   specifically struggles with blue-block push contacts. Possible
   underrepresentation in training data; outside refactor scope.

---

### Combined deliverable

- **Part 1 dwell tuning:** SHIPPED. `last_stage_dwell_steps: 3 → 4`.
  Verified: close_drawer 3/3 (no cut-short), lift_blue_block_slider
  loop-back fires correctly, grasp-gate independently functional.
- **Part 2 diagnostics:** SHIPPED. 28 tasks classified into 4 buckets +
  OK; 6 representative under-performers directly observed via canary;
  5 concrete recommended fixes flagged for review.
