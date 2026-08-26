# Task 3b — Loop-back fix + composer perturbation robustness + value-map redesign

**Owner:** planner → refactorer (after team-lead approval)
**Branch:** `refactoring`
**Task ID:** `#6` (status: in_progress)
**Scope:** Three concerns folded into one task with three sequenced phases. See §0 for the bundle rationale.

Differs from Task 3a: 3a was prompt-only and shipped in 5 iters across 1 day. 3b spans (a) a bounded bug fix in `steering/stage_manager.py`, (b) prompt-engineering across 4 perturbation axes, and (c) a real geometry change in `voxposer/value_map.py`. Total expected work: 4–6 iters. Same audit-first protocol as 3a; same per-iter stop/ship/rollback discipline.

---

## 0. Task-status check (no split needed)

Task #6 in the task list already covers all three concerns per its expanded description (set by team-lead at 3b kickoff):

> "1. Value-map redesign (original 3b): bbox-full affordances replacing centroid + obstacle EDT masking so gradients don't bleed into non-target AABBs.
> 2. Composer perturbation robustness: prompt must handle P1-P4 rephrasings from perturbed_language_annotations.json without ever throwing on invalid primitive ID. Leverage LLM open-world knowledge to intuit correct primitive/object across very terse to very verbose instruction variants.
> 3. Multi-stage loop-back bug: policy reaches final stage basin, idles, no success, loop-back doesn't fire. Reproduce + fix."

**No task-split is needed.** The three concerns interact (composer changes affect loop-back recovery; value-map changes interact with composer's spatial-offset emissions). Sequencing them as phases inside 3b gives clean attribution while preserving a single shippable artifact.

| Task ID | Subject | Status |
|---|---|---|
| #3 | Task 3a — VoxPoser prompts cleanup + vocab linter + dead-helper removal | ✅ completed |
| #6 | Task 3b — Value-map redesign + composer perturbation robustness + multi-stage loop-back fix + held-block resolution + infinite-loop cap + 2-stage lift composer rule | 🟡 in_progress (this plan) — scope expanded post-Phase 0 with **§3.0.5 composer/parse-query bug bundle** per user direction; team-lead approved Option A (separate phase). |
| #4 | Task 4 — Multi-stage grasp-gate verification runs | ⬜ pending — **scope TBD post-3b** (likely becomes regression-test harness for both grasp-gate and loop-back, per team-lead's Q8 answer) |
| #5 | Task 5 — Visualization cleanup | ⬜ pending |
| #7 | **Task #7 — VLM scene-image ingestion for value-map construction** | ⬜ pending — **NEW**, blocked by #6/#4/#5. Owns P4-ambiguous perturbation cases that genuinely need scene info ("Pick up the block from the drawer", "Turn on the light"). Architectural change: composer ingests an overhead camera image at inference time. Out of 3b's scope. |

Task #4 scope clarification deferred to Phase 3b close — depends on what the loop-back fix looks like and whether smoke runs subsume regression-test coverage. Task #7 absorbs the scene-grounding work that was briefly scoped into 3b's 3b-prompts phase (Q3 option (b)); see §3.2 for the revised 3b-prompts boundary.

---

## 1. Original brief excerpt (quoted verbatim for self-contained doc)

User kickoff (2026-05-18):

> "We should be iterating on the composer prompts so that they are generalizable. That is, if a query is slightly different as shown in the various perturbation axes in @perturbed_language_annotations.json we are still leveraging the open world knowledge of the language model to intuit the right action and object primitive and so that we never get into a situation where we throw an error that we do not have the right primitive ID
>
> It does not seem like multi-stage looping is working after we are in the basin of the final costmap and idling with no success.
>
> You have full permission to delete files that are redundant or unnecessary after we have made the appropriate changes."

Plus team-lead's note that the user listed *perturbations first* but list-order ≠ implementation-order: this plan sequences **loop ▸ prompts ▸ valuemap** for reasons captured in §3 below.

Hard constraints retained from 3a:
- **No silent fallback to closest-match.** `VocabValidationError` continues to hard-fail on exhaustion. Solution path is to make the *prompt* robust, not to weaken the linter.
- **No reading of `action_primitive_object_annotations.json` at inference.** Negative grep test re-runs at 3b acceptance.
- **No structural edits to `steering/stage_manager.py`** beyond the loop-back bug fix (Task 1 territory boundary). No structural edits to `policies/` (Task 2 territory).

---

## 2. Phase 0 — Audit protocol (read-only, blocking)

Three parallel audits. Run composer + value-map (read-only) concurrently with the loop-back reproduction (rollouts in background). Total wall-clock target: ~30–60 min.

**Deliverable:** `docs/refactor/task3b_baseline.json` with three top-level sections (`composer_perturbation`, `loop_back_repro`, `value_map_state`), plus per-section markdown summary in this plan's iter log.

### 2.0 Composer model switch: gpt-4o → gpt-5.4-mini (per user direction; resolved post-Phase 0)

**Original direction (3b kickoff):** user requested `gpt-5`. Refactorer probed during Phase 0 prep: the literal `gpt-5` (reasoning model) is NOT drop-in compatible with the existing LLMBackend (rejects `temperature=0`, `max_tokens`, `stop=[...]`; consumes 200–800 hidden reasoning tokens before producing output).

**Resolution:** team-lead approved `gpt-5.4-mini` after a side-coordination with the refactorer. `voxposer/lmp.py` patched for gpt-5-family parameter compatibility. Fresh cache dir: `/tmp/task3b_audit_cache_gpt5/`. Edit at `conf/steering/voxposer.yaml:149`: `llm_model: gpt-5.4-mini`.

This change applies for **ALL of 3b** (Phase 0 audit + all iters).

**Future-revisit hypothesis** (captured in iter log per team-lead): "gpt-5.4-mini baseline -3.6pp vs gpt-4o 3a iter-5. Hypothesis: affordance LMP code generation differs (§2.1 only audited composer, not affordance LMPs). Revisit if post-3b regression persists. Don't reconsider model swap until 3b.3 completes."

### 2.0a Phase 0 baseline refresh on gpt-5.4-mini (REQUIRED — no reuse of 3a iter-5)

The 3a iter-5 baseline (78/140 = 55.7%) was generated on **gpt-4o**. With the model switch to gpt-5.4-mini, that baseline is no longer comparable — gpt-5-family shifts both composer emission behavior and affordance LMP code generation. **Re-run the 28-task × 5-ep canonical eval as Phase 0's primary deliverable.** This new baseline is the comparison anchor for all 3b iters.

**Phase 0 §2.0a result (COMPLETED):**
- **73/140 = 52.1%** on gpt-5.4-mini (vs 3a iter-5 gpt-4o 78/140 = 55.7%; -3.6pp delta).
- 5 tasks improved (+7 ep), 3 tasks regressed substantially (-10 ep). Biggest swings:
  - ↑ `lift_blue_block_slider`: 2/5 → 5/5 (+3)
  - ↓ `turn_on_lightbulb`: 5/5 → 1/5 (-4)
  - ↓ `turn_off_lightbulb`: 3/5 → 0/5 (-3)
  - ↓ `unstack_block`: 4/5 → 1/5 (-3)
- Composer emissions identical between gpt-4o and gpt-5.4-mini on canonical (§2.1 verified); regression source likely affordance-LMP-side code generation differences.
- Output committed: `docs/refactor/task3b_baseline_gpt5_canonical.json` with per-task pass counts.

**Wall-clock:** ~50–60 min GPU. **Cost:** ~$1 LLM.

**The 3-pronged Phase 0 audit (§2.1–§2.3) runs against this gpt-5.4-mini baseline.** All `cache_dir` overrides for 3b iters branch from `/tmp/task3b_audit_cache_gpt5/`.

The 140-call composer perturbation audit (§2.1) is still needed — it classifies which P-axes are LLM-tractable vs scene-dependent, which directly informs Task #7's scope (VLM scene-image ingestion).

### 2.1 Composer perturbation audit (140 LLM calls, no rollouts)

For each of the 28 tasks × 5 instruction variants (canonical from `task_order.json` + P1–P4 from `perturbed_language_annotations.json`), run the composer with cache disabled and record:

| Field | Notes |
|---|---|
| `task_name` | e.g. `move_slider_left` |
| `variant` | one of `canonical`, `P1`, `P2`, `P3`, `P4` |
| `instruction` | the natural-language string fed to the composer |
| `raw_emission` | full LLM output (executed Python from `exec_safe`) |
| `parsed_stages` | list of `{mode, primitive, object, has_rot_target}` after `parse_composer_stages` |
| `repromptings_used` | count of corrective re-prompts the linter fired |
| `final_status` | one of: `valid_correct`, `valid_wrong_object`, `valid_wrong_primitive`, `invalid_after_retries`, `structurally_broken` |

**`final_status` classification rules:**
- `valid_correct`: parsed stages match the task's expected (primitive, object) per the eval-ground-truth `action_primitive_object_annotations.json`. **This file is loaded ONLY by the audit script post-emission, never by the composer at inference time.** Negative grep test (§2.4) verifies this.
- `valid_wrong_object`: parser accepted the stages, but the object is wrong for the task (e.g., `lightbulb_switch` for `turn_on_led`).
- `valid_wrong_primitive`: object correct, but primitive wrong (e.g., `push` instead of `grasp` for a lift task).
- `invalid_after_retries`: `VocabValidationError` raised after `max_repromptings=2` exhausted.
- `structurally_broken`: composer crashed or returned a non-list / wrong-tuple-length structure.

**Output:** `task3b_baseline.json::composer_perturbation` with one entry per (task, variant) = 140 rows + a summary header (counts of each `final_status` per variant axis).

**Cost:** 140 LLM calls × ~3s = ~7 min, ~$0.50 in API charges.

**Cache:** fresh `cache/voxposer_audit_3b_baseline/` per 3a's iter-isolation discipline.

**Audit script:** extend `scripts/audit_composer.py` with `--instruction-source {canonical,perturbed,all}` flag. Defaults to `canonical` (preserves 3a's audit behavior); 3b audits use `all`. Source data is the existing `perturbed_language_annotations.json` (not a refactor target).

### 2.2 Loop-back reproduction (rollouts in background)

**Goal:** capture per-step data on `_steps_in_last_stage_basin`, `dist`, and whether `_maybe_loop_back` actually fires. Plus verify that `_loop_back_enabled` is True at runtime (Q3 hypothesis (d)).

**Target tasks (3 ep each, 9 episodes total):**
- `close_drawer` — regressed 100→40% in 3a iter-5; multi-stage `(grasp, drawer_handle) → (push, drawer_handle)`; loop-back-candidate.
- `stack_block` — regressed 33→0% in 3a iter-5; multi-stage `(grasp, block) → (place, block)`; loop-back-candidate.
- `open_drawer` — passing 100% in 3a iter-5; control task to confirm loop-back doesn't fire on success cases.

**Instrumentation (TEMPORARY — added in Phase 0, removed at 3b.1 ship):**
- Add to `_maybe_loop_back`: a single INFO line per call (currently logs only on successful fire), with fields: `(stage_idx, dist, radius, basin_count, dwell_thresh, loop_count, max_loops)`.
- Add to `setup_episode`: log the value of `_loop_back_enabled` at episode start (one line per ep).
- Add to `check_transition`: count how many times the function is invoked per env step (detect Q3 (b) duplicate-fire bug).

**Per-task analysis (per team-lead's Q5 refinement — three-step classification):**

**Step 1: Does the policy ever reach last-stage basin?** Track `min(dist)` over the episode. If `min(dist) > _loop_back_radius` at all steps, classify as **`policy_never_in_basin`** — this is a value-map / policy issue, NOT a loop-back bug. Loop-back fix won't help these.

**Step 2: If policy reaches basin, does dwell counter reach `dwell_steps=15`?**
- **Counter reaches 15:** classify as **`dwell_met_no_fire`** — investigate why `_maybe_loop_back` didn't fire (max_loop_backs exhausted? `_loop_back_enabled` False? Some other guard?). Sub-causes:
  - `loop_back_disabled` — `_loop_back_enabled` was False at runtime (config-merge bug).
  - `loop_back_excluded` — `len(self._stages) < 2` excluded a multi-stage task.
  - `max_loops_exhausted` — `_loop_back_count >= _max_loop_backs` (3 by default).
- **Counter stays low:** classify as **`dwell_not_met`** and sub-classify:
  - `wiggle_reset` — basin_count peaked at some N < 15, then reset due to radius-exit; this repeated until rollout timeout (PRIMARY hypothesis).
  - `dwell_too_slow` — basin_count rose monotonically but rollout ended before reaching 15 (episode timeout before dwell).
  - `duplicate_fire` — `check_transition` called >1× per env step inflating counter (counter-intuitive: could mask wiggle by hitting 15 faster, OR could trip false positives).
- **Counter reaches 15, fires, but rollout still fails:** classify as **`loop_back_fired_no_help`** — loop-back is working but the second pass converges to the same wrong basin.

**Step 3: classify the dominant cause across all 9 episodes.** The fix shape locks in §3.1 based on this.

**Output:** `task3b_baseline.json::loop_back_repro` with one entry per episode (9 rows) + a summary classifying the dominant failure mode.

**Cost:** 9 episodes × ~120s ≈ 18 min GPU wall-clock.

### 2.3 Value-map current-state survey (read-only)

For the 7 residual problem tasks (`push_blue_block_right`, `place_in_slider`, `stack_block` + the 4 graspable-block regressions `close_drawer`, `push_into_drawer`, `push_blue_block_left`, plus `stack_block` already counted), inspect the actual affordance emission shape:

| Field | Notes |
|---|---|
| `task_name` | one of the 6 unique residuals (stack_block counted once) |
| `composer_call_sites` | list of `parse_query_obj` + `get_affordance_map` calls emitted |
| `voxel_writer` | for each stage: `set_voxel_by_radius`, `set_voxel_by_box`, or single-voxel direct set |
| `raw_affordance_voxel_count` | count of `_raw_affordance > 0` voxels for each stage (1 = centroid collapse; >1 = filled region) |
| `centroid_from_raw_position` | the world-frame position used as the stage target (from `stage_manager.py:331–340` mean-of-voxels) |

**Output:** `task3b_baseline.json::value_map_state` with one entry per task per stage + a summary of "how often does the current code path emit single-voxel affordances vs filled regions?"

**Cost:** 6 LLM calls (cache likely hits from 3a iter-5 cache; otherwise ~30s + ~$0.02) + reading 6 ValueMap objects at activation time. **Bundle with §2.1's composer audit — same LMP runtime.**

### 2.4 Negative leakage grep (regression check from 3a)

```bash
grep -rn "action_primitive_object_annotations" -- voxposer/ steering/ policies/
```

Expected: empty inside these directories. Any hit is a critical regression requiring triage before 3b continues. Re-runs at 3b acceptance.

### 2.5 Phase 0 stopping criterion

Refactorer pings planner with `task3b_baseline.json` + the Phase 0 markdown summary in `docs/refactor/task3b_log.md`. Planner reviews and confirms:
- Loop-back primary hypothesis is supported by repro data.
- Composer perturbation failure modes are clustered (e.g., "P4 ambiguity is the dominant failure" vs "P1/P2 also have problems").
- Value-map current-state confirms whether 3b.3 needs full-bbox rewrite or just centroid-shift.

**No Phase 3b.1 code edit lands before this review.**

---

## 3. Phase specs

### 3.0.5 Phase 3b.0.5 — Composer/parse-query bug bundle (NEW)

**Added post-Phase 0 per user direction + team-lead's Option-A approval.** Bundles 3 bug fixes in the composer/LMP layer. Per the bug-fix-grouping principle, these land before architectural redesigns (3b.3). Same file-locality discipline: all 3 fixes touch `voxposer/` (prompt + calvin_interface + lmp).

**Goal:** unblock `place_in_slider` from the composer-side ambiguity bug (`detect('block')` can't resolve generic vocab during place context) + add infinite-loop safety + reinforce 2-stage lift emissions so the 5 `loop_back_excluded` cases from Phase 0 §2.2 become loop-back-eligible.

**Pre-conditions:** Phase 0 complete; Q1 (model), Q2 (composer-fix shape), Q3 (bundling) resolved.

**Three changes:**

**(a) 2-stage lift normative rule** in `voxposer/prompts/calvin/composer_prompt.txt`:
- Add a normative paragraph clarifying: *"Lifts MUST emit 2 stages — stage 1 grasp at the block, stage 2 affordance ~15cm above the block. Single-stage lift emissions are incorrect; the loop-back mechanism requires multi-stage structure for failure recovery."*
- Phrasing fits the existing prompt style (no new examples needed — the iter-2 lift example already demonstrates 2-stage).
- Target: resolves the 5 `loop_back_excluded` cases identified in Phase 0 §2.2.

**(b) Held-block resolution helper** in `voxposer/calvin_interface.py`:
- New `get_held_block(robot_obs, scene_obs) -> str | None` method.
- Detection: gripper closed (`width < grasp_max_width=0.07m`) AND a block centroid within ~3cm of EE position.
- Returns the block name (`'blue_block' | 'red_block' | 'pink_block'`) or `None`.
- `detect('block')` (the generic-vocab query path) updated to call `get_held_block()` as a fallback when the generic `'block'` query is ambiguous AND the gripper is closed. Default behavior otherwise.
- ~30 lines of code, no LLM involvement, no prompt change beyond (a).

**(c) Infinite-loop cap** in `voxposer/lmp.py`:
- Audit the `parse_query_obj` retry mechanism — likely in the fuzzy-match-fail loop.
- Add `max_attempts=3` with `ObjectResolutionError` (new exception, analog of `VocabValidationError`) on exhaustion.
- Propagates to runner same as `VocabValidationError` — episode counts as failed with reason `object_resolution_exhausted`.

**Validation:**

Canary: `lift_red_block_table` + `place_in_slider` × 3 ep each (team-lead-spec'd).
- Pre-fix: `place_in_slider` hangs / silent timeout; `lift_red_block_table` emits 1-stage and gets `loop_back_excluded`.
- Post-fix: `place_in_slider` no longer hangs (passes OR hard-fails fast with `ObjectResolutionError`); `lift_red_block_table` emits 2-stage (verify via composer audit re-run).
- Spot-check: held-block helper doesn't false-positive on transport-context tasks (`push_*_block_*` where gripper is closed for transport but task is not "place").

**Stopping rule (any of):**
- **SHIP:** `place_in_slider` no longer hangs (passes ≥ 1/3 OR hard-fails fast); `lift_red_block_table` emits 2-stage; no false-positive regressions on transport tasks.
- **ROLLBACK:** held-block helper introduces false positives on transport tasks → roll back the helper, keep the composer rule + infinite-loop cap.
- **PING-PLANNER:** ambiguous result.

**Expected impact:** +5–10 pp on the 5 `loop_back_excluded` lift tasks (2-stage rule unlocks them for 3b.1 wiggle-reset fix to address); `place_in_slider` goes from 0% / hang to either ≥ 1/3 or fast-fail.

**Iter cache:** `/tmp/task3b_audit_cache_iter3b05/`.

### 3.1 Phase 3b.1 — Loop-back wiggle_reset fix

**Goal:** loop-back fires correctly when the policy is genuinely stuck near the last-stage basin without success. Bounded change in `steering/stage_manager.py` (counter logic only).

**Pre-conditions:** Phase 3b.0.5 shipped; Phase 0 §2.2 confirmed `wiggle_reset` as primary root cause (16/28 tasks; peak basin counter = 3 vs threshold 15 across 140 episodes — 0 loop-back fires anywhere).

**Locked root cause:** `_steps_in_last_stage_basin = 0` reset on radius-exit (L472–473 in `stage_manager.py`) — policy oscillates around the 0.1m basin boundary; counter never accumulates 15 consecutive in-basin steps.

**Fix-shape options (test in order; ship the first that hits the canary criteria):**

- **(i) Decay-not-reset (PRIMARY candidate per Phase 0).** Change L472–473 from `self._steps_in_last_stage_basin = 0` to `self._steps_in_last_stage_basin = max(0, self._steps_in_last_stage_basin - 1)`. Decays slowly on radius-exit instead of resetting. Cheapest fix; preserves the consecutive-dwell semantic loosely.
- **(ii) Widen `_loop_back_radius`.** Currently inherits `stage_proximity_threshold=0.1m`. Decouple to 0.15–0.20m so the policy stays in-basin even with wider oscillations. Independent knob.
- **(iii) Recent-window fraction-in-basin.** Track "N of last M steps in basin" instead of consecutive. Most general; most invasive. Reserve for if (i) + (ii) don't unlock the canary.

**Strategy:** start with (i); add (ii) if canary still doesn't show loop-back fires; layer (iii) as last resort.

**Validation:**
- Canary: `close_drawer` + `stack_block` + `push_into_drawer` × 3 ep each (team-lead-spec'd — the 3 multi-stage tasks with worst regression from 3a iter-5).
- Expected: basin_count actually reaches dwell_thresh on at least 1 task; loop-back fires at least once; final pass-rate ≥ baseline.
- Negative test: `open_drawer` (control) doesn't regress — loop-back should NOT fire on a successful rollout.

**Stopping rule (any of):**
- **SHIP:** loop-back fires correctly on at least 1 of canary set AND `open_drawer` doesn't regress.
- **ROLLBACK:** `open_drawer` regresses (loop-back firing prematurely on success cases).
- **PING-PLANNER:** (i) + (ii) + (iii) all fail to unlock the canary — surface as a deeper design question.

**Instrumentation cleanup:** at ship, REMOVE the temporary INFO logging added in Phase 0 §2.2. Keep the loop-back fire-event INFO line (load-bearing for runtime debugging) and add a per-episode peak-basin-count log line at episode end.

**Expected impact:** +5–15 pp across the 16 `wiggle_reset` multi-stage tasks. If lift is smaller, fix-shape (i) wasn't enough — escalate to (ii)/(iii) within this same phase.

**Iter cache:** `/tmp/task3b_audit_cache_iter3b1/`.

### 3.2 Phase 3b.2 — Composer perturbation robustness (P1–P3 ONLY; P4 deferred to Task #7)

**Scope boundary (revised per user direction at 3b kickoff):**
- **In scope for 3b-prompts:** P1–P3 perturbations that the LLM can disambiguate from language alone (terse rephrasings, verbose rephrasings, synonym terminology). Open-world synonym hints + cheat-sheet entries + example-matrix balance. Same shape as 3a iter 2.
- **OUT of scope for 3b-prompts (deferred to Task #7):** P4 ambiguous cases that genuinely need scene info to resolve — "Pick up the block from the drawer" (which color?), "Open it" (which object?), "Turn on the light" (LED vs lightbulb? — collides identically with `turn_on_lightbulb` P4), "Stack them." (which onto which?). These cases will hard-fail with `VocabValidationError` in 3b; **that is acceptable**. The relaxation of the user's original "never throw" constraint is: "never throw on tasks that can be disambiguated from language alone." Task #7 introduces a VLM that ingests an overhead camera image to resolve these.

**No scene-context fetch is added to the composer call path in 3b.** No `get_all_detections()` primer. No new in-context examples that demonstrate scene-state grounding. No change to the composer prompt's input shape.

**Goal:** zero `invalid_after_retries` emissions across the **P1–P3 subset** (84 = 28 × 3 entries) of the 140-call composer audit. P4 entries are catalogued for Task #7 scope but not gated by 3b acceptance.

**Pre-conditions:** Phase 3b.1 shipped; Phase 0 §2.1 composer audit data committed; gpt-5 baseline established (Phase 0 §2.0a).

#### 3.2.1 Iteration plan

Cache discipline: bump `cache_dir` to `cache/voxposer_audit_3b_iter{N}/` per iter, all branching from `/tmp/task3b_audit_cache_gpt5/`.

**Iter 1 — Open-world synonym hints in `composer_prompt.txt` (prompt-only).** Target the P1–P3 axes where the LLM has all the info needed but needs a hint to use open-world knowledge.
- Add a section near the top of `composer_prompt.txt`: "Instructions may be terse, verbose, or use synonym terminology. Use your open-world knowledge to map: 'sliding door' / 'sliding panel' / 'cabinet door' → slider; 'pull bar' / 'drawer pull' → drawer_handle; 'light switch' / 'toggle' → lightbulb_switch unless context specifies LED; 'twist' / 'rotate' / 'turn left/right' → rotate; etc."
- Add 2–3 paraphrased examples (NOT 1:1 with CALVIN tasks) demonstrating the LLM resolving terse vs verbose forms of the same query to the same stage spec.
- Re-run §2.1 audit (140 calls, fresh cache). Targets:
  - P1–P3 `valid_correct` ≥ 90% (the gated metric).
  - P1–P3 `invalid_after_retries` count → 0 (the gated metric).
  - P4 `invalid_after_retries` rate is **catalogued, not gated** — feeds Task #7 scope.

**Iter 2 (optional patch-up) — failure-cluster fixes on P1–P3.** If iter 1 leaves a specific perturbation cluster broken (e.g., specific synonym fails on a particular task), one focused fix. Capped at 1 patch iter per 3a's discipline.

**Smoke-canary after each iter:** 4 tasks × 3 ep = 12 episodes on **canonical + perturbed instructions for P1–P3 ONLY**:
- `close_drawer` canonical + P1 + P3 (verbose rephrasing).
- `turn_on_led` canonical + P1 ("push down the button to turn on the green light" — P1 has the color cue).
- `move_slider_left` canonical + P2 (P2 = decomposition variant; P4 has direction-less ambiguity, skip).
- `open_drawer` canonical (control — should stay at baseline).

**Stopping rules (any of):**
- **SHIP iter:** P1–P3 `invalid_after_retries` count = 0 AND no canary regression vs Phase 3b.1 baseline.
- **ROLLBACK iter:** prompt change introduces new failure mode on canary set OR `valid_correct` count drops on P1–P3.
- **PING-PLANNER:** iter goal unmet after 1 patch-up; team-lead reviews next-step.

**Expected impact:** +3–7 pp on the P1–P3 perturbation slice. Likely smaller signal on canonical eval (most canonical instructions already work post-3a).

**Composer prompt soft-cap:** ≤ 280 lines (was 227 after 3a; +~50 lines for the new synonym section + 2 examples). The scene-context primer growth (+30 lines) is dropped along with Q3 (b).

### 3.3 Phase 3b.3 — Value-map redesign

**Goal:** affordance for AABB-targeted stages fills the full bounding box (not centroid); EDT gradient field masks non-target object AABBs so the gradient doesn't pull through obstacles.

**Pre-conditions:** Phase 3b.0.5 + 3b.1 shipped (3b.2 SKIPPED); Phase 0 §2.3 value-map current-state data committed.

**Phase 0 §2.3 audit confirms the value-map redesign hypothesis empirically:**
- **78.6% centroid-collapse rate** across residual tasks (11/14 stages emit single-voxel affordances; 0 filled-region `set_voxel_by_box` emissions today).
- **`place_in_slider` stage 1 centroid placed INSIDE the slider cavity** at `[0.18, -0.009, 0.356]` — textbook cavity-target failure. Single voxel buried in obstacle geometry.
- **`place_in_drawer` shows the same pattern** but is currently 4/5 (80%) — the centroid extraction lands compliantly by luck on a loose-bbox cavity; not fundamentally correct.
- **`push_into_drawer` and 3 lift-from-{slider,drawer}** tasks show similar inside-cavity centroid placements per the spot-check.

This data unambiguously justifies (a) bbox-fill affordances replacing single-voxel centroid writes, (b) surface-projection for the position target on place/push/pull primitives, and (c) obstacle-aware EDT masking so gradients don't pull through non-target AABBs.

**Two changes:**

#### 3.3.1 bbox-full affordances (composer-prompt + LMP-call shape)

- Current path: composer emits `get_affordance_map('a point at the center of X')` → `set_voxel_by_radius(map, target_pos, radius=5cm)`. EDT then computes gradient toward the center.
- New path: for stages where the target is at/inside an object, emit `get_affordance_map('the bounding box of X')` → `set_voxel_by_box(map, obj.aabb)`. EDT computes gradient toward the *nearest face of the bbox* rather than the centroid.
- **Prompt-side change:** add a "When to use bbox vs radius" disambiguation block in `get_affordance_map_prompt.txt`. Bbox = "at/inside object" (e.g., place_in_drawer destination, lift centroid for graspable cube). Radius = "near a point in free space" (e.g., 15cm above the block).
- **Centroid extraction change:** `stage_manager.py:331–340` currently computes the mean of all `raw_affordance > 0` voxels as the position target. For a filled bbox, this still gives the centroid — which is the *inside* of the box. **For obstacle-containing AABBs (slider, drawer cavities), this is exactly where the policy gets stuck.** Fix: compute the position target as the **closest surface point of the bbox to the current EE** rather than the centroid, when the EE is outside the bbox; centroid when inside.

#### 3.3.2 Obstacle-aware EDT masking

- Current path: `value_map.smooth()` computes `distance_transform_edt(1 - raw_affordance)` — distance from each voxel to the nearest target voxel, ignoring all other objects.
- New path: extend `smooth()` to accept an optional `obstacle_mask: Optional[np.ndarray]` parameter. Where `obstacle_mask > 0`, set the EDT to a high penalty (infinity-equivalent). This pushes the gradient field around obstacle interiors.
- **Obstacle_mask construction:** `stage_manager._activate_stage` builds the mask from `lmp_interface.get_all_detections()` excluding the target object's AABB. Passes the mask to `ValueMap.smooth(obstacle_mask=mask)`.
- **Sharpness sweep (per the iter loop the team-lead spec'd in the task description):**
  - **Iter 1 — sharp obstacle mask** (binary mask, hard EDT cutoff at obstacle boundary).
  - **Iter 2 — padded obstacle mask** (mask AABBs expanded by ~1 voxel = ~1cm).
  - **Iter 3 — soft obstacle mask** (Gaussian-smoothed mask instead of binary).
  - **Iter 4 — full 28×5 stability eval** with the chosen mask sharpness from iters 1–3.

#### 3.3.3 Iteration plan

Cache discipline: bump `cache_dir` per iter.

**Iter 1 — bbox affordances ONLY (no obstacle mask yet).** Goal: residual 0% tasks improve; no regression on currently-passing tasks. Canary: 6 tasks × 3 ep = 18 episodes on `{push_blue_block_right, place_in_slider, stack_block, open_drawer, turn_on_led, move_slider_left}`.

**Iter 2 — sharp obstacle mask.** Goal: lift_*_drawer / lift_*_slider / place_in_drawer / place_in_slider variance drops (per the audit, these tasks have non-target cavity obstacles in the workspace). Same canary set.

**Iter 3 (if needed) — padded or soft mask.** Goal: smooth the regression on tasks where the sharp mask cut too aggressively. Padded vs soft is empirical — refactorer picks based on iter 2 failure mode.

**Iter 4 — 28×5 final stability eval.**

**Stopping rules (any of):**
- **SHIP iter:** ≥2 of `{push_blue_block_right, place_in_slider, stack_block}` improve AND no regression on currently-passing canary tasks.
- **ROLLBACK iter:** canary regression ≥2 tasks OR a residual task gets worse (e.g., 0% → -ε).
- **PING-PLANNER:** value-map sharpness sweep doesn't unblock residuals after 3 iters; consider whether the bottleneck is actually value-map (vs policy-side limits).

**Expected impact:** +5–10 pp overall, primarily on the 3 residual 0% tasks. If improvement is < +3 pp, the residuals are policy-side limited and 3b.3's value-map work has ceiling.

### 3.4 28×5 final stability eval

Same shape as 3a iter-5. Run the 28-task × 5-ep eval with all 3 phase changes integrated. Fresh `cache_dir`. Compare per-task to:
- **Phase 0 §2.0a baseline (gpt-5.4-mini, 73/140 = 52.1%)** is the apples-to-apples anchor. 3a iter-5 (gpt-4o, 78/140 = 55.7%) is informational only — not comparable due to model swap.
- Hard-rule regression check: no task with Phase 0 §2.0a baseline ≥ 3/5 drops below 1/5 in 3b final eval. (Recalibrated from 3a's "≥ 2/3 drops below 1/5" — 5-ep denominator gives cleaner mapping; 60% baseline → 20% floor = 40pp allowed degradation, slightly tighter than 3a's 47pp.)

This eval is run with **canonical instructions only** (consistent with 3a baseline). Perturbation evaluation is a separate sub-eval per §4.

---

## 4. Success metrics — DEFERRED to post-audit calibration

Per team-lead's stance on Q6 (and 3a §6.3 precedent), numerical targets are **not locked in this plan**. They'll be set after Phase 0 lands, with a separate planner→team-lead reconciliation message.

**Stretch targets the task description proposed:**
- Smoke gate: 28×5 ep ≥ 60% overall on gpt-5.4-mini (Phase 0 baseline is 52.1% — +7.9pp lift target).
- Final acceptance: ≥20/28 tasks at ≥66% on 28×3 ep.

**Planner's prior on calibration after Phase 0:**
- If Phase 0 confirms the loop-back bug is the dominant cause of 3a's 4 regressions, the smoke gate target of 60% is plausible (recovering even half of those regressions adds ~+5pp).
- The final acceptance target (20/28 ≥ 66%) is aggressive — current 3a iter-5 has 11/28 at ≥60% and 9/28 at ≥66%. Hitting 20/28 ≥ 66% requires +11 tasks to clear the threshold. Plausible only if value-map redesign unblocks the residuals AND iter-2 perturbation work tightens variance on the 40–60% bucket.

**Hard regression rule (non-negotiable):** no task with Phase 0 §2.0a gpt-5.4-mini baseline ≥ 3/5 drops below 1/5 in 3b final eval. (Recalibrated from 3a's "≥ 2/3 drops below 1/5" — see §6 for the rationale.)

**Perturbation rollout sub-eval is DROPPED from 3b.** The 84-ep perturbed sub-eval (28 × 3 × random P-axis) was scoped into 3b under Q3 (b); with that decision reversed, the perturbed rollout eval moves to **Task #7's acceptance criterion** (VLM scene-image ingestion). 3b's perturbation work is composer-only — the 140-call composer audit (§2.1) catalogues emissions, but no rollouts on perturbed instructions.

**Note on apples-to-apples:** the 28×5 final eval stays on **canonical** instructions (matching 3a baseline). The new gpt-5 28×5 canonical baseline established in Phase 0 §2.0a is the comparison anchor.

---

## 5. Risks / gotchas

1. **Loop-back fix side effects.** If the wiggle-reset fix lowers the threshold (decay-not-reset), loop-back may fire on tasks where the policy is making slow progress and would have succeeded eventually. Mitigation: `open_drawer` control + monitor `loop_back_count` per episode at smoke gate. If a task that was at 100% drops to 80% with `loop_back_count > 0`, the threshold is too loose.

2. **Duplicate-fire interaction.** If Phase 0 confirms both (a) wiggle_reset AND (b) duplicate_fire as bugs, fix (b) FIRST (it's a call-site bug in `run_evaluation.py` — 3 lines). Then re-measure (a). Possible the duplicate-fire was actually masking the wiggle issue by accidentally hitting the dwell threshold faster.

3. **Perturbation prompt bloat.** Iter 1's perturbation cheat-sheet could push `composer_prompt.txt` back over 200 lines (currently 227, already 27 over 3a's soft target). Acceptance: stay under 280 lines or justify. Audit lesson 2 from 3a applies — strip with care; example-set coverage is holistic.

4. **P4 ambiguity catalogued for Task #7.** P4 cases needing scene info are out of scope for 3b. They hard-fail in 3b; the failure pattern is catalogued in `task3b_p4_for_task7.json` to inform Task #7's VLM design. Note the `turn_on_led` P4 = `turn_on_lightbulb` P4 collision — even Task #7 may need scene *state* (which light is currently off), not just scene *image*.

5. **Value-map sharpness regressions.** Sharp obstacle mask may push the gradient too aggressively away from obstacle interiors, making the policy *unable* to reach into cavities (e.g., `place_in_drawer` requires the EE to enter the drawer AABB). Mitigation: padded or soft mask in iter 3; "target object's AABB is NEVER an obstacle" rule in `stage_manager._activate_stage`.

6. **Centroid-to-surface change in `stage_manager.py:331–340` interacts with rotate tasks.** Rotate tasks emit a track-stage on the block centroid — surface-projection would shift the position target to the block's edge, which is wrong for rotate. Mitigation: only apply surface-projection for stages with primitive ∈ {place, push, pull}; keep centroid for grasp + rotate.

7. **Composer-prompt cache invalidation.** Each iter that touches `composer_prompt.txt` invalidates the audit cache. Each iter that touches `get_affordance_map_prompt.txt` invalidates the per-stage affordance cache. Both are fine (cold cache is ~30s) but refactorer must bump `cache_dir` per iter to avoid stale-cache silent-pass.

8. **Eval flakiness.** Diffusion + steering has stochastic elements. If a per-iter eval shows a 1-task regression and re-run shows it passing, that's noise. Mitigation: at ±1 ep variance on the canary set, re-run; at >1 ep variance, accept as real.

9. **Task-list boundary discipline.** 3b touches `voxposer/`, `steering/stage_manager.py` (loop-back fix only), `voxposer/value_map.py` (smooth + EDT masking), the four prompt files, and possibly `scripts/run_evaluation.py` (call-site fix). Does NOT touch `policies/`, `core/`, training, or visualization.

10. **Process discipline (carry-over from 3a):**
   - No silent plan rewrites after team-lead approval.
   - Numerical targets set post-baseline.
   - Refactorer pings planner per-phase boundary; planner pings team-lead per Phase boundary.
   - Deletions BATCHED at 3b close (not inline mid-iter).

11. **Iter budget is soft, not hard** (team-lead's pushback response to my scope-creep flag). Full 3-concern scope retained — we're not scope-reducing. If iter N of any phase is still chasing regressions and we've burned the day on it, refactorer pings planner; planner pings team-lead for re-evaluation. Same discipline as 3a's rollback rules. Estimated wall-clock budget: ~5–7 hours across all phases + final evals.

---

## 6. Acceptance criteria (3b "done")

Refactorer marks 3b complete when all of:

**Audit deliverables:**
- ✅ `docs/refactor/task3b_baseline.json` committed with all 3 sections (composer_perturbation, loop_back_repro, value_map_state).
- ✅ `docs/refactor/task3b_log.md` committed with per-phase iter entries (mirroring 3a's `task3a_phase3a_log.md`).
- ✅ Negative leakage grep passes: `grep -rn "action_primitive_object_annotations" -- voxposer/ steering/ policies/` returns 0 hits.

**Phase 3b.0.5 (composer/parse-query bug bundle):**
- ✅ 2-stage lift normative rule added to `composer_prompt.txt`; composer audit re-run confirms 0 single-stage lift emissions.
- ✅ `get_held_block(robot_obs, scene_obs)` added to `voxposer/calvin_interface.py`; `detect('block')` updated to use it as fallback in place-context.
- ✅ `ObjectResolutionError` + `max_attempts=3` cap added to `voxposer/lmp.py` `parse_query_obj` retry path; propagates to runner.
- ✅ Canary: `place_in_slider` no longer hangs (passes ≥ 1/3 OR hard-fails fast with `ObjectResolutionError`); `lift_red_block_table` emits 2-stage.
- ✅ Held-block helper does not false-positive on transport-context tasks (`push_*_block_*`).

**Phase 3b.1 (loop-back wiggle_reset fix):**
- ✅ Phase 0 §2.2 confirmed `wiggle_reset` as primary root cause (16/28 tasks; 0 loop-back fires across 140 episodes; peak basin counter = 3 vs threshold 15).
- ✅ Fix-shape progression: (i) decay-not-reset → (ii) widen `_loop_back_radius` → (iii) recent-window fraction. Ship the first that unlocks canary.
- ✅ Loop-back fires correctly on at least 1 of `close_drawer` / `stack_block` / `push_into_drawer` (basin counter reaches dwell threshold; loop-back logs at INFO level).
- ✅ `open_drawer` control task does not regress.
- ✅ Temporary Phase 0 instrumentation logging removed; per-episode peak-basin-count log line added.

**Phase 3b.2 (composer perturbation robustness — P1–P3 only):**
- ✅ Zero `invalid_after_retries` emissions across **P1–P3 subset** (84 = 28 × 3 entries) at iter-final cache.
- ✅ Composer perturbation audit log shows P1–P3 `valid_correct` count strictly > Phase 0 baseline.
- ✅ P4 cases that genuinely need scene info are catalogued (not gated) — refactorer commits the P4 emission catalog as `task3b_p4_for_task7.json` to feed Task #7's scope spec.
- ✅ Composer prompt ≤ 280 lines (was 227 in 3a).

**Phase 3b.3 (value-map redesign):**
- ✅ `voxposer/value_map.py::smooth()` accepts `obstacle_mask` parameter (backward-compatible).
- ✅ Composer prompt + `get_affordance_map_prompt.txt` updated for bbox-vs-radius disambiguation.
- ✅ `stage_manager.py:331–340` position-target computation supports surface-projection for {place, push, pull} primitives.
- ✅ At least 2 of 3 residual 0% tasks (`push_blue_block_right`, `place_in_slider`, `stack_block`) improve over baseline.

**Final 28×5 stability eval:**
- ✅ Overall pass rate ≥ 60% on 28×5 gpt-5.4-mini (Phase 0 baseline 52.1% → +7.9pp lift target). **Hard floor:** no overall regression vs Phase 0 §2.0a baseline = 52.1% on gpt-5.4-mini.
- ✅ Hard regression rule: **no task with Phase 0 §2.0a gpt-5.4-mini baseline ≥ 3/5 drops below 1/5.** Recalibrated from 3a's "≥ 2/3 drops below 1/5" per team-lead's post-Phase-0 sign-off: 5-ep denominator gives a cleaner mapping (60% baseline → 20% floor = 40pp allowed degradation, slightly tighter than 3a's 47pp). Tasks at 5/5 or 4/5 baseline are protected; tasks at 3/5 are protected; tasks at 2/5 or 1/5 are below the gate threshold (iter goals can move them in either direction without gating).
- ✅ Specific recovery targets: `close_drawer` ≥ 3/5, `stack_block` ≥ 1/5 (currently 2/5 and 0/5 in 3a iter-5).

**Perturbation rollout sub-eval:** DROPPED from 3b acceptance. Moved to Task #7 acceptance (VLM scene-image ingestion is the architectural answer for perturbation robustness on scene-dependent cases).

**Code quality:**
- ✅ `ruff check voxposer/ steering/stage_manager.py voxposer/value_map.py` clean.
- ✅ Production smoke test passes end-to-end:
  ```bash
  uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer
  ```

**Deletion housekeeping:**
- ✅ Refactorer's iter logs list every "DELETION CANDIDATE" flagged during 3b.
- ✅ Planner reviews and approves the deletion batch BEFORE any `rm`; team-lead signs off.
- ✅ Approved deletions land in a single commit with explicit "DELETIONS:" list in commit message.

**Task #4 scope clarification:**
- ✅ At 3b close, planner pings team-lead with proposed Task #4 revised scope (grasp-gate-only / regression-tests / cancel).

---

## 7. Sign-offs — all resolved by team-lead (2026-05-18), with REVISE update

| # | Topic | Resolution |
|---|---|---|
| Q1 | Scope structure | One task with phases (no 3b/3c/3d split). |
| Q2 | Phase ordering | Loop ▸ prompts ▸ valuemap (bug-fix-first). |
| Q3 | P4 ambiguity policy | **DROPPED scene-context from 3b** (REVISE). P4 cases that need scene info defer to Task #7 (VLM scene-image ingestion). 3b-prompts handles P1–P3 only via prompt-side work. |
| Q4 | Perturbation eval scope | **REVISED to composer-only.** Audit per iter (140 calls catalogues emissions); perturbed rollout sub-eval DROPPED from 3b and moved to Task #7. |
| Q5 | Loop-back repro target | Instrument all 3 candidates (`close_drawer`, `stack_block`, `open_drawer`). Three-step classification in §2.2. |
| Q6 | Smoke gate metrics | Recalibrate post-audit. **REVISED:** gpt-5 model switch invalidates the 3a iter-5 baseline; refresh 28×5 canonical baseline on gpt-5 in Phase 0 §2.0a. |
| Q7 | Deletion process | Batched-at-close. Flag candidates inline in iter logs as `DELETION CANDIDATE:` markers. Planner reviews + team-lead approves before any `rm`. |
| Q8 | Task #4 scope | Defer to 3b close — likely (b) regression-test harness for grasp-gate + loop-back. |
| **REVISE** | Composer model | **gpt-4o → gpt-5.4-mini** (RESOLVED post-Phase 0). `gpt-5` reasoning model probed and rejected as LLMBackend-incompatible; `gpt-5.4-mini` settled with side-coordination + lmp.py compat patch. -3.6pp baseline regression captured as hypothesis; revisit if 3b doesn't recover. |
| **REVISE** | Task #7 created | **New task: VLM scene-image ingestion for value-map construction.** Blocked by #6/#4/#5. Owns P4-ambiguous perturbation cases. |
| **POST-PHASE-0** | 3b.0.5 added | **New phase 3b.0.5** (composer/parse-query bug bundle) per user direction + team-lead Option-A approval. 3 fixes: 2-stage lift normative rule + held-block resolution helper + infinite-loop cap. Lands BEFORE 3b.1. |
| **POST-PHASE-0** | 3b.2 SKIPPED | Per Phase 0 §2.1: 0 invalid_after_retries on 140 emissions; P1–P3 = canonical accuracy. No prompt-side work needed. P4 catalogued for Task #7. |
| **POST-PHASE-0** | 3b.1 fix locked | `wiggle_reset` confirmed primary root cause (16/28 tasks; 0 fires across 140 ep). Fix-shape (i) decay-not-reset locked as PRIMARY; (ii) widen radius + (iii) recent-window fraction as escalation path. |

Phase 0 starts immediately after the team-lead approves this revised plan.

---

## 8. Out of scope for Task 3b

- **VLM scene-image ingestion for value-map construction** — **deferred to Task #7** (blocked by 3b/4/5). Owns P4-ambiguous perturbation cases that genuinely need pixel-level scene info. **Boundary rule:** if `scene_obs` (proprioceptive state) has the answer, fix it in 3b; if you need pixels (visual disambiguation), defer to Task #7. Concrete split: held-block resolution (gripper state + block centroids) → 3b.0.5; "the block from the drawer" / LED-vs-lightbulb / "stack them" (visual context) → Task #7.
- **Scene-context exposure via text primer** (Q3 option (b)) — superseded by VLM approach in Task #7.
- **Perturbed rollout sub-eval** (28×3 = 84 ep) — moves to Task #7 acceptance.
- **Steering module structural edits** beyond the loop-back fix in `_maybe_loop_back` and (possibly) `_loop_back_radius` decoupling — Task 1 territory.
- **Policy refactors** — Task 2 territory.
- **Training code** — preprocessing scripts, `training/policies/diffuser_actor/`.
- **Visualization redesign** — Task 5 territory.
- **New primitives or new objects in the vocabulary** — locked by Task 1's `PRIMITIVE_VOCAB`/`OBJECT_VOCAB` and the trained model. Adding vocab would require retraining.
- **`parse_query_obj_prompt.txt` strip retry.** Deferred per 3a Audit lesson 2 (example-set coverage is holistic; needs a coverage-aware strip methodology before re-attempt).
- **3 accidentally-exposed host-side methods (`update_state`, `get_object_names`, `get_all_detections`)** in `setup_lmp`'s `dir(lmp_interface)` reflection — open item from 3a. Separate follow-up task touching `voxposer/lmp.py::setup_lmp` to use an explicit allowlist.

---

## 9. Iteration map (placeholder — refactorer fills in)

```
PHASE 0 — AUDIT + gpt-5 BASELINE REFRESH — ~90 min wall-clock — COMPLETE
  §2.0  Composer model switch: gpt-4o → gpt-5.4-mini.
        (Original plan said gpt-5; refactorer probed gpt-5 reasoning
         model — incompatible with LLMBackend. Settled on gpt-5.4-mini
         after team-lead/refactorer side-coordination; lmp.py patched
         for gpt-5-family compat.)
  §2.0a 28×5 canonical baseline on gpt-5.4-mini = 73/140 = 52.1%.
        -3.6pp vs 3a iter-5 (gpt-4o 78/140 = 55.7%). Hypothesis:
        affordance LMP code generation differs (§2.1 only audited
        composer). Revisit if post-3b regression persists.
  §2.1  Composer perturbation audit: 0 invalid_after_retries on 140
        emissions; P1–P3 = canonical accuracy; P4 = 12/28 wrong_object,
        all scene-info-dependent (Task #7 territory).
        → 3b.2 zero-iter empirically supported.
  §2.2  Loop-back reproduction (extracted from §2.0a baseline data):
        0 loop-back fires across 140 ep; peak basin counter = 3 vs
        threshold 15. 16 tasks classified `wiggle_reset` (primary);
        5 tasks `loop_back_excluded` (composer 1-stage); 7 tasks
        `policy_never_in_basin`. Fix-shape (i) decay-not-reset locked.
  §2.3  Value-map current-state survey: 78.6% centroid-collapse rate
        on residual tasks; `place_in_slider` stage-1 centroid INSIDE
        slider cavity confirmed. 3b.3 surface-projection + bbox
        unambiguously required.
  §2.4  Negative leakage grep: PASS (0 hits).

PHASE 3b.0.5 — Composer/parse-query bug bundle (NEW; 1 iter, +1 patch-up) — ~45 min
  (a) 2-stage lift normative rule → composer_prompt.txt.
  (b) get_held_block() → voxposer/calvin_interface.py.
  (c) ObjectResolutionError + max_attempts=3 → voxposer/lmp.py.
  Canary: lift_red_block_table + place_in_slider × 3 ep each.
  Spot-check: held-block helper false-positives on transport tasks.
  Ship or rollback.

PHASE 3b.1 — Loop-back wiggle_reset fix (1 iter, +escalation if needed) — ~30 min
  Fix-shape (i) decay-not-reset locked (PRIMARY).
  Escalation path: (ii) widen `_loop_back_radius` → (iii) recent-window
                   fraction if (i) doesn't unlock canary.
  Canary: close_drawer + stack_block + push_into_drawer × 3 ep each.
  open_drawer (control) must not regress.
  Remove temporary Phase 0 instrumentation; add per-episode peak-basin
  log line.
  Ship or rollback.

PHASE 3b.2 — SKIPPED (pre-approved by team-lead post-Phase 0)
  Empirical justification: P1–P3 audit shows 0 composer failures on
  gpt-5.4-mini at the iter-2 prompt; no prompt-side work needed.
  P4 cases catalog committed as task3b_p4_for_task7.json for Task #7.

PHASE 3b.3 — Value-map redesign (3-4 iters) — ~2 hrs
  Iter 1: bbox affordances (no obstacle mask) +
          surface-projection fix in stage_manager.py:331–340.
  Iter 2: sharp obstacle mask.
  Iter 3: padded or soft mask (if needed).
  Iter 4: 28×5 stability eval on gpt-5.4-mini.
  Validate via 18-ep canary per iter + full 140-ep at iter 4.

PHASE 3b CLOSE — ~1 hr (lighter — no perturbed rollout sub-eval)
  Final 28×5 canonical stability eval (apples-to-apples vs Phase 0
    §2.0a baseline = 73/140 = 52.1%). Folded into §3.3 iter 4 to avoid
    duplicate runs; that's also the "3b close" eval reference.
  P4 emission catalog committed as task3b_p4_for_task7.json
    (handoff to Task #7).
  Deletion batch review (refactorer compiles DELETION CANDIDATE list,
                       planner reviews, team-lead approves).
  Ping team-lead with Task #4 scope proposal.
  Hand to scribe for docs/refactor/task3b_*.md write-up.
```

Total estimated wall-clock: **~5 hours** (Phase 0 ~90 min COMPLETE + 3b.0.5 ~45 min + 3b.1 ~30 min + 3b.3 ~2 hrs + close ~1 hr; SKIP 3b.2 saved ~1 hr).

If 3b.0.5 surfaces a structural surprise (e.g., infinite-loop is NOT in `parse_query_obj` but elsewhere; held-block false-positives are unrecoverable), refactorer pings planner with a revised iteration map before committing to 3b.1.
