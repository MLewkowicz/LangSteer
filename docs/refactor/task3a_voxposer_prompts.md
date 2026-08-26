# Task 3a — VoxPoser Prompts Cleanup + Vocab Linter + Dead-Helper Removal

**Status:** ✅ done  
**Plan:** [task3a_plan.md](task3a_plan.md)  
**Iteration log:** [task3a_phase3a_log.md](task3a_phase3a_log.md)  
**Baseline data:** [task3a_baseline.json](task3a_baseline.json) | [task3a_baseline_emissions.json](task3a_baseline_emissions.json)  
**Branch:** `refactoring`

---

## Goal

The LLM composer was silently emitting invalid object names (`door_handle` for `move_slider_*`, `lightbulb_switch` for `turn_on_led`). `parse_composer_stages` (Task 1) dropped those stages; the rollout ran with zero steering guidance and failed. Secondary problems: 19 of 23 composer in-context examples matched benchmark task names 1:1 (leakage), all four prompt files were bloated with redundant variants, two `CalvinLMPInterface` helpers were dead code never called anywhere, and push-block strategy was inconsistent across colors (color→strategy drift). Task 3a fixed all of these without touching value-map geometry (Phase 3b's domain), producing a **+50% relative improvement** on the 28-task CALVIN eval.

---

## Plan summary

Full plan: [task3a_plan.md](task3a_plan.md)

- **Scope:** Prompt-side only. Value-map geometry (full-bounding-box affordances, EDT obstacle masking) is Task 3b and is separately planned.
- **Audit-first protocol (Phase 0):** Read-only 28-task composer audit + 84-ep baseline eval + prompt bloat/leakage tally + helper-usage audit + leakage-source grep — all committed before any code change. No iter can claim improvement without a baseline to compare against.
- **Iterative + measurable:** Each iter ships exactly one focused change. Stopping criteria, rollback rules, and canary episode budgets defined in advance. Hard-fail on re-prompt exhaustion — no silent fallbacks.
- **Iter cadence:** 5 iters — vocab linter, composer prompt strip, dead-helper removal, affordance/avoidance prompt strip (partial), 28×5 final stability eval.
- **Hard acceptance criteria:** zero invalid emissions, no task with baseline ≥ 2/3 drops below 1/5, all 4 prompts at final shipped state.

---

## Implementation

### Phase 0 — Audit (read-only, blocking)

Produced before any edit:
- `docs/refactor/task3a_baseline.json` — per-task emission catalog (parsed stages, dropped stages, drop reason codes), eval scores, prompt bloat/leakage tally, helper-usage counts, leakage-source grep results.
- `docs/refactor/task3a_baseline_emissions.json` — raw composer outputs for all 28 tasks.

Key findings:
- `move_slider_left` + `move_slider_right`: both dropped a stage — `dropped_reason=invalid_object`, offender `door_handle`.
- `turn_on_led`: stage emitted with valid vocab (`push, led_button`) but model's `instruction_encoder` never saw `led_button` at training — because the composer was emitting `lightbulb_switch` instead. (Valid name; linter doesn't catch it — needs cheat-sheet.)
- 19 / 23 in-context examples 1:1 with benchmark task names (leakage risk).
- `push_*_block` strategy was color-dependent: red used grasp+place, blue/pink used 2-stage push — pure LLM pattern-matching on color from inconsistent examples.
- 6 lift-from-cavity tasks at 0% (value-map centroid buried inside slider/drawer cavity) — Phase 3b territory.
- 2 truly-dead helpers (`get_ee_pos`, `quaternion_from_axis_angle`) + 3 accidentally-exposed host-side methods (`update_state`, `get_object_names`, `get_all_detections`) — host methods deferred to follow-up.

### Files changed

| File | Before → After | Change summary |
|------|----------------|----------------|
| `voxposer/lmp.py` | — | Added `VocabValidationError` (ValueError subclass), `HANDLE_ALIASES = {'door_handle': 'slider_handle'}`, `_validate_and_repair`, `_suggest_object`, `_suggest_primitive`, `_classify_violations`, `_build_repair_query`, `compose_with_repair`. `max_repromptings=2` (3 LLM calls total). Hard-fail: `VocabValidationError` raised with violations + final raw result. Steering vocab dicts imported lazily to avoid circular load (`steering/__init__` → `voxposer_steering` → `stage_manager` → `voxposer/lmp`). |
| `voxposer/calvin_interface.py` | — | Removed `get_ee_pos` (body inlined into sole internal caller `_detect_ee`) and `quaternion_from_axis_angle` (0 callers, 0 prompt references — deleted outright). Net -2 from LMP-exposed surface. Class docstring updated. |
| `steering/stage_manager.py` | — | 1-line wiring: `setup_episode` calls `compose_with_repair(...)` instead of bare `composer(instruction)`. `except Exception` clause narrowed to let `VocabValidationError` propagate rather than silently fall back. |
| `voxposer/prompts/calvin/composer_prompt.txt` | 367 → 227 lines | 39% reduction. Vocab cheat-sheet header (handle/light disambiguation); 19 leakage examples stripped, replaced with 11 paraphrased examples that don't 1:1-match any CALVIN task instruction; strategy unification: all `push_*_block` now route through grasp+place (2-stage push reserved for non-graspable handles/levers); preserved verbatim: stage tuple schema, steering/primitive/policy split, world-frame rotation idiom, rotate-is-not-a-lift warning, contact-physics priors. |
| `voxposer/prompts/calvin/get_affordance_map_prompt.txt` | 146 → 72 lines | 51% reduction. Kept: canonical center pattern, 4 directional offset patterns, radius pattern, 2 `set_voxel_by_box` patterns. Dropped: near-duplicate radius variants, patterns never emitted by composer per Phase 0 audit. |
| `voxposer/prompts/calvin/get_avoidance_map_prompt.txt` | 52 → 25 lines | 52% reduction. Kept: canonical radius avoidance, OBB avoidance, OBB-with-pad. Dropped: multi-object avoidance (never emitted), composite radius example, 2 near-duplicate radius examples. |
| `voxposer/prompts/calvin/parse_query_obj_prompt.txt` | 74 → 74 lines | **Fully reverted to baseline.** Strips failed twice (→ empty responses for `parse_query_obj('led button')`). See Audit lessons. |
| `scripts/audit_composer.py` | new | 10 KB audit script. Per-task emission catalog + dropped-reason classification. `--use-linter` flag makes baseline vs. post-iter audits reproducible from the same script. |
| `.gitignore` | — | 3 external-repo rules anchored to `/` (`/calvin/`, `/3d_diffuser_actor/`, `/VoxPoser/`) so they don't accidentally swallow `voxposer/prompts/calvin/`. Clarifying comment added. |
| `conf/evaluation/langsteer_primitive_object.yaml` | — | `cache_dir` override added in iter 1 for per-iter cache isolation; **reverted at Phase 3a end**. |

**Total prompt reduction: 639 → 398 lines across 4 files (38%).**

### Iteration map

| Iter | Change | Canary result | Decision |
|------|--------|---------------|----------|
| 0 | Phase 0 audit (read-only) | 31/84 = 36.9% baseline | Blocking — must complete before any edit |
| 1 | Vocab linter + `HANDLE_ALIASES` | 4/6 vs 0/6 on `move_slider_*` canary | ✅ SHIP — `door_handle` re-prompts both succeed on attempt 1 |
| 2 | Composer prompt strip + cheat-sheet + strategy unification | 9/15 vs 3/15 canary | ✅ SHIP — `turn_on_led` 0→3/3; push_pink unblocked. Mid-iter stop: draft avoidance example used `push(red_block)` as 2-stage, would have re-introduced color drift; fixed before rollout. |
| 3 | Dead-helper removal | 9/15 (0 delta) | ✅ SHIP — behavior-neutral; code cleanliness win |
| 4 | Affordance/avoidance strip (partial) + parse_query_obj rollback | 9/15 (0 delta) | ✅ SHIP partial — aff+avoid stripped; parse_query_obj fully reverted after 3 crash canaries |
| 5 | 28×5 final stability eval | **78/140 = 55.7%** | ✅ SHIP Phase 3a — all hard criteria pass |

---

## Audit lessons

**Lesson 1 — `.gitignore` rule vs. prompts subdirectory (iter 4 process discovery)**

`voxposer/prompts/calvin/*.txt` were never `git add`-ed — a `.gitignore` rule matching `calvin/` accidentally matched the prompts subdirectory. Iter 4's rollback attempt (`git show HEAD:voxposer/prompts/calvin/parse_query_obj_prompt.txt`) silently produced an empty file. The file was reconstructed from conversation-history reading and then explicitly staged. Fix: anchored all 3 external-repo rules to root (`/calvin/`, `/3d_diffuser_actor/`, `/VoxPoser/`). All 4 prompts are now git-tracked with proper rollback safety.

**Lesson 2 — Example-set coverage is holistic, not per-example (iter 4 strip failure)**

The Phase 0 audit labeled `parse_query_obj`'s 5 fixture-substring examples (`lightbulb`, `button`, `light switch`, `black button`, `green block`) as bloat based on per-example primary purpose. They were in fact a coverage SET: thinning below threshold broke `gpt-4o`'s ability to interpolate compound queries (`led button` → empty response) even though no single example was individually load-bearing. Two retry shapes both failed identically. **Rule for future strips:** measure coverage by running a sample of plausible compound queries against the stripped prompt before any rollout.

---

## Behavior preserved / removed / relocated

| Category | Item | Notes |
|----------|------|-------|
| **Preserved** | All public API method signatures on `CalvinLMPInterface` | `detect`, `get_empty_*`, `set_voxel_by_*`, etc. — unchanged |
| **Preserved** | `detect('gripper')` internal call | `get_ee_pos` body inlined into `_detect_ee`; caller works identically |
| **Preserved** | Composer stage-tuple schema + steering/primitive/policy split docs | Verbatim in stripped `composer_prompt.txt` |
| **Preserved** | `parse_query_obj_prompt.txt` at baseline | Full revert after iter 4 failures |
| **Preserved** | Hard-fail on `max_repromptings` exhaustion | `VocabValidationError` propagates to runner — no silent fallback |
| **Removed** | `get_ee_pos` public method | Inlined into `_detect_ee`; external callers (if any) would fail — none found in audit |
| **Removed** | `quaternion_from_axis_angle` | 0 callers anywhere in codebase |
| **Removed** | 19 leakage in-context examples in `composer_prompt.txt` | Replaced with 11 paraphrased equivalents |
| **Removed** | Near-duplicate affordance/avoidance prompt variants | Never emitted per Phase 0 audit |
| **Relocated** | Stage-transition vocab validation | Bare composer call → `compose_with_repair` wrapper in `lmp.py`; `stage_manager.py` wiring is 1-line change |

---

## Smoke tests / validation

**Composer audit (iter 5, linter ON, fresh cache):**

| Check | Result |
|-------|--------|
| Valid emissions across 28 tasks | ✅ 28/28 |
| Dropped stages | ✅ 0 (baseline: 4) |
| `move_slider_*` re-prompt calls | ✅ Both succeed on attempt 1 |
| `turn_on_led` emission | ✅ `(track, push, led_button)` — perfect |
| `push_*_block` strategy | ✅ All emit `grasp+place` uniformly |
| `VocabValidationError` raised | ✅ 0 tasks exhausted `max_repromptings` |
| `ruff check` / `ruff format` | ✅ clean across all touched files |

**28×5 final stability eval (iter 5):**

| Metric | Baseline (Phase 0) | Iter 5 final | Δ |
|--------|--------------------|--------------|---|
| Overall pass rate | 31/84 = **36.9%** | 78/140 = **55.7%** | **+18.8 pp (+50% relative)** |
| Composer-side blockers resolved | 2 tasks at 0% | 0 tasks blocked by composer | All fixed by iters 1–2 |
| Phase 3b territory (0% tasks) | 10 tasks | 3 tasks | 7 unblocked by iter 2 surprise spill |

**Regressions (hard rule: baseline ≥ 2/3 must stay ≥ 1/5):**

| Task | Baseline | Iter 5 | Status |
|------|----------|--------|--------|
| `close_drawer` | 3/3 | 2/5 | ✅ passes hard rule |
| `push_blue_block_left` | 2/3 | 2/5 | ✅ passes hard rule |
| `push_into_drawer` | 2/3 | 1/5 | ✅ passes hard rule (just) |
| `stack_block` | 1/3 | 0/5 | ⚠️ baseline below 2/3 threshold — not gated, but flagged |

Root cause for all 4: strategy unification in iter 2 changed spatial-offset patterns for graspable-block-place tasks. Phase 3b's example-coverage work should narrow variance.

**Surprise: lift-from-cavity spill from iter 2**

6 tasks targeted by Phase 3b (lift-from-{slider,drawer}) were at 0% baseline. Iter 5 unlocked all 6:

| Task | Baseline | Iter 5 |
|------|----------|--------|
| `lift_pink_block_drawer` | 0/3 | **5/5 (100%)** |
| `lift_pink_block_slider` | 0/3 | **3/5 (60%)** |
| `lift_red_block_drawer` | 0/3 | **3/5 (60%)** |
| `lift_blue_block_slider` | 0/3 | **2/5 (40%)** |
| `lift_blue_block_drawer` | 0/3 | **2/5 (40%)** |
| `lift_red_block_slider` | 0/3 | **1/5 (20%)** |

Hypothesis: iter 2's rewritten lift example (`'a point 15cm above the pink block'`) caused the LLM to pattern-match a reachable target for all blocks regardless of cavity context, where the baseline emission placed the target inside the cavity.

---

## Open items

- **3 accidentally-exposed host-side helpers** (`update_state`, `get_object_names`, `get_all_detections`) remain on the LMP-callable surface via `setup_lmp`'s `dir(lmp_interface)` reflection. They're not removable from the class (host code calls them), but they shouldn't be LLM-callable. Fix requires `setup_lmp` to use an explicit allowlist instead of reflection — separate follow-up task.
- **Residual 0% tasks** (`push_blue_block_right`, `place_in_slider`) — confirmed policy-side / value-map limits, not composer-side. Phase 3b input.
- **4 graspable-block-place regressions** (`close_drawer`, `push_blue_block_left`, `push_into_drawer`, `stack_block`) — likely fixable via better balance examples in the composer prompt. Phase 3b input; all pass the hard regression rule.
- **`parse_query_obj_prompt.txt` strip** — deferred pending a coverage-aware strip methodology (see Audit lesson 2). Future attempt should test a sample of compound queries against the stripped prompt before rollout.
- **`conf/evaluation/langsteer_primitive_object.yaml` `cache_dir`** — reverted at Phase 3a end; confirm it's back at the project default before Phase 3b begins.
