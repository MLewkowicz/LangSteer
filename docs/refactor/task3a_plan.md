# Task 3a — VoxPoser prompts cleanup + re-prompt loop + helper API audit

**Owner:** planner → refactorer (after team-lead approval)
**Branch:** `refactoring`
**Scope:** Prompt-side fixes only. Value-map shape redesign (full-bounding-box affordances, EDT obstacle masking) is **Task 3b**, separately planned. 3a ships first; 3b is built on the clean prompts from 3a.

Differs from Tasks 1–2: iterative LLM-driven empirical work, not a one-shot diff. The plan has an **audit phase** (read-only, no edits) before the rewrite phase, and each phase has explicit stopping criteria.

---

## 0. TaskList split (executed by team-lead)

The Task 3 split was approved and applied by the team-lead. Current state:

| Task ID | Subject | Status | Blocks |
|---|---|---|---|
| #3 | Task 3a — VoxPoser prompts cleanup + vocab linter + dead-helper removal | in_progress | #4 |
| #6 | Task 3b — Value-map redesign (bbox affordances + obstacle EDT masking) | pending | #4 |
| #4 | Verify multi-stage steering (grasp gate + loop-back) | pending | — |
| #5 | Clean up visualization utils | pending | — |

This plan is the working spec for task #3 (the 3a phase). Task 3b (#6) gets a separate plan written **after** 3a's Phase 0 audit lands — the audit may shift 3b's scope (e.g. if the dominant failure mode is composer-side rather than value-map-side, 3b's value-map work may shrink).

---

## 1. Original brief excerpt (quoted verbatim for self-contained doc)

> "Iterate on voxposer/prompts and helper functions: enforce action+object primitive vocabulary, eliminate in-context-example leakage, fix slider_handle/door_handle confusion. Redesign value maps for full-bounding-box affordances + obstacle-aware tapering. Test via `uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object` on small episode counts per task."
>
> *— from `refactoring_tasks.md` (the user's original brief). Full-bounding-box / EDT-tapering parts are out of 3a's scope (they're 3b).*

User-reported symptom that anchors 3a: the LLM kept emitting `door_handle` for `move_slider_left`; `parse_composer_stages` (Task 1's parser) dropped the stage silently because `door_handle ∉ OBJECT_VOCAB`; the rollout had no guidance; the task failed.

Hard constraint: `action_primitive_object_annotations.json` is post-hoc eval ground truth only. **The composer must never read it at inference time.** Verified by `grep -rn` — no current hits in `voxposer/`, `steering/`, or `policies/`. 3a's negative test repeats this grep.

---

## 2. Audit protocol (Phase 0 — read-only, blocking)

**Goal:** Produce baseline measurements before any code change. Phase 1 (rewrite) compares against this baseline; without it, "did the change help?" is unanswerable.

**Deliverable:** `docs/refactor/task3a_baseline.json`, committed before any rewrite edits.

### 2.1 Composer emission catalog

For each of the 28 tasks in `conf/evaluation/task_order.json`, run the composer **once with cache disabled**, recording:

| Field | Notes |
|---|---|
| `task_name` | e.g. `move_slider_left` |
| `instruction` | the natural-language string fed to the composer |
| `raw_emission` | full LLM output (the executed Python from `exec_safe`) |
| `parsed_stages` | list of `{mode, primitive, object, has_rot_target}` after `parse_composer_stages` |
| `dropped_stages` | list of `{index, reason}` for each rejected stage |
| `dropped_reason_codes` | set of unique reasons (`invalid_primitive`, `invalid_object`, `wrong_tuple_length`, `unparseable`, `none_returned`) |

For `move_slider_left`, this should immediately tell us whether the symptom is:
- **(a) Vocab gap** — `dropped_reason = invalid_object` with offender `'door_handle'`. The re-prompt loop fixes it.
- **(b) Semantic gap** — `object = 'drawer_handle'` (valid name, wrong fixture). Prompt needs handle disambiguation ("slider is the horizontal sliding panel; drawer is the pull-out box below").
- **(c) Something else** — e.g. composer crash, empty emission, off-vocab primitive.

The audit drives §3.4's handle-disambiguation copy: write it for the symptom we actually have, not the one we guessed at.

**Suggested mechanism:**
- Standalone script `scripts/audit_composer.py` (outside production scope; not part of the refactor diff).
- Uses `voxposer.lmp.setup_lmp(cfg)` to build the LMP system the same way `StageManager` does.
- Iterates `task_order.json`, calls `lmps['composer'](instruction)`, captures `raw_emission` + runs it through `parse_composer_stages`.
- Writes `task3a_baseline.json` with one entry per task plus a summary header (counts of each dropped-reason code).

Cost: 28 LLM calls. At ~3s/call (uncached, claude-sonnet) ≈ 90 seconds + ~$0.10 in API charges. Cheap. No GPU required.

### 2.2 Baseline rollout eval

```bash
uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object \
    steering=voxposer \
    --tasks conf/evaluation/task_order.json
```

The user's brief says "small episode counts per task" — **start with 3 ep × 28 tasks = 84 episodes**, which mirrors the precedent the team-lead set ("3 episodes per task = 84 episodes total"). On a modern GPU this is ~30–60 minutes; LLM-API cost is dominated by cache hits after the composer audit ran (same cache_dir reused). If after seeing the audit-emission catalog the team-lead wants to drop to 1 ep × 28 to save time, that's a one-line CLI change — the audit script is the load-bearing artifact, eval is just confirmation.

Record per task:
- Success count / 3 (or /1 if scoped down).
- For each failure, a one-line **failure-mode classification**:
  - `composer_side:invalid_emission` — stage(s) dropped by parser.
  - `composer_side:wrong_decomposition` — valid emission, semantically wrong (e.g. 1 stage when 2 are needed).
  - `value_map_side:gradient_pulls_into_obstacle` — trajectory diverges into another object.
  - `policy_side:trajectory_fights_guidance` — guidance signal is right but the policy ignores it (rare).
  - `unknown` — needs deeper trace.

Output: `docs/refactor/task3a_baseline.json::per_task_eval_baseline`.

The classification is what feeds team-lead's "target ≥X% of composer-side failures fixed after 3a, ≥Y% of value-map-side failures fixed after 3b". Numbers are set in §6 **post-baseline**, not now.

### 2.3 Prompt-content audit

For each of the four prompt files, the refactorer reads through and tags every example block:

| Label | Definition |
|---|---|
| `load_bearing` | The example demonstrates a non-obvious pattern the LLM cannot reconstruct from peers. Keep verbatim. Examples: world-frame rotation idiom (composer L60–72); the "rotate is not a lift" warning (L74–77); 2-stage push contact-physics prior (L79–96). |
| `bloat` | Redundant or near-duplicate of another example. Slight phrasing change, same shape. Drop. |
| `task_name_leakage` | Example query that matches a CALVIN task name verbatim ("open the drawer", "move the slider to the left", "push the red block to the left", "rotate the blue block to the left"). Replace with a paraphrased/unrelated probe (e.g., "retract the drawer's handle slowly", "shift the sliding panel toward you"). |
| `decomposition_leakage` | Example whose stage decomposition exactly matches what a CALVIN task needs (e.g. the `open_drawer` example emits `[(grasp, drawer_handle), (pull, drawer_handle)]` — that's the CALVIN-task ground-truth recipe). Replace with a non-CALVIN-named example that demonstrates the same general pattern. |
| `coverage_gap` | Not an example — a **missing** category. E.g. no `rotate × pink_block`, no `push × red_block × right`. Tally these to drive Phase 1's balance-matrix work. |

The output is a markdown table per prompt file under `docs/refactor/task3a_prompt_audit.md`.

### 2.4 Helper-usage audit

Grep each prompt file for every name in the LMP namespace exposed by `CalvinLMPInterface.get_object_names()`:

```bash
grep -E "parse_query_obj|detect|get_affordance_map|get_avoidance_map|get_empty_affordance_map|get_empty_avoidance_map|set_voxel_by_radius|set_voxel_by_box|cm2index|get_ee_pos|rotation_about_axis|compose_rotation|current_ee_rotation" voxposer/prompts/calvin/*.txt
```

Tally per helper: `<helper> → N example occurrences`. Helpers with 0 occurrences are dead code in the LMP namespace from the original VoxPoser/RLBench port. The output goes in `task3a_baseline.json::helper_usage`.

Refactorer also greps `voxposer/calvin_interface.py` itself for unused internal methods (defined but never called by exposed methods or examples).

### 2.5 Leakage-source verification (negative test)

```bash
grep -rn "action_primitive_object_annotations" -- voxposer/ steering/ policies/
```

Expected: empty inside these directories. Any hit is a critical pre-existing leakage that must be triaged before 3a continues. Today: empty (verified by planner during this plan's prep). This grep runs again at 3a's acceptance gate to catch regressions.

### 2.6 Phase 0 stopping criterion

Refactorer pings planner with `task3a_baseline.json` + `task3a_prompt_audit.md` when both files are committed. Planner reviews the dropped-reason-codes distribution and confirms the §3 rewrite priorities are aligned with what the data actually shows. **No Phase 1 code edit lands before this review.**

---

## 3. Prompt-rewrite spec

**Targets** (gross, not bytewise — refactorer iterates):

| Prompt file | Today | Target |
|---|---:|---:|
| `composer_prompt.txt` | 367 lines | ≤ 200 |
| `get_affordance_map_prompt.txt` | 146 lines | ≤ 80 (3a target — bigger rewrite happens in 3b) |
| `get_avoidance_map_prompt.txt` | 52 lines | ≤ 35 |
| `parse_query_obj_prompt.txt` | 74 lines | ≤ 40 |

These are guidelines, not contracts. The acceptance criterion is "no `bloat`, `task_name_leakage`, `decomposition_leakage` rows remain in the audit table" — line counts follow naturally.

### 3.1 Vocabulary cheat-sheet (composer prompt header)

Add a normative block near the top of `composer_prompt.txt` (above any example). Lists the vocabs explicitly with the known confusion alias resolved:

```
# VOCABULARY (the parser rejects anything outside these — re-prompt loop will
# force a fix, so emitting invalid values just wastes turns):
#
# primitives: grasp, push, pull, place, rotate
# objects:    block, blue_block, drawer_handle, led_button, lightbulb_switch,
#             pink_block, red_block, slider_handle
#
# Cabinet door handle = slider_handle. There is NO 'door_handle'.
# Drawer pull bar     = drawer_handle.
# Light's toggle      = lightbulb_switch.
# LED's push button   = led_button.
```

The handle disambiguation lines are the cheapest possible fix for the `move_slider_left` symptom. They go in regardless of whether the audit confirms vocab-gap vs. semantic-gap.

If the audit shows semantic-gap (LLM picks `drawer_handle` when it should pick `slider_handle`), add ONE more disambiguation block describing the physical objects:

```
# CALVIN scene primer:
# - slider: horizontal sliding panel on the upper cabinet face. Grab via
#   slider_handle (the small groove near the front edge).
# - drawer: pull-out box BELOW the slider. Grab via drawer_handle (the
#   horizontal pull bar flush with the drawer's front face).
```

### 3.2 Balance-matrix rewrite (composer prompt body)

Today's coverage gaps (planner's audit during this plan's prep):

- **rotate × block:** red→{left, right}, blue→left present; blue→right and pink→{left, right} **absent** → LLM may default to memorized red/blue patterns.
- **push × block:** red→left, blue→right (alt as grasp-place), pink→left (alt as grasp-place) — missing red→right and any pink→right.
- **lift × block × surface:** blue from table, pink from slider — no red, no other surfaces explicitly demonstrated.

**Rewrite shape:** instead of trying to enumerate every cell of the (color × direction × action) matrix (would explode the prompt), **decouple** color from action by using **non-CALVIN-named example queries**:

- Replace "push the red block to the left" → "shift the {neutral_color} marker toward the {arbitrary_side}". The example demonstrates the 2-stage push shape, but the LLM cannot pattern-match its substring against a CALVIN task name.
- Use 2 examples per primitive (1 canonical + 1 with twist) rather than 3+ examples per (color × direction) combination. Total example count drops; coverage per pattern stays.

The structured-form queries at the bottom of today's prompt (`rotate blue_block`, `pull drawer_handle`) are a different problem — they're meant to demonstrate the schema, not to teach decomposition. Keep 2–3 of them as a "schema sample" block but strip the surplus.

### 3.3 Leakage-strip checklist

For each leakage row from the prompt audit (§2.3), the refactorer:

1. Identifies the example's load-bearing pattern (what general shape it teaches — 2-stage push, grasp+place alt, rotate idiom, etc.).
2. Rewrites the example with the **same shape** but a **non-CALVIN-named query**. Suggested phrasings:
   - "open the drawer" → "retract the drawer's handle"
   - "move the slider to the left" → "shift the sliding panel toward you"
   - "push the red block to the left" → "displace the red marker sideways"
   - "rotate the blue block to the left" → "yaw the blue cube counterclockwise"
3. Verifies the rewrite still parses the same way (same stage tuple shape, same primitives, same objects).
4. Logs the rewrite in `task3a_phase1_log.md` so the audit/prompt diff is reviewable.

### 3.4 Handle disambiguation (driven by audit)

The exact copy depends on what the audit shows for `move_slider_left`. Three branches in the rewrite:

- **Vocab-gap branch** (LLM emits `door_handle`): §3.1's cheat-sheet alone is likely enough. The re-prompt loop (§4) is the second line of defense.
- **Semantic-gap branch** (LLM emits `drawer_handle` for the slider task): add §3.1's CALVIN-scene primer. Also add 1 explicit example to the composer prompt: "move the slider to the left" → 2-stage push with `slider_handle`, with an inline comment "the slider is the sliding panel — NOT the drawer".
- **Other branch:** plan a third disambiguation per the symptom and ping team-lead before continuing.

### 3.5 What stays load-bearing (don't strip)

Planner's protected list (refactorer must not remove without re-justification):
- Composer prompt L39–58: steering / primitive / policy split block.
- Composer prompt L60–72: world-frame rotation idiom.
- Composer prompt L74–77: "rotate is not a lift" warning.
- Composer prompt L79–96: contact-physics priors / 2-stage push doc.
- Composer prompt L99–101: vocabulary terminal sentence (the "NEVER emit" rule).
- Affordance prompt L135–146: `set_voxel_by_box` example for handles (load-bearing for 3b's full-bbox work).

---

## 4. Re-prompt loop spec

**Where it lives:** wrapper around the composer call. Two siting options:

| Option | Pros | Cons |
|---|---|---|
| **(a) Inside `voxposer/lmp.py`** | LMP runtime is where the composer is invoked today; one file owns the LMP-runtime concerns | Slight bloat in `lmp.py` (currently 401 lines) |
| **(b) New `voxposer/composer_validator.py`** | Cleanly separated; testable independently | One more file; needs explicit wiring |

**Default:** (a) — single file, no new module. Planner's lean.

### 4.1 Public API

```python
# voxposer/lmp.py — added near the composer-call site

from steering.stage_spec import PRIMITIVE_VOCAB, OBJECT_VOCAB, VALID_STAGE_MODES

# Known LLM confusions to surface in repair hints. Data, not code logic.
# Extended by refactorer as audit surfaces new failure modes.
HANDLE_ALIASES: dict[str, str] = {
    'door_handle': 'slider_handle',
    # Add more as discovered, e.g.:
    # 'cabinet_handle': 'slider_handle',
    # 'pull_handle':    'drawer_handle',
}


def composer_with_retries(
    composer_lmp,
    instruction: str,
    *,
    max_retries: int = 2,
) -> Any:
    """Call the composer; if the emission has invalid primitives/objects,
    re-prompt with a corrective hint up to `max_retries` times.

    On exhaustion, raise `ComposerValidationError` with the last raw emission
    and the per-stage violation list. Caller is `StageManager.setup_episode`,
    which will surface the error to the rollout.

    No fallback to closest-match; we surface failures rather than hide them.
    """
    ...
```

### 4.2 Retry budget

**Team-lead-locked: max_retries = 2.** Total LLM calls per task ≤ 3 (1 initial + 2 retries). On exhaustion, hard-fail with `ComposerValidationError` carrying the last emission + violation list. No closest-match fallback.

### 4.3 Corrective hint format

The hint is prepended to the original prompt for the retry:

```
Your previous response contained values outside the allowed vocabulary.

{per-stage violation details — one block per offending stage:}
  Stage {i}: {field}={offender!r} is not in the vocabulary.
  Valid {field}s: {sorted(VOCAB)}
  Suggested fix: {closest_match_from_HANDLE_ALIASES_or_edit_distance}

Re-emit the FULL stage list with all stages valid. Stages that were valid
before should be re-emitted unchanged.
```

Closest-match resolution order:
1. `HANDLE_ALIASES` lookup (explicit known confusions).
2. Levenshtein edit-distance to the nearest vocab entry (only suggested if distance ≤ 3).
3. If neither, omit the "Suggested fix" line (don't invent).

### 4.4 Validation hookpoint

Today the composer call is `result = self._lmps['composer'](instruction)` inside `StageManager.setup_episode`. After 3a, the call becomes:

```python
result = composer_with_retries(self._lmps['composer'], instruction)
```

`composer_with_retries` internally calls `parse_composer_stages` (Task 1's parser) for validation — same vocab sets, no duplication. If `parse_composer_stages` drops zero stages, return the raw `result` unchanged; if it drops any, retry.

This means **the parser is the source of truth for validity**. The re-prompt loop merely reacts to it.

### 4.5 Logging

Every retry emits a structured INFO line so the rollout transcript shows what happened:

```
[composer-retry] task=move_slider_left attempt=1/2 invalid=[Stage 0 object='door_handle' → suggest 'slider_handle']
[composer-retry] task=move_slider_left attempt=2/2 success
```

Exhaustion logs at ERROR level with the full final emission.

### 4.6 Cache interaction

`voxposer/llm_cache.py` keys on `(provider, model, prompt, temperature, max_tokens)`. Each retry has a different prompt text (carries the corrective hint), so cache keys naturally diverge. No special handling needed.

Side effect for refactorer to be aware of: a failing task's retries get cached, so a second run of the same task with the same prompts will hit cache on the retry path. That's the intended behavior — it speeds up iteration. To force a clean re-eval, bump `cache_dir`.

### 4.7 Failure mode if exhausted

`ComposerValidationError` propagates up through `StageManager.setup_episode` → `VoxPoserSteering.setup_episode` → `setup_voxposer_episode` (in `run_evaluation.py`) → `run_condition`. Today's pattern is `if steering._value_map is not None: ... else: logger.error("VoxPoser value map generation FAILED")` — the error path already exists. 3a's change is that `_value_map is None` becomes a more specific exception that the runner can log structurally instead of detecting it post-hoc.

Refactorer decision (defer to execution): does the rollout for that episode count as a failure with reason `composer_exhausted`, or does the episode get skipped entirely? Default: count as failure with that reason. This lets §2.2's failure-mode classification distinguish "composer gave up" from "trajectory diverged".

---

## 5. Helper API audit (in `voxposer/calvin_interface.py`)

**Scope per team-lead's Q1 answer: kill dead code, keep what's used. No full rename.** This is narrower than the helper-redesign brainstorm I'd floated in the v0 plan.

### 5.1 What the audit captures

Per §2.4's grep:

```
helper                          examples_using   action
parse_query_obj                  ~all             keep
detect                            internal only    keep (internal)
get_empty_affordance_map         12               keep
get_empty_avoidance_map           5               keep
set_voxel_by_radius              ~5               keep
set_voxel_by_box                  2               keep (3b expands usage)
cm2index                         ~10              keep
get_ee_pos                        0?              flag for review
rotation_about_axis               ~5               keep
compose_rotation                  ~5               keep
current_ee_rotation               ~5               keep
```

(Exact counts come from the audit. The table above is a planner's estimate.)

### 5.2 What to remove

- Helpers in `calvin_interface.py` with **0 example occurrences** AND **no internal usage** by other helpers.
- Internal methods that are dead (defined but never called).

Refactorer commits a `git diff --stat voxposer/calvin_interface.py` line count delta as evidence (e.g. "662 → 580 lines, no behavior change for any helper still referenced by prompts").

### 5.3 What NOT to do (per team-lead Q1)

- **No surface rename** (`parse_query_obj` → `scene`, etc.). The brainstorm from my v0 plan is shelved.
- **No new helper functions in 3a.** New helpers for full-bbox affordance / EDT obstacle mask come in 3b.
- **No re-signature** of existing helpers. `set_voxel_by_box(map, obj, value, pad_cm)` keeps that signature.

### 5.4 Tests for the helper-audit step

After removal: the audit grep from §2.4 re-runs. Every example in every prompt file must still resolve to a defined helper. Any "name 'foo' is not defined" failure inside `exec_safe` is a bug in the removal.

A test command:

```bash
uv run python scripts/audit_composer.py --tasks conf/evaluation/task_order.json \
    --validate-only \
    --cache-dir /tmp/post_helper_audit_cache
```

Runs the composer for all 28 tasks; any `NameError` from a removed helper surfaces immediately.

---

## 6. Iteration shape

3a is at least **2 iterations** (baseline → rewrite → re-eval → adjust → re-eval). Each iteration has a stopping criterion. Refactorer logs progress in `docs/refactor/task3a_phase1_log.md`.

### 6.1 Iteration plan (refactorer's loop)

```
PHASE 0 — AUDIT (no edits)
  Run audit script + baseline eval. Commit task3a_baseline.json + prompt audit.
  Ping planner. Wait for sign-off on rewrite priorities.

ITERATION 1 — re-prompt loop + vocab cheat-sheet + handle disambiguation
  Goal: zero invalid emissions across 28 tasks.
  Changes:
    - voxposer/lmp.py: add composer_with_retries + HANDLE_ALIASES.
    - composer_prompt.txt: add vocabulary cheat-sheet near top (§3.1).
    - composer_prompt.txt: add handle disambiguation if audit says semantic gap (§3.4).
  Validate:
    - Re-run audit script: dropped_stage_count == 0 across 28 tasks.
    - Re-run eval on the canary set: {move_slider_left, open_drawer, push_red_block_left,
      lift_blue_block_table, rotate-task-if-present} × 3 ep.
  Decision:
    - If invalid emissions = 0 AND no regression on canary set → ship iteration 1.
    - If invalid emissions > 0 → diagnose (the LLM is fighting the new prompt? cache hit?), fix, re-run.
    - If canary regression → rollback the iteration, ping planner.

ITERATION 2 — leakage strip + bloat consolidation + balance matrix
  Goal: composer prompt ≤ 200 lines, no task_name/decomposition leakage rows, all
        coverage gaps from §2.3 audit are either filled or explicitly skipped.
  Changes:
    - composer_prompt.txt: leakage strip + replace with paraphrased queries (§3.3).
    - composer_prompt.txt: collapse redundant rotate/push examples per §3.2.
    - composer_prompt.txt: structured-form section pruned to 2-3 entries.
    - affordance/avoidance/parse_query prompts: similar strip (smaller targets — §3 table).
    - calvin_interface.py: dead-helper removal (§5).
  Validate:
    - Re-run audit: dropped_stage_count still 0; line counts hit targets.
    - Re-run eval on full 28 tasks × 3 ep.
  Decision:
    - If no regression vs. baseline pass rate (task-by-task) → ship iteration 2.
    - If 1-task regression with clear attribution (e.g. removed an example the LLM
      relied on for this exact task) → reintroduce a paraphrased version of that
      example, re-run.
    - If ≥2-task regression OR unclear attribution → rollback the iteration, ping planner.

ITERATION 3 (optional) — patch-up
  Goal: handle anything iterations 1 and 2 surfaced that needs a focused fix.
  Capped at one iteration of patch-up. If a third iteration's not enough, ping planner
  for re-scope.
```

### 6.2 Per-iteration stopping rules (any of)

- **SHIP:** iteration goal met + no regression vs. previous iteration's pass rate.
- **ROLLBACK:** invalid-emission count increased OR ≥2-task pass-rate regression OR refactorer can't diagnose.
- **PING-PLANNER:** ambiguity about whether to ship or rollback (e.g. 1-task regression with unclear cause).

### 6.3 Numerical targets (set POST-baseline, not now)

Per team-lead's Q6 answer: don't lock numbers before audit. After Phase 0 lands, planner reviews `task3a_baseline.json` and proposes targets of the form:

> Of the **N** tasks failing today with `composer_side:*` classification, target **≥75%** passing after 3a. No regression on currently-passing tasks. `move_slider_left` specifically passes ≥ 2/3.

The exact `N` and the regression list are set by the baseline. This planner→team-lead reconciliation happens between Phase 0 and Iteration 1.

### 6.4 Phase 1 acceptance criteria (3a "done")

All of:
- ✅ `task3a_baseline.json` committed.
- ✅ `task3a_phase1_log.md` committed with per-iteration entries.
- ✅ Re-audit script run at 3a's end shows `dropped_stage_count == 0` over 28 tasks.
- ✅ `composer_prompt.txt` ≤ 200 lines.
- ✅ Per-task pass rate at 3 ep × 28 tasks meets the post-baseline targets.
- ✅ `move_slider_left` specifically passes ≥ 2/3.
- ✅ Grep negative test passes: no inference-side imports of `action_primitive_object_annotations`.
- ✅ `ruff check voxposer/` clean (or no new errors introduced).
- ✅ Production smoke test passes: `uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer` runs end-to-end without crash.

---

## 7. Risks / gotchas

1. **Audit budget.** Phase 0 is 28 LLM calls + 84 GPU episodes. Total ~30–60 min wall-clock, ~$0.10 LLM. Tolerable. If the team-lead wants to scope down: drop to 1 ep × 28 tasks (84 → 28 episodes, ~15 min). The audit script's value is the emission catalog, not the eval — the eval is just confirmation. Flag this for team-lead in the summary.

2. **Cache contamination.** Every prompt edit invalidates all cached LLM responses for that prompt. The 28 tasks fire fresh LLM calls per iteration. Refactorer should bump `cache_dir` between iterations OR accept cold cache (28 × ~3s = ~90s/iteration). Either is fine.

3. **Re-prompt loop infinite-loop guard.** `max_retries = 2` caps total LLM calls per task at 3. A pathologically broken composer can't ping-pong forever. Refactorer logs per-task retry counts and watches for tasks that consistently hit the cap.

4. **Re-prompt hint may bias the LLM.** Suggesting "did you mean `slider_handle`?" is a strong hint that may steer the LLM toward `slider_handle` even when the right answer is `drawer_handle`. Mitigation: the suggestion is just a hint string; the LLM is asked to "re-emit the FULL stage list" so it has the opportunity to reconsider primitive AND object together. If post-3a we see the hint over-steering, weaken to just listing the valid set without a specific "fix".

5. **Prompt-strip regression.** Removing an example the LLM was implicitly relying on causes a task that passed in baseline to fail. Mitigation: 1-task regression with clear attribution → reintroduce a paraphrased equivalent. ≥2-task regression → rollback the iteration.

6. **Helper removal breakage.** Dead code might be "dead" in the prompt examples but reachable from a path the audit script doesn't exercise (e.g. internal calls). Mitigation: refactorer runs the validate-only audit (§5.4) before shipping, plus the production smoke test catches anything the audit misses.

7. **Eval flakiness.** Diffusion + steering has stochastic elements (random seed in `set_seed`, MCMC corrector). If a per-iteration eval shows a 1-task regression and the next run shows it passing, that's noise. Mitigation: refactorer re-runs the offending task at 5 ep before concluding regression.

8. **Composer hard-fail vs. parse-failure today.** Today, when `parse_composer_stages` returns `[]`, `StageManager.setup_episode` logs `Composer failed` and proceeds with no guidance. Post-3a, `composer_with_retries` raises `ComposerValidationError`. The runner (`run_evaluation.py::setup_voxposer_episode`) needs to handle this exception — either catch it and log the per-episode failure with reason `composer_exhausted`, or let it propagate. Default: catch, log, count the episode as failed. Refactorer wires this in iteration 1.

9. **Task 1 / Task 2 territory boundary.** 3a touches `voxposer/`, the four prompt files, and ONE wiring point at `run_evaluation.py` for the `ComposerValidationError` handler. Does NOT touch `steering/`, `policies/`, `core/`, training, or visualization. The boundary discipline mirrors Tasks 1 and 2.

10. **Process discipline (Tasks 1 / 2 lessons).**
    - **No silent plan rewrites after team-lead approval.** If iteration in Phase 1 reveals the plan's framing is wrong, planner pings team-lead with "should we reconsider X?" before any plan edit.
    - **Numerical targets set post-baseline.** Don't pre-commit to "20/28 pass" or similar before seeing the audit.
    - **Refactorer pings planner per-iteration if anything's ambiguous.** Iteration breakpoints are natural pause moments.

---

## 8. Out of scope for Task 3a (these belong elsewhere)

- **Value-map shape changes** — full-bounding-box affordance painting, EDT obstacle masking, gradient-direction safety inside obstacle interiors. All deferred to **Task 3b**.
- **Helper API renames** — `parse_query_obj` → `scene`, etc. Per team-lead Q1 answer, scope is dead-code cleanup only.
- **Steering refactor** (Task 1 territory): callbacks, grasp gate, transitions, loop-back.
- **Policy variants** (Task 2 territory): wrapper splits, conditioning paths.
- **Visualization** (Task 5 territory): live viewer, HTML dumps.
- **Training-data preprocessing** (`scripts/preprocess_primitive_object_annotations.py`, `training/policies/diffuser_actor/trainer.py`). Phase 3 is inference-only.
- **Evaluation harness** (`conf/evaluation/task_order.json`, `conf/evaluation/langsteer_primitive_object.yaml`). 28-task list is the eval contract.
- **New primitives or new objects in the vocabulary.** Vocab is frozen at Task 1's `PRIMITIVE_VOCAB`/`OBJECT_VOCAB`. Extending it would require retraining (training-data implications).
- **Generic LLM-prompt-engineering uplift beyond the four named issues** (vocab adherence, leakage, handle confusion, helper-API audit). E.g. NOT redesigning the rotation idiom; NOT changing the stage-tuple schema.

### Captured as future scope
- Full helper API rename / signature redesign (deferred indefinitely; team-lead's Q1 answer rules out for 3a).
- Auto-tuning the re-prompt hint copy via offline analysis of repair logs.
- Adding a regression-test suite that exercises the composer on a fixed instruction set (would help 3b's stability check).

---

## 9. Sign-offs requested from team-lead BEFORE Phase 0 starts

- **(P0) TaskList split per §0.** Should I proceed to TaskUpdate or wait for separate confirmation?
- **(P0) Audit episode budget.** 84 episodes (3 ep × 28 tasks) ≈ 30–60 min. If you want this scoped down (e.g. 1 ep × 28 = 28 episodes), say so. Default is 84.
- **(P0) Failure-handling at the runner.** When `composer_with_retries` exhausts retries, should the runner: (a) catch + log + count episode as failed (default), (b) skip the episode entirely, or (c) crash the run with a clear error?
- **(P0) Helper rename scope.** Confirmed `dead-code-only` per your Q1 answer; just double-checking there's no leftover ambition for a surface rename hiding in Sub-goal D from the v0 plan. Reply "confirmed dead-code-only" if you want me to lock that in.

Phase 0 starts when these four are answered. Once Phase 0's data lands, planner returns with numerical targets for §6.3 + final iteration plan sign-off.

---

## 10. Acceptance criteria (Task 3a "done")

Refactorer marks 3a complete when all of:

- ✅ §6.4 acceptance criteria met (re-audit shows 0 dropped stages, composer prompt ≤ 200 lines, target pass rates met).
- ✅ `task3a_baseline.json`, `task3a_prompt_audit.md`, `task3a_phase1_log.md` all committed.
- ✅ Production smoke test passes end-to-end.
- ✅ `move_slider_left` canary passes ≥ 2/3 ep.
- ✅ `git diff --stat voxposer/calvin_interface.py` shows a non-zero line reduction (dead code removed).
- ✅ `ruff check voxposer/` clean.
- ✅ Negative leakage test passes: `grep -rn "action_primitive_object_annotations" -- voxposer/ steering/ policies/` returns 0 hits.

Once 3a closes, 3b is unblocked. 3b's plan is a separate doc (`docs/refactor/task3b_plan.md`); the brief is captured in the future-scope section of this doc + team-lead's existing notes.
