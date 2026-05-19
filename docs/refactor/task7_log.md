# Task 7 — VLM scene-image ingestion — Iteration log

Plan: `docs/refactor/task7_plan.md`.
Branch: `refactoring`.

---

## Phase 0 — Vision model A/B audit (read-only, blocking)

**Date:** 2026-05-19.
**Scope (per plan §5 Phase 0):** decide gpt-4o vs gpt-5.4-mini for the
`scene_grounding` LMP. 8 hand-labeled scenes × 2 models = 16 vision calls.

### Plumbing built for Phase 0

(All additive; production paths untouched. Vision-call cache lives in a
separate namespace via the `image_sha256` cache-key field.)

- `voxposer/lmp.py::LLMBackend.generate / _call_openai`: accept optional
  `image_bytes`. When provided, builds OpenAI multimodal `content` list
  with the JPEG inlined as a base64 `data:image/jpeg;base64,...` URL with
  `detail: high`. Anthropic path raises `NotImplementedError` (deferred).
- `envs/calvin.py::get_static_camera_matrices(W, H, fov)`: new helper
  returning the PyBullet `(view_matrix, projection_matrix)` tuple used by
  `render_high_res_static`. ~15 LoC additive.
- `voxposer/scene_image.py`: new file (~155 LoC) — overlays OBB edges +
  name labels on the captured frame via OpenCV (with Pillow fallback);
  returns JPEG bytes. Annotation discipline locked: **identity + geometry
  only, no state info**.
- `voxposer/prompts/calvin/scene_grounding_prompt.txt`: new prompt
  (~100 lines) — system spec + 3 in-context examples covering canonical,
  cavity, and P4-disambiguation cases. Output discipline: `ret_val = {...}`
  with three top-level keys (`blocks_visible`, `fixtures_state`,
  `ambiguous_resolutions`).
- `scripts/audit_scene_grounding.py`: new audit harness (~280 LoC).
  Captures 8 scenes from `CALVIN`'s `TASK_INITIAL_CONDITIONS` via
  `env.reset()` + `env.render_high_res_static(600, 600, fov=20)`, runs
  each through both models, scores against hand-labeled ground truth,
  emits aggregate metrics.

### Audit scenes

Per plan §5: 5 canonical + 3 P4 = 8 total. Ground truth derives from
`TASK_INITIAL_CONDITIONS`:
- `lift_red_block_table_canonical` — drawer closed, slider right
- `place_in_slider_canonical` — drawer closed, slider right
- `lift_blue_block_drawer_canonical` — **drawer open**, slider right
- `turn_on_lightbulb_canonical` — all off
- `turn_on_led_canonical` — all off
- `lift_red_block_table_P4` — instruction stripped to "Lift a block from the table" (color ambig → red_block)
- `turn_on_lightbulb_P4` — "Turn on the light." (light-type ambig → lightbulb)
- `lift_blue_block_drawer_P4` — "Pick up the block from the drawer." (color ambig + scene/instruction mismatch — block actually on table)

### Scene-capture sanity check

A pre-API dry-run confirmed projection math + annotation pipeline:
- OBB edges + name labels overlay on the correct objects.
- Colour palette matches `LiveCostmapTkRenderer` (`_OBJ_COLORS`).
- Fix: removed an erroneous BGR-swap in cv2.line — PIL reads numpy as RGB
  on JPEG encoding, so cv2.line should take RGB directly.

### Pre-flight smoke (gpt-4o, 1 scene)

`lift_red_block_table` canonical: gpt-4o emitted a clean grounding dict
on first try. Schema valid. `blocks_visible` exactly correct. One field
miss: `slider: 'center'` vs ground truth `'right'` (the canonical
discrete state). 3.6s wall-clock.

### Aggregate audit results

| Model | Aggregate | Blocks | Fixtures | Ambig | Schema valid |
|---|---:|---:|---:|---:|---:|
| **gpt-4o**       | 17.7% | 12.5% | 21.9% | 16.7% | **3/8** ❌ |
| **gpt-5.4-mini** | 55.4% | **87.5%** | 34.4% | 44.4% | **8/8** ✓ |

### Key findings

**1. gpt-4o is unusable due to OpenAI's safety filter.** 5/8 scenes get refused with
the literal string `"I'm sorry, I can't help with identifying or describing the
contents of the image."` (and one variant `"I can't help with that."`). The
refusal isn't tied to scene content — both canonical and P4 scenes hit it.
Likely cause: the SYSTEM_PROMPT framing ("write Python code") + image
content triggers the model's "don't analyze images of unknown subjects"
guardrail. The schema then fails to parse the refusal text.

The 3 scenes that DID succeed on gpt-4o all happened to bypass the filter
(no obvious pattern — likely a stochastic sampler quirk). Not a stable
production target.

**2. gpt-5.4-mini holds schema (8/8) but misreads fixture states systematically.**
Per-scene fixture-state diffs against ground truth:

| Field | Systematic error |
|---|---|
| `drawer` | Reports `'open'` when actually `'closed'` (6/8 scenes) |
| `slider` | Reports `'center'` when actually `'right'` (8/8 scenes) |
| `lightbulb` | Reports `'on'` when actually `'off'` (5/8 scenes) |
| `led` | Reports `'on'` when actually `'off'` (5/8 scenes) |

The drawer/slider/light states are visually distinguishable in the captured
frames (verified by manual inspection), so this is a VLM perception ceiling,
not a frame-quality issue.

**3. gpt-5.4-mini's STRONG point is `blocks_visible`** (87.5%) — exactly the field
that matters most for the Task 7 targets (cavity-task disambiguation,
P4 block-color resolution).

**4. Ambig resolution is mixed (44.4%).** Many "soft errors" are over-resolution
(emitting more disambiguations than GT, which costs accuracy points but
isn't structurally wrong). The one real miss: `lift_blue_block_drawer_P4`
→ `pink_block` instead of `blue_block`.

### Per-call cost (estimated from token counts)

| Model | Image tokens (600×600 high-detail) | Text tokens | Cost/call |
|---|---:|---:|---:|
| gpt-4o | ~425 | ~2050 + ~200 out | **~$0.008** |
| gpt-5.4-mini | ~425 | ~2050 + ~200 out | **~$0.0005** |

Both well under the $0.025 ceiling. gpt-5.4-mini is ~16× cheaper.

### Decision

**Neither model crosses the 70% bar.** Per team-lead's protocol:
"If both are <70% accurate, stop and ping me." Pausing Phases 2-5 pending
team-lead direction.

**Phase 1 (text-only Task-4 composer fixes) remains shippable in parallel**
— it does not depend on VLM grounding.

**Three possible pivots flagged for team-lead:**

1. **Narrow VLM scope to `blocks_visible` + `ambiguous_resolutions` only.**
   Drop `fixtures_state` from the grounding LMP — the actual drawer /
   slider / lights state is already available in `scene_obs` and can be
   read deterministically by `CalvinLMPInterface` (no VLM round-trip).
   With this scope, gpt-5.4-mini's accuracy becomes:
   `(blocks 87.5% × 24 + ambig 44.4% × 9) / 33 = 75.8%` — crosses 70%.

2. **Iterate on the prompt** to improve fixture-state perception. Possible
   levers: explicit instruction "look only at the raw pixels for state",
   higher image resolution (1024×1024), per-state visual cues in the
   prompt ("a closed drawer shows no interior").

3. **Switch to Anthropic claude-4.5-sonnet via direct multimodal call.**
   Adds the `_call_anthropic` image branch (currently `NotImplementedError`).
   Anthropic models historically less prone to safety-filter refusal on
   robot scenes.

Refactorer recommends (1) — leverages the model's strongest field
(blocks_visible at 87.5%) and removes the fixtures_state bottleneck. The
new accuracy estimate clears the 70% bar with the current model + prompt.

---

### Phase 2 residuals (documented, not blocking)

- **Held-block detection in `format_scene_state`.** The helper uses
  z-bucket fallbacks (z<0.42→drawer_inside, y>0.05+z>0.5→slider_inside,
  else→table). A block elevated by the gripper (z>0.55) currently
  buckets as `slider_inside`. The VLM grounding dict's `blocks_visible`
  field covers the "held" bucket via the visual path, so the gap is
  benign for the canary scope. A future Phase 2.5 could wire
  `_block_aabbs` + `robot_obs[6]` (gripper width) into the helper for
  full deterministic held-block detection.
- **Runtime malformed-VLM canary skipped.** Unit-level
  `validate_grounding` smoke covers the schema-validation path; the
  hard-fail / fallback branches in `stage_manager.setup_episode` are
  exercised by code review only, not by a degenerate-prompt runtime
  canary. Team-lead OK on this scope per Phase 2 close brief.

## Phase 1 — Task-4 deferred composer fixes (text-only)

Shipped `2434d68`. Composer prompt edits:
- Specific-color stage-0 rule (no generic `'block'` at stage 0).
- Cavity lift relaxation (`LIFTS FROM TABLE` 2-stage; `LIFTS FROM CAVITY` 1-stage).
- New `lift the blue block from the drawer` in-context example.

Canary (4 tasks × 3 ep, fresh `/tmp/task7_phase1_cache`):
| Task | Phase 1 | Task 4 baseline |
|---|---|---|
| stack_block | 0/3 | 0/2 (matched) |
| push_into_drawer | **2/3** | 0/2 hard-fail (improved) |
| lift_blue_block_slider | **3/3** | 1/3 (improved) |
| lift_red_block_drawer | 2/3 | 2/3 (matched) |

0 ObjectResolutionError, 0 generic-`block` emissions. Acceptance ✓.

## Phase 1.5 — context preambles

Shipped `fc95aa7`. Added `# TASK PURPOSE` preambles to composer + affordance
+ avoidance + scene_grounding prompts. Composer preamble specifically calls
out the disambiguation pattern ("Close it" → drawer-handle when drawer is
open).

## Phase 2 — VLM plumbing + scene_obs injection

Shipped `08503b9`. LLMBackend multimodal extension, `scene_grounding` LMP
factory entry with narrowed schema, `format_scene_state(scene_obs)` helper,
frame capture wired in `setup_voxposer_episode`. Lazy-eval fix:
`LMP.__call__` exclusion list now includes `scene_grounding`.

Phase 2 smoke (`open_drawer` 1×1):
- `enabled=false`: 1/1 success, 0 scene_grounding log lines — byte-identical
  to Phase 1 baseline.
- `enabled=true`: 1/1 success, VLM grounding called, composer prompt
  showed both `# Current scene state` (from `format_scene_state`) AND
  `# Scene state (from VLM grounding)` blocks.

## Phase 3 — wire scene-grounding USE into composer + affordance

Three iterations. Final iter 3 ships per user direction (2026-05-19):
**remove deterministic `scene_state` injection entirely; VLM is sole
source of scene perception.**

### Iter 1 — initial Phase 3 examples (NOT SHIPPED)

Added 2 composer examples + 1 affordance example. 5×3 canary:
| Task | Phase 1 | Iter 1 |
|---|---|---|
| stack_block | 0/3 | 1/3 (+1) |
| push_into_drawer | 2/3 | **1/3** (regression) |
| lift_blue_block_slider | 3/3 | 3/3 |
| lift_red_block_drawer | 2/3 | 2/3 |
| place_in_slider | 0/3 | **0/3** (gate missed) |

**Two gates missed.** Root causes per log audit:
1. VLM `ambiguous_resolutions['the block']` → red_block (not Phase 1's
   load-bearing pink_block). Policy weaker on red+blue contact geometry
   for push_into_drawer.
2. My new "outer edge of slider interior" affordance pattern triggers,
   but `y - cm2index(8)` moves AWAY from cabinet (cavity interior is at
   HIGHER y); affordance lands in empty air in front of cabinet.

### Iter 2 — A+B refinements (NOT SHIPPED)

Per team-lead direction: precision rule on `ambiguous_resolutions` +
swap cavity-lift example for `place_in_slider` example. Same canary:
| Task | Iter 1 | Iter 2 |
|---|---|---|
| stack_block | 1/3 | 0/3 |
| push_into_drawer | 1/3 | **0/3** (worse) |
| lift_blue_block_slider | 3/3 | 3/3 |
| lift_red_block_drawer | 2/3 | 2/3 |
| place_in_slider | 0/3 | **0/3** |

Both gates still failed; overall 5/15 (worse than iter 1's 7/15).
Precision rule didn't help push_into_drawer because the instruction's
"the block" IS ambiguous — composer correctly used VLM ambig
(pointing at red_block) per the rule. Cavity affordance still
geometrically wrong.

### Iter 3 (SHIPPED) — VLM-only, no scene_state injection

User direction: drop the deterministic `format_scene_state` injection
entirely. `scene_obs` is privileged simulator info; injecting it as text
into the composer bypasses the VLM-as-perception premise and isn't
defensible for real-deployment claims.

**Code patch (`steering/stage_manager.py`):** removed the
`format_scene_state(scene_obs)` call + `combined_scene_text` merge.
Composer + affordance/avoidance LMPs now receive ONLY the VLM grounding
dict's formatted text (or `None`). `format_scene_state` helper kept in
`voxposer/calvin_interface.py` for future opt-in but no callers remain.

**Prompts reverted to Phase 1.5 ship state** via `git checkout` of the
iter 2 edits. No Phase 3 examples, no precision rule.

Canary (5 tasks × 3 ep, fresh `/tmp/task7_phase3_iter3_cache`,
scene_grounding=on):
| Task | Phase 1 | Iter 3 | Δ |
|---|---|---|---|
| stack_block | 0/3 | 0/3 | matched |
| push_into_drawer | **2/3** | **0/3** | **−2 regression** |
| lift_blue_block_slider | 3/3 | 3/3 | matched |
| lift_red_block_drawer | 2/3 | **3/3** | **+1 improvement** |
| place_in_slider | 0/3 | 0/3 | matched |
| OVERALL | 7/15 | **6/15** | −1 |

**Log audit:** 0 `# Current scene state` lines (deterministic block
gone ✓), 39 `# Scene state (from VLM grounding)` lines (VLM grounding
still active across composer + affordance + avoidance prompts).

### Phase 3 interpretation (per team-lead's decision tree)

Maps to "push_into_drawer drops" branch: VLM ambig is net-harmful for
unambiguous canonical instructions where Phase 1's hardcoded color was
already correct. Lift_red_block_drawer improved unexpectedly (likely VLM
`blocks_visible` voting blue→drawer caused composer to pick a different
trajectory shape; or N=3 noise). place_in_slider unchanged from 0/3
baseline — VLM grounding alone doesn't unlock cavity targets without a
geometrically-correct affordance pattern (out of Task 7 scope per
team-lead).

**Aggregate VLM-on signal: weakly net-negative (-1 ep) but with clear
"hurt push_into_drawer, helped lift_red_drawer" split.** Suggests
per-task VLM toggling could net positive overall — escalate to user for
Phase 5 design call.

## Phase 4 — _(pending Phase 3 close + Phase 5 design)_

## Phase 5 — _(pending)_
