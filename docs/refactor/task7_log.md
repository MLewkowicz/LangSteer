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

## Phase 1 — _(pending Phase 0 sign-off)_

## Phase 2 — _(pending)_

## Phase 3 — _(pending)_

## Phase 4 — _(pending)_

## Phase 5 — _(pending)_
