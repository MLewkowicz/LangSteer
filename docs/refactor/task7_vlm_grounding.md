# Task 7 — VLM scene-image ingestion (final scribe doc)

**Status:** ✅ SHIPPED
**Branch:** `refactoring`
**Commit chain:** `40591bd` → `2434d68` → `fc95aa7` → `08503b9` → `56f5b10` → `task7 phase 5`
**Total wall-clock:** ~12 hours across phases 0-5
**Total LLM cost:** ~$0.43 vs $15 ceiling
**Plan:** [`task7_plan.md`](task7_plan.md). **Iter log:** [`task7_log.md`](task7_log.md).

---

## TL;DR

Task 7 added an OpenAI multimodal **scene-grounding LMP** that ingests an
annotated overhead camera frame at episode start and emits a structured
disambiguation dict (`blocks_visible` + `ambiguous_resolutions`) for the
composer + affordance/avoidance LMPs to consume as text context. The
narrowed schema (per Phase 0 audit pivot) drops fixture state from the
VLM's job since scene_obs is ground truth — but a user-driven decision
during Phase 3 removed deterministic scene_obs injection too: **VLM is
the sole source of scene perception** to keep the design defensible for
real-deployment claims.

Phase 5 acceptance evals shipped under an option-(A) split:
- **Canonical 28×5 (VLM OFF): 110/140 = 78.6%** vs 3b final v3 76/140 = 54.3% (+24.3pp)
- **P4 perturbed 28×N=90 (VLM ON): 41/90 = 45.6%** with 0 hard-fails

The canonical jump is primarily attributable to Phase 1's composer fixes
(specific-color stage-0 rule + cavity lift relaxation) + Phase 1.5
context preambles + post-Task-4 `max_steps=360`. The P4 result validates
the VLM grounding pipeline end-to-end: 100% emission validity, correct
disambig on representative phrases like `"Open it." → drawer` and
`"the block from the drawer" → blue_block`.

---

## Phase chain summary

| Phase | What shipped | Commit | Outcome |
|---|---|---|---|
| 0 | VLM model A/B audit | `40591bd` | gpt-4o refuses 5/8 (safety filter); gpt-5.4-mini 55.4% agg (87.5% blocks). PIVOT: narrow VLM to blocks_visible + ambig_resolutions only. |
| 1 | Task-4 deferred composer fixes (specific-color stage-0 + cavity lift relax) | `2434d68` | 4-task canary: 3 of 4 improve, push_into_drawer 0/2→2/3, lift_blue_slider 1/3→3/3 |
| 1.5 | Context preambles on 4 prompts (composer/affordance/avoidance/scene_grounding) | `fc95aa7` | Explains task purpose + disambig pattern to LLMs |
| 2 | VLM plumbing (LLMBackend multimodal + scene_grounding LMP + frame capture + `format_scene_state` helper) | `08503b9` | Smoke: enabled=false byte-identical; enabled=true grounding dict + scene-state in composer prompt |
| 3 (iter 3) | Drop deterministic scene_state injection; VLM-only | `56f5b10` | Per user direction. 6/15 canary, push_into_drawer regression vs Phase 1 baseline. Data informs Phase 5 design. |
| 5 | Split canonical (VLM off) + P4 (VLM on) | _(see commit)_ | Canonical 78.6%, P4 45.6%, 0 hard-fails |

---

## Architecture (final)

Architecture B locked in Phase 0 with the Phase 3-iter-3 user refinement:

```
Episode start:
  scripts/run_evaluation.py::setup_voxposer_episode
    └─ env.render_high_res_static(600, 600, fov=20)
    └─ render_annotated_overhead(rgb, detections, view, proj)
         (OBB edges + name labels overlay, no state info per design)
    └─ steering.setup_episode(scene_image=jpeg_bytes)
         └─ stage_manager._lmps["scene_grounding"](instruction, image_bytes=jpeg)
              → OpenAI multimodal call → `{blocks_visible, ambiguous_resolutions}` dict
         └─ validate_grounding(dict)
              raise GroundingValidationError on malformed
         └─ format_scene_context(dict) → text block
         └─ compose_with_repair(composer, instruction, scene_context=text)
              composer reads `# Scene state (from VLM grounding ...)`
         └─ each affordance/avoidance LMP gets _scene_text=text (read at prompt build)
```

Key invariant: **only one vision call per episode**. Cached by image SHA-256
so identical scenes hit cache. Downstream LMPs stay text-only,
preserving their cache behavior.

**Decision NOT to inject deterministic `scene_obs` state into prompts**
(per user, 2026-05-19): scene_obs is privileged simulator info; using it
as composer/affordance context bypasses the VLM-as-perception premise
and isn't defensible for real-deployment claims. `format_scene_state`
helper stays in `voxposer/calvin_interface.py` for opt-in research but
has no callers in production.

---

## What worked vs what didn't

### Worked

- **VLM scene-grounding plumbing.** End-to-end, 100% schema-valid emissions
  on 90 P4 episodes. Correct disambig on `"it" → drawer`, `"the block from
  drawer" → blue_block`, etc. Cost-effective ($0.05 for 90 vision calls).
- **Phase 1 composer fixes** (text-only). Specific-color stage-0 rule
  eliminated ObjectResolutionError on push_into_drawer / stack_block.
  Cavity lift relaxation unlocked lift_blue_block_slider 1/3 → 3/3.
- **Phase 1.5 context preambles.** No measurable regression; informally
  better LMP reasoning per log audit.
- **`close_drawer` P4 polish** (`"Close the compartment." → "Close it."`).
  5/10 success rate, VLM correctly resolves to `drawer`.

### Didn't work / out of scope

- **gpt-4o for vision.** OpenAI safety filter refuses 5/8 scene-grounding
  calls with `"I can't help with identifying contents of the image."`
  Unusable as production target. gpt-5.4-mini works reliably (but with
  44.4% ambig accuracy — sub-perfect but functional).
- **Phase 3 prompt examples** (initial attempt). Adding 2 composer + 1
  affordance example targeting place_in_slider + cavity-lift regressed
  push_into_drawer. Reverted. Replaced with user-direction Phase 3 iter 3
  (drop scene_state injection entirely).
- **`place_in_slider` cavity geometry.** Stuck at 0/5 across canonical and
  0/1 in P4. The affordance interior of the slider cabinet needs
  primitive-aware EDT smoothing (out of Task 7 scope; future work).
- **Light tasks under P4 (`"Turn on the light."`)** — `turn_on_led` 0/3,
  `turn_off_led` 0/1, etc. The instruction is ambiguous between lightbulb
  and led, VLM resolution is inconsistent. Likely needs scene-state-aware
  disambig (which one is currently off?) — but we deliberately don't
  inject scene_state per architectural decision.

---

## Known residuals (filed for future work)

1. **`place_in_slider` cavity-aware affordance geometry.** The "outer
   edge of slider interior" affordance pattern attempted in Phase 3 iter
   1 had a y-axis sign bug (cavity is at HIGHER y from slider_handle, not
   lower). Fixing requires per-primitive EDT smoothing — orthogonal to
   VLM grounding.
2. **VLM hallucination on `blocks_visible`.** Phase 0 audit: 87.5% block
   accuracy; some scenes have the model voting "block X in drawer" when
   all blocks are on table. Doesn't affect downstream affordance (the
   composer's `get_affordance_map('a point at the center of red_block')`
   call reads LIVE positions, not VLM hallucination). Could affect ambig
   color choice in edge cases.
3. **Light disambig under P4 "Turn on the light."** Both lightbulb and
   led can be the target; VLM has no fixture-state input to choose. A
   future architecture could expose a `lights_state` summary from
   `scene_obs` SELECTIVELY (e.g., only for light-related tasks) while
   keeping the VLM-perception premise for blocks.

---

## Final file inventory

**New files:**
- `voxposer/scene_image.py` — annotated overhead frame renderer
- `voxposer/prompts/calvin/scene_grounding_prompt.txt` — grounding LMP system prompt
- `scripts/audit_scene_grounding.py` — Phase 0 audit harness
- `docs/refactor/task7_*.md` + `task7_phase0_audit.json`
- `conf/evaluation/_task7_phase{0..5}*.yaml` — per-phase canary configs

**Modified files:**
- `voxposer/lmp.py` — multimodal LLMBackend, scene_grounding factory, format_scene_context, GroundingValidationError
- `voxposer/calvin_interface.py` — `set_scene_context` / `_scene_context` + `format_scene_state` helper (unused per architecture decision but kept for opt-in)
- `envs/calvin.py` — `get_static_camera_matrices` helper
- `steering/stage_manager.py` — `scene_image` kwarg + grounding call site
- `steering/voxposer_steering.py` — `scene_image` kwarg pass-through + `lmp_interface` property
- `scripts/run_evaluation.py` + `scripts/run_experiment.py` — frame capture wiring
- `conf/steering/voxposer.yaml` — `scene_grounding.*` config block
- `voxposer/prompts/calvin/composer_prompt.txt` — Phase 1 fixes + Phase 1.5 preamble
- `voxposer/prompts/calvin/get_affordance_map_prompt.txt` + `get_avoidance_map_prompt.txt` — Phase 1.5 preambles
- `perturbed_language_annotations.json` — `close_drawer.P4` polish

**Untracked tooling (kept on disk):**
- `tmp/p4_validation/` — render + label tools used for P4 starting-condition filtering
- `tmp/p4_validation/renders/` — per-task scene PNGs (~140 files)

**Stripped:** none — Task 7 is purely additive.
