# Task 7 — VLM scene-image ingestion

**Owner:** planner → refactorer (after team-lead approval)
**Branch:** `refactoring`
**Task ID:** `#7` (status: in_progress — planning phase)
**Scope:** Add a separate `scene_grounding` LMP that ingests an overhead camera image at episode start, emits a structured dict describing scene state, and injects that dict as text context into composer + affordance/avoidance LMPs. Plus a text-only preflight iter for Task 4's two deferred composer fixes. Architecture decision locked: **B** (per team-lead 2026-05-19) — only `scene_grounding` is multimodal; downstream LMPs receive scene-state text and stay cacheable as before.

This is the final task. Biggest scope of the refactor — extends `LLMBackend` for OpenAI multimodal, adds a new LMP, updates 4 prompt files, adds new env wiring for image capture, and produces two acceptance evals (28×5 canonical + 28×3 P4-perturbed). 5-6 iters across one session. Expected wall-clock: 8-12 hours.

---

## 0. Task-status check (no split)

Task #7 in the task list is the consolidated "VLM scene-image ingestion" assignment with 4 expanded targets. No sub-split is needed — the architecture (separate grounding LMP feeding text-context into downstream LMPs) is a single design that addresses all 4 targets:

| Target | How architecture B addresses it |
|---|---|
| **(1) P4 perturbation disambiguation** ("the light" → LED vs lightbulb; "the block in the drawer" → which color) | Grounding LMP's `ambiguous_resolutions` field maps the P4 noun phrase to a canonical OBJECT_VOCAB token. Composer reads the resolved name when emitting stages. |
| **(2) Cavity-target value-map construction** (place_in_slider, stack_block) | Grounding LMP's `fixtures_state` + `blocks_visible` lets composer + affordance LMPs know whether the destination is a cavity interior (drawer_inside, slider_inside) vs a clear surface (table). Affordance LMP picks a reachable approach point accordingly. |
| **(3) Lift-from-cavity trajectory targets** (lift_*_slider/drawer regressions) | `blocks_visible` lets composer condition the 1-vs-2-stage lift decision on the source location. Cavity sources → relax to 1-stage (per Task-4 recommendation 2). Table sources → keep 2-stage. |
| **(4) Two deferred Task-4 composer fixes** (specific-color stage-0 rule + cavity lift relaxation) | Ship as a **text-only preflight iter** before VLM plumbing lands (Phase 1) so wins are attributable. VLM later refines per-scene where useful. |

Dependencies: blocked by #4 (✅ done — dwell tuning + diagnostics) and #5 (✅ done — Renderer Protocol). Status verified on README at plan-draft time.

---

## 1. Original brief (verbatim)

Task description (assigned by team-lead, 2026-05-19, expanding the user's `refactoring_tasks.md` kickoff line):

> "Final task. Add OpenAI multimodal VLM (gpt-5.4-mini / gpt-4o) to ingest overhead camera image at composer + affordance-map construction time. Four targets per user direction: (1) P4 perturbation disambiguation (color, LED/lightbulb), (2) cavity-target value-map construction (place_in_slider, stack_block), (3) lift-from-cavity trajectory targets (lift_*_slider/drawer regressions from 3b), (4) the two deferred Task 4 composer fixes (specific-color stage 0 rule + cavity lift relaxation) folded in if/where they interact with the VLM pipeline."

User kickoff line (`refactoring_tasks.md`, 2026-05-18 follow-up after the original refactor list):

> "we will need the VLM to ingest an image of the scene from the overhead camera to inform the construction of the value map. This should be an additional task after the previous tasks have been completed."

**Architecture decision (locked, team-lead 2026-05-19, Q1):** the "ingest at composer + affordance-map construction time" phrasing is honored by feeding scene-image-derived **text context** into both LMPs — not by making both LMPs literally multimodal. A separate `scene_grounding` LMP runs once at episode start, ingests the image, and emits a structured dict that both downstream LMPs read. Single vision call per episode; downstream cache behavior preserved.

**Hard constraints retained from prior tasks:**
- No silent fallback for vocab violations. `VocabValidationError` / `ObjectResolutionError` continue to hard-fail on exhaustion (3a + 3b discipline).
- No reading of `action_primitive_object_annotations.json` at inference (negative-grep test re-runs at acceptance).
- No structural edits to `steering/stage_manager.py` beyond image-plumbing kwargs.
- No structural edits to `policies/` (Task 2 territory).
- No new heavyweight deps. OpenCV + Pillow (already deps via Task 5) cover image capture + JPEG encoding.

---

## 2. Surveyed surface (KEEP / EXTEND / NEW / NEW-PROMPT)

Inventory of files this task touches. Net delta: +1 file (`scene_grounding_prompt.txt`), +~250 LoC (LLMBackend multimodal + new LMP + frame capture + prompt injection), -0 LoC (no deletes).

| Path | LoC | Role | Verdict |
|------|-----|------|---------|
| `voxposer/lmp.py::LLMBackend._call_openai` | ~40 (of 150) | Text-only `chat.completions.create`. `is_gpt5_family` parameter dispatch. | **EXTEND** — accept optional `image_bytes` (JPEG); when present, build `content` as list `[{"type":"text",...}, {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]`. Cache key extended with `image_hash` (SHA-256 of image bytes). |
| `voxposer/lmp.py::LLMBackend.generate` | ~50 | Cached text call wrapper. | **EXTEND** — accept optional `image_bytes`; thread through to `_call_openai`. Cache key gains image hash. |
| `voxposer/lmp.py::LMP.__call__` | ~40 | Generate prompt → backend.generate → execute → return. | **EXTEND** — accept optional `image_bytes` kwarg; thread through. Default `None` → text-only path unchanged. |
| `voxposer/lmp.py::compose_with_repair` | ~50 | Composer with vocab-linter re-prompt loop. | **EXTEND** — accept `scene_context: Optional[str]` kwarg (the pretty-printed grounding dict). Prepended to composer's prompt as a `# Scene state` block. The composer is text-only; only `scene_grounding` is vision. |
| `voxposer/lmp.py::DEFAULT_LMP_CONFIGS` | ~40 | Dict of LMP configs. | **EXTEND** — add `scene_grounding` entry: `{"prompt_fname": "scene_grounding_prompt", "stop": ["# Query:"], "has_return": True, "return_val_name": "ret_val", "maintain_session": False, "include_context": False}`. |
| `voxposer/lmp.py::setup_lmp` | ~50 | Build LMP hierarchy. | **EXTEND** — instantiate `scene_grounding` LMP, return it alongside composer + low-level LMPs. |
| `voxposer/prompts/calvin/scene_grounding_prompt.txt` | — | — | **NEW** (~80 LoC) — system prompt + 2-3 in-context examples for "look at image, emit structured dict." See §4.2 for exact contract. |
| `voxposer/prompts/calvin/composer_prompt.txt` | 252 | Composer prompt with vocab cheat-sheet + 11 in-context examples. | **EXTEND** — new section near top: "When `# Scene state` is provided, use it to (a) pick specific block color when instruction is ambiguous, (b) decide 1-vs-2-stage for lifts based on `blocks_visible` location, (c) resolve P4 ambiguities from `ambiguous_resolutions`." Plus 2 new vision-conditioned examples. Net: ~280 lines. **Phase 1 fixes** also live here: specific-color stage-0 rule + cavity lift relaxation. |
| `voxposer/prompts/calvin/get_affordance_map_prompt.txt` | 72 | Affordance code-generation prompt. | **EXTEND** — new section: "When `# Scene state` is provided, use `fixtures_state` to decide cavity-target approach (slider_inside → outer-edge approach point, not centroid)." 1 new example. Net: ~85 lines. |
| `voxposer/prompts/calvin/get_avoidance_map_prompt.txt` | 25 | Avoidance code-generation prompt. | **NO EDIT for v1.** Cavity-aware obstacle masking is future work (3b shelved it; needs primitive-aware EDT first). |
| `voxposer/prompts/calvin/parse_query_obj_prompt.txt` | 75 | Simple object-name resolver. | **NO EDIT.** Per Q2 — text-only with existing object context. |
| `steering/stage_manager.py::setup_episode` | ~85 | Composer + parser + stage 0 activation. | **EXTEND** — add `scene_image: Optional[bytes] = None` kwarg. When non-None: call `scene_grounding(scene_image=...)` first; format result as text scene-context string; pass via `compose_with_repair(scene_context=...)`. Cache the grounding output on the stage_manager for downstream affordance LMPs to read (via a method like `lmp_interface.set_scene_context(dict)`). |
| `steering/voxposer_steering.py::setup_episode` | ~15 | Forwards to stage_manager. | **EXTEND** — same kwarg plumb. |
| `voxposer/calvin_interface.py::CalvinLMPInterface` | ~600 | Object detection + helpers. | **EXTEND** — add `_scene_context: Optional[dict]` field + `set_scene_context(d)` setter. Affordance LMPs read this via the existing variable-vars reflection. |
| `scripts/run_evaluation.py::setup_voxposer_episode` | 15 | Pulls scene state + calls steering.setup_episode. | **EXTEND** — capture `env.render_high_res_static(W, H, fov=20)` → `_render_annotated_overhead(frame, detections)` → JPEG bytes → pass as `scene_image=`. |
| `scripts/run_experiment.py` | — | Mirror site. | **EXTEND** — same wiring shape. |
| `voxposer/scene_image.py` | — | — | **NEW** (~60 LoC) — `render_annotated_overhead(frame, detections, scale=1.0)` overlays OBB edges + name labels on the captured frame using `obb_world_corners` + project-to-camera math. Returns JPEG bytes. |
| `envs/calvin.py::render_high_res_static` | ~30 | Already exists (Task 5 reuses this). | **NO EDIT** — but needs to also return the camera intrinsics + extrinsics so `voxposer/scene_image.py` can project world OBBs to pixel coords. May need a sibling method `_get_static_camera_matrix()` exposing the projection. ~10 LoC addition. |
| `conf/steering/voxposer.yaml` | — | Steering config. | **EXTEND** — new keys: `scene_grounding.enabled: bool`, `scene_grounding.model: str` (default `gpt-4o`), `scene_grounding.image_width: int` (default 600), `scene_grounding.image_height: int` (default 600), `scene_grounding.image_fov: float` (default 20), `scene_grounding.annotate: bool` (default true), `scene_grounding.fallback_on_error: bool` (default true). |
| `voxposer/llm_cache.py::DiskCache` | — | Cache k/v store. | **NO EDIT** — image-hash is part of the cache key dict, no schema change needed. |
| `conf/evaluation/langsteer_primitive_object.yaml` | — | Eval config. | **NO EDIT for v1.** May add a per-iter cache override for Phase 0 audit (reverted at Phase 5 close, per 3a/3b convention). |

**Reuse from prior tasks:**
- Task 5: `env.render_high_res_static(...)` (already used by `VideoRecorder`). `get_all_detections()` returns OBBs in the shape `scene_image.py` needs.
- Task 3a: `compose_with_repair` repair-loop pattern. Apply same shape to grounding LMP — if grounding output has an invalid vocab token, re-prompt with a hint.
- Task 3b: `is_gpt5_family` dispatch in `LLMBackend._call_openai`. Reuse — the mixed-model strategy (gpt-4o for grounding, gpt-5.4-mini for composer + affordance) drops cleanly into the existing dispatch.

---

## 3. Target file layout post-refactor

```
voxposer/
├── lmp.py                                  # EXTENDED — multimodal LLMBackend, scene_grounding LMP factory, scene_context plumbing
├── scene_image.py                          # NEW (~60 LoC) — annotated overhead frame renderer
├── calvin_interface.py                     # EXTENDED — set_scene_context / _scene_context field
└── prompts/calvin/
    ├── composer_prompt.txt                 # EXTENDED — scene-context section + Phase 1 fixes
    ├── get_affordance_map_prompt.txt       # EXTENDED — scene-context section
    ├── get_avoidance_map_prompt.txt        # UNCHANGED (v1)
    ├── parse_query_obj_prompt.txt          # UNCHANGED
    └── scene_grounding_prompt.txt          # NEW (~80 LoC)

envs/
└── calvin.py                               # MINOR EXTEND — expose static camera matrix

steering/
├── stage_manager.py                        # EXTENDED — scene_image kwarg plumb, grounding LMP call site
└── voxposer_steering.py                    # EXTENDED — same kwarg plumb

scripts/
├── run_evaluation.py                       # EXTENDED — capture+annotate frame in setup_voxposer_episode
└── run_experiment.py                       # EXTENDED — same wiring shape

conf/
└── steering/voxposer.yaml                  # EXTENDED — scene_grounding.* keys

docs/refactor/
├── task7_plan.md                           # this file
├── task7_log.md                            # NEW — iter log
├── task7_phase0_audit.json                 # NEW — Phase 0 audit deliverable
└── task7_vlm_grounding.md                  # NEW — final scribe doc
```

---

## 4. Design spec

### 4.1 LLMBackend multimodal extension

```python
def generate(
    self,
    prompt: str,
    stop: list,
    image_bytes: Optional[bytes] = None,  # NEW
) -> str:
    cache_key = {
        "provider": self._provider,
        "model": self._model,
        "prompt": prompt,
        "temperature": self._temperature,
        "max_tokens": self._max_tokens,
    }
    if image_bytes is not None:
        cache_key["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()  # NEW
    # ... existing cache lookup ...
    if self._provider == "anthropic":
        # OpenAI-only for v1; Anthropic image path deferred.
        if image_bytes is not None:
            raise NotImplementedError("Anthropic multimodal deferred")
        result = self._call_anthropic(client, context, query, stop)
    else:
        result = self._call_openai(client, context, query, stop, image_bytes=image_bytes)
    # ... existing cache write ...
```

In `_call_openai`, when `image_bytes` is provided:

```python
b64 = base64.b64encode(image_bytes).decode("ascii")
user_content = [
    {"type": "text", "text": f"I will give you context code, then a query to complete.\n\nContext:\n```\n{context}\n```\n\nComplete this:\n{query}"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
]
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_content},
]
```

Other model dispatch behavior (`is_gpt5_family` parameter handling, client-side stop truncation) is unchanged — applies to both text-only and multimodal calls.

**Cache key change is additive.** Text-only calls produce the same cache key as today (no `image_sha256`), so existing 3b/Task-5 cache entries remain valid. Vision calls get a new cache key namespace.

**Model locked: gpt-5.4-mini for grounding (team-lead 2026-05-19, post-Phase-0).** Phase 0 audit revealed gpt-4o has a safety-refusal pattern that hard-blocks production use (~5/8 refusals on robotic manipulation scene queries). gpt-5.4-mini passed cleanly (100% schema validity, no refusals). The multimodal extension is tested only on the gpt-5.4-mini code path for v1; gpt-4o multimodal stays as a fallback that we don't exercise. The cfg key `scene_grounding.model` defaults to `gpt-5.4-mini`; do not change without re-running the safety-refusal canary. (Anthropic claude as an alternative vision backend is §9 future work.)

### 4.2 `scene_grounding` LMP — prompt + output contract

**Schema narrowed post-Phase-0 (team-lead 2026-05-19, pivot (1)):** original schema had three top-level keys; `fixtures_state` is dropped. Phase 0 audit showed both models <70% on the full schema, but gpt-5.4-mini hit 87.5% on `blocks_visible` and 44.4% on `ambiguous_resolutions` with 100% schema validity — so the narrow schema clears the production bar. Deterministic state (drawer open/closed, slider position, lightbulb/LED on/off, gripper state) flows through `scene_obs` via `CalvinLMPInterface` — VLM only handles visual disambiguation where ground truth isn't directly available (block location buckets + ambiguous-phrase resolution). Cleaner architecture: VLM ≡ "what color is where, what does the noun phrase refer to"; scene_obs ≡ "discrete fixture/joint states."

**Output schema** (Python dict, JSON-serializable, 2 top-level keys):

```python
{
    "blocks_visible": {
        # Each known block → location bucket. Buckets: "table" | "drawer_inside" |
        # "slider_inside" | "held" | "absent" (if block not visible in frame).
        "red_block": "table",
        "blue_block": "drawer_inside",
        "pink_block": "slider_inside",
    },
    "ambiguous_resolutions": {
        # Any phrase from the original instruction that VLM disambiguates.
        # Empty dict when instruction is unambiguous. Keys are the original
        # ambiguous phrase verbatim; values are OBJECT_VOCAB tokens.
        "the light": "led_button",
        "the block in the drawer": "blue_block",
    },
}
```

Deterministic state lookup (when composer or affordance LMP needs fixture state) reads `scene_obs` slices via existing `CalvinLMPInterface` helpers — `scene_obs[24]` is drawer joint position, `scene_obs[25]` is slider, etc. (per CALVIN's documented scene-state vector). No VLM call required for state.

**Prompt structure** (`scene_grounding_prompt.txt`, ~80 lines):

1. Header: explain the task. The LMP receives an annotated overhead image plus the task instruction. It emits a Python dict with the three top-level keys. Vocabulary constraint: bucket values are fixed; block names are OBJECT_VOCAB tokens.
2. Image annotation spec: "Each object in the image has an OBB outlined in its category color (red/blue/pink for blocks, orange for slider/drawer, gold for lights). Object name labels are at the OBB centers."
3. 2-3 in-context examples covering: (a) unambiguous canonical instruction (P1-P3 case — empty `ambiguous_resolutions`), (b) ambiguous P4 with multiple-object disambiguation, (c) cavity scene with a held block.
4. Output discipline: emit ONLY `ret_val = {...}` — no commentary.

The LMP uses `compose_with_repair`-style hard-fail semantics: if the emitted dict has invalid bucket values or non-OBJECT_VOCAB block names, re-prompt up to 2x. On exhaustion, raise a new `GroundingValidationError` (subclass of ValueError) that propagates to the runner — episode fails fast with a clear log.

### 4.3 Annotated overhead frame renderer (`voxposer/scene_image.py`)

```python
def render_annotated_overhead(
    rgb_frame: np.ndarray,        # (H, W, 3) uint8 from env.render_high_res_static
    detections: list[Observation], # from CalvinLMPInterface.get_all_detections()
    camera_matrix: np.ndarray,    # (4, 4) view-projection from env
    obj_colors: dict[str, str] = _OBJ_COLORS,
) -> bytes:
    """Overlay OBB edges + name labels on the captured frame; return JPEG bytes."""
    out = rgb_frame.copy()
    for det in detections:
        if det.get("obb_center_world") is None:
            continue
        corners_world = obb_world_corners(
            det["obb_center_world"], det["obb_size"], det["obb_rotation"]
        )
        # Project corners to pixel space using camera_matrix.
        pixel_corners = _project_world_to_pixel(corners_world, camera_matrix, W, H)
        color = _obb_color_for(det["name"], obj_colors)
        _draw_obb_edges(out, pixel_corners, color)
        _draw_label(out, _centroid(pixel_corners), det["name"], color)
    return _encode_jpeg(out, quality=85)
```

Total ~60 LoC. Uses OpenCV (already a dep) for line drawing + JPEG encoding. Pillow (already a dep) is a fallback if cv2 unavailable. The `_project_world_to_pixel` helper is ~15 LoC of standard view-projection math.

The `camera_matrix` exposure on `envs/calvin.py` is a small addition (~10 LoC): a `_get_static_camera_matrix(width, height, fov)` method that returns the PyBullet view-projection matrix used by `render_high_res_static`. PyBullet exposes these via `getDebugVisualizerCamera()` / explicit `computeViewMatrix` + `computeProjectionMatrixFOV` calls.

**Annotation discipline (locked, team-lead 2026-05-19, refinement to Q3):** the overlay carries **static object identities + geometric extents only** — OBB edges + name labels. State information (drawer open/closed, slider position, light on/off, gripper closed/open) is **deliberately NOT overlaid.** The VLM must infer state from the raw pixels. This preserves the grounding LMP's job ("look at the scene and report what's happening") and prevents the annotation pipeline from accidentally pre-empting the VLM's work via labels like "drawer (OPEN)". Concretely, `_draw_label` writes only `det["name"]` (canonical OBJECT_VOCAB token), never any state-derived field from the Observation dict. This is enforced by the renderer's signature — no state argument exists.

### 4.4 Scene-context injection into downstream LMPs

The grounding dict gets pretty-printed and injected as a comment block prepended to the composer + affordance prompts:

```python
def format_scene_context(grounding: dict) -> str:
    """Pretty-print grounding dict as comment block for prompt injection."""
    lines = ["# Scene state (from VLM grounding at episode start):"]
    for k, v in grounding.items():
        if isinstance(v, dict) and v:
            lines.append(f"# {k}:")
            for k2, v2 in v.items():
                lines.append(f"#   {k2}: {v2!r}")
        else:
            lines.append(f"# {k}: {v!r}")
    lines.append("")
    return "\n".join(lines)
```

`stage_manager.setup_episode` does:

```python
if scene_image is not None:
    grounding = self._lmps["scene_grounding"](
        instruction, image_bytes=scene_image
    )
    self._lmp_interface.set_scene_context(grounding)  # affordance LMPs read this
    scene_ctx_text = format_scene_context(grounding)
else:
    scene_ctx_text = None
    # legacy text-only path — composer + affordance LMPs work as today

result = compose_with_repair(
    self._lmps["composer"], instruction, scene_context=scene_ctx_text,
)
```

`compose_with_repair` threads `scene_context` into each composer call (initial + repairs). The affordance LMPs, when called from composer-emitted code via `get_affordance_map(...)`, read `self._lmp_interface._scene_context` directly and prepend the formatted block to their own prompt build.

This keeps the contract simple: ONE shared dict, two read paths (composer-via-arg, affordance-via-interface). Cache keys stay text-only for both downstream paths (grounding dict serialization is deterministic per scene, so it's part of the text content that gets hashed).

**Parallel scene-state injection (added 2026-05-19, user direction):** alongside the VLM grounding dict, downstream LMPs also receive a deterministic scene-state text block sourced from `scene_obs` via the new `format_scene_state` helper (§4.5). The two injections are **independent**:
- **VLM grounding** (this section): visual disambiguation only — `blocks_visible` location buckets + `ambiguous_resolutions` for noun phrases. Comes from the vision-LMP at episode start.
- **Scene state** (§4.5): deterministic ground-truth — fixture open/closed, light on/off, slider position, gripper state, block-by-fixture grouping. Comes from `scene_obs` slices, no LLM in the loop.

The composer/affordance prompts see both as text blocks at the top of their prompts (after the new preambles in §4.6). For ambiguous prompts like "close it", the composer reads scene-state to find what's open ("drawer state: open"), then resolves the pronoun.

### 4.5 Deterministic scene-state injection (`format_scene_state` helper)

**Purpose:** give composer + affordance LMPs structured text access to deterministic CALVIN scene state that doesn't need a vision call to resolve. The existing 3a/3b composer prompts encode some state-dependent rules inline (e.g., the lift-up offset for graspable blocks), but state-sensitive disambiguation cases like "close it" (only one openable fixture is currently open) need runtime data, not prompt examples.

**New helper** in `voxposer/calvin_interface.py`:

```python
def format_scene_state(scene_obs: np.ndarray, robot_obs: np.ndarray) -> str:
    """Format CALVIN scene_obs + robot_obs as a fixed-schema text block.

    Output is deterministic given (scene_obs, robot_obs); safe to cache in
    the composer/affordance LMP prompt as a normal text block.

    Schema (fixed sections, fixed order so prompts can reference deterministically):
        # Scene state (deterministic, from scene_obs):
        # drawer: {open|closed}
        # slider: {left|right|center}
        # lightbulb: {on|off}
        # led: {on|off}
        # gripper: {open|closed} (width=N.NNNm)
        # blocks:
        #   red_block: {on_table|in_drawer|in_slider|held|elsewhere}
        #   blue_block: ...
        #   pink_block: ...
    """
    # ~25 LoC: slice scene_obs at the documented indices, threshold the
    # joint positions for open/closed binarization, threshold gripper width
    # against HELD_BLOCK_GRIPPER_CLOSED_MAX_WIDTH (already a constant in
    # calvin_interface), spatial-test each block centroid against fixture
    # AABBs to assign location bucket, return formatted string.
```

**Threshold conventions** (so the LMP output is predictable across episodes):
- `drawer: open` when `scene_obs[24] > 0.05` (CALVIN's joint convention); else `closed`.
- `slider: left|right|center` from `scene_obs[25]` thresholds.
- `lightbulb/led: on|off` from `scene_obs[26]`, `scene_obs[27]` (state booleans).
- `gripper: closed` when `robot_obs[6] < 0.07` (matches `HELD_BLOCK_GRIPPER_CLOSED_MAX_WIDTH`).
- `block: in_drawer` when its centroid is inside the drawer fixture's AABB (per `_detect_fixture` returns); else `in_slider` / `on_table` / `held` (latter via existing `_get_held_block` heuristic).

**Caching note:** because the output is deterministic given the scene/robot obs, the composer and affordance LMP prompts that include this block hash identically across episodes with the same starting state. No LLM call wasted; cache behavior unchanged.

**Injection point:** `stage_manager.setup_episode` calls `format_scene_state(scene_obs, robot_obs)` (always — gated by no toggle, since it's free text data) and passes the resulting string to `compose_with_repair(scene_state=...)`. The composer prepends it as a comment block above the existing `# Scene state (from VLM grounding ...)` block. Affordance LMP reads it the same way as the VLM grounding (via `lmp_interface._scene_state` field, set alongside `_scene_context`).

### 4.6 Context preambles (3 active prompts)

**Purpose:** the existing prompts dive straight into stage-tuple schema or code examples; they don't explain to the LMP what the larger system is doing or how its emission fits in. Adding a 5-10 line preamble at the top of each active prompt locks the LMP's understanding of its role + how to leverage the scene context blocks injected below.

**Preamble template** (~5-10 lines, vary slightly per prompt for tone but keep the structure):

```
# === SYSTEM CONTEXT ===
# You are the <LMP_NAME>. Your job is to <ROLE>.
# Upstream of you: <WHAT_PRODUCES_YOUR_INPUT>.
# Downstream of you: <WHAT_CONSUMES_YOUR_OUTPUT>.
# Scene context is provided below in two blocks:
#   (1) deterministic scene state from scene_obs (fixture/light/block state)
#   (2) visual scene grounding from a vision LM (block locations + ambiguous phrase resolutions)
# Use both to make scene-aware decisions:
#   - For ambiguous instructions ("close it", "the light"), read the scene state to find what's currently open / which light is the target.
#   - For block-location reasoning (lift-from-cavity etc.), use blocks_visible from the grounding dict.
# ======================
```

**Per-prompt fills:**

| File | LMP role | Upstream | Downstream |
|------|----------|----------|------------|
| `composer_prompt.txt` | Composer | User instruction + scene context | Affordance/avoidance LMP code-emission |
| `get_affordance_map_prompt.txt` | Affordance map generator | Composer-emitted query string + scene context | Steering trajectory guidance |
| `get_avoidance_map_prompt.txt` | Avoidance map generator | Composer-emitted query string + scene context | Steering obstacle guidance |

**No preamble** on `parse_query_obj_prompt.txt` or the new `scene_grounding_prompt.txt`:
- `parse_query_obj` stays text-only with existing object-context (per Q2 — no scope expansion).
- `scene_grounding` already has its own task-specific header (§4.2); a generic preamble would conflict.

**Net per-prompt LoC delta:** +5-10 lines preamble; no in-context-example changes from §4.6 alone (Phase 1's composer fixes in §4.7 layer on top). Composer prompt: 252 → ~262. Affordance: 72 → ~82. Avoidance: 25 → ~35.

### 4.7 Phase 1 (preflight) — Task-4 composer fixes (text-only)

These ship BEFORE any VLM plumbing lands so wins are attributable. Both are pure prompt edits in `composer_prompt.txt`:

**Fix 1: Specific-color stage 0 rule.** Add to the vocab cheat-sheet near line 22:

> Stage 0 object MUST be a specific block color (`red_block`, `blue_block`, or `pink_block`) — never generic `block`. Stage ≥1 may use `block` only when a prior stage's grasp produced a held block (the held-block fallback in `_detect_object` resolves it).

Plus update the existing `push_into_drawer` and `place_in_slider` examples to demonstrate specific colors at stage 0.

**Fix 2: Cavity lift relaxation.** Modify the existing "LIFTS MUST EMIT 2 STAGES" rule:

> LIFTS FROM TABLE MUST EMIT 2 STAGES: stage 1 = grasp at block centroid, stage 2 = grasp 15cm above block start.
>
> LIFTS FROM CAVITY (slider, drawer) MAY EMIT 1 STAGE: cavity geometry already constrains the lift trajectory; an explicit lift-up stage can over-specify and regress.

Add an in-context example showing a 1-stage lift from drawer with the corresponding instruction.

**Phase 1 canary:** 4 tasks × 3 ep = 12 ep. Tasks: `stack_block`, `push_into_drawer`, `lift_blue_block_slider`, `lift_red_block_drawer`. Pass criteria:
- `push_into_drawer`: composer emits specific-color (no ObjectResolutionError). ≥1/3 success (was 0/2 in Task-4 canary).
- `stack_block`: composer emits specific-color. ≥1/3 success (was 0/2 in Task-4 canary).
- `lift_blue_block_slider`: composer emits 1-stage (per relaxed rule). Recovers toward Phase 0 baseline 5/5.
- `lift_red_block_drawer`: composer emits 1-stage. Recovers toward 3a's 3/5.

Phase 1 is independently shippable — if these fixes alone close half the gap, that informs Phase 2 prompt scope. The text-only fixes ship to main (via the refactoring branch) BEFORE the VLM plumbing iter starts.

---

## 5. Iteration plan (5 phases, refactorer commits per phase)

Each phase ends in a working repo (eval-runnable, lint-clean). Each phase has explicit ship/rollback criteria. Audit-first protocol carried forward from 3a/3b.

### Phase 0 — Vision model A/B audit (read-only, blocking) ✅ DONE

**Outcome (2026-05-19):** schema narrowed + model locked + pivot (1) approved. See `docs/refactor/task7_phase0_audit.json` for raw data.

- **gpt-4o:** ~5/8 safety-refusal pattern on robotic-manipulation scene queries — production-blocking. Even when it answered, output quality on the narrow schema was comparable to gpt-5.4-mini. **Disqualified.**
- **gpt-5.4-mini:** 100% schema-valid; 87.5% accurate on `blocks_visible`; 44.4% accurate on `ambiguous_resolutions`; ~0% on `fixtures_state` (the model conflated open/closed/left/right buckets). **Locked for production grounding.**
- **Pivot (1) — schema narrowed:** dropped `fixtures_state` from the grounding LMP output (now 2 keys: `blocks_visible` + `ambiguous_resolutions`). Deterministic fixture state flows through `scene_obs` via `CalvinLMPInterface` — see §4.2 for the cleaner architecture rationale.
- **Cost savings:** the full task budget drops from ~$10 to ~$1 total, because (a) gpt-5.4-mini vision is ~10× cheaper than gpt-4o, and (b) the narrowed schema halves prompt + output token counts.

**Decision rule for future Phase 0-style runs:** ≥70% accuracy on every retained schema field. The 44.4% `ambiguous_resolutions` rate is below that bar but not a blocker — see §7 risk row + §9 future-work options. Phase 5 acceptance gates measure end-to-end P4 success, not isolated grounding-field accuracy, so the residual is tracked downstream.

**Phase 0 budget actual:** ~$0.30 (gpt-5.4-mini calls cost less than projected). Phase 0 deliverable: `task7_phase0_audit.json` shipped.

### Phase 1 — Text-only prompt updates (preflight)

**Goal:** ship Task-4's two deferred composer fixes + the §4.6 context preambles as pure prompt edits. Validate independently before VLM plumbing.

**Steps:**
1. **Add §4.6 context preambles** to 3 prompts: `composer_prompt.txt`, `get_affordance_map_prompt.txt`, `get_avoidance_map_prompt.txt`. 5-10 lines each at the top; per-prompt role + upstream/downstream fills per the §4.6 table. No preamble on `parse_query_obj_prompt.txt` or `scene_grounding_prompt.txt`.
2. **Apply Task-4 composer fixes** (§4.7) to `composer_prompt.txt`: specific-color stage-0 rule + cavity lift relaxation + updated examples.
3. Canary: 4 tasks × 3 ep (stack_block, push_into_drawer, lift_blue_block_slider, lift_red_block_drawer). Cache: fresh `/tmp/task7_phase1_canary/` (composer + affordance + avoidance prompts all change; old cache invalid).
4. **Broader regression sweep:** because preambles affect ALL 3 active prompts, run a 6-task × 2 ep secondary canary on previously-passing tasks (open_drawer, close_drawer, turn_on_led, push_red_block_right, push_pink_block_right, lift_pink_block_drawer) to catch any preamble-induced regressions outside the Task-4-targeted set.
5. Ship/rollback: per Phase 1 criteria below. Granular rollback: if regressions trace to one specific prompt's preamble (e.g., the avoidance preamble breaks something), revert that preamble alone and ship the rest.

**Phase 1 acceptance:**
- **Task-4 canary:** at least 2 of 4 canary tasks improve, none regress below their Task-4 canary value. Composer emits 0 ObjectResolutionError on canary.
- **Regression sweep:** all 6 secondary-canary tasks within ±1 episode of their 3b baseline. (Looser than hard rule — sample size is small.)
- **Lint/smoke:** ruff clean. Composer + affordance + avoidance prompts still produce parseable output on a single-ep open_drawer smoke.

**Phase 1 budget:** 12 ep (primary) + 12 ep (secondary) × ~5s each + composer + 1-3 affordance/avoidance calls per ep = ~$0.20 (text-only, gpt-5.4-mini).

### Phase 2 — VLM plumbing + scene-state helper (flag-gated)

**Goal:** all the new infrastructure, no behavior change to existing eval paths yet. Scene-state injection ships unconditionally; VLM plumbing gated on the new cfg toggle.

**Steps:**
1. **`format_scene_state` helper** (`voxposer/calvin_interface.py`): ~25 LoC per §4.5 spec. Threshold conventions documented inline. Plus a `set_scene_state(s: str)` setter on `CalvinLMPInterface` that affordance LMPs read from at code-gen time. Ships ALWAYS — no toggle (deterministic text data, no LLM call cost).
2. **LLMBackend extension** (`voxposer/lmp.py`): add `image_bytes` kwarg through `generate` + `_call_openai`. Cache key + base64 + content-list shape. ~30 LoC.
3. **scene_grounding LMP** (`voxposer/lmp.py` + `voxposer/prompts/calvin/scene_grounding_prompt.txt`): new prompt + `DEFAULT_LMP_CONFIGS` entry + `setup_lmp` instantiation. Returns the **2-key** dict (`blocks_visible` + `ambiguous_resolutions`, per pivot (1)). Wrapped in `compose_with_repair`-style re-prompt loop → `GroundingValidationError` on exhaustion.
4. **Scene image renderer** (`voxposer/scene_image.py`): `render_annotated_overhead(rgb, detections, camera_matrix)` → JPEG bytes. Plus the `envs/calvin.py::_get_static_camera_matrix(...)` helper. Annotation discipline per §4.3: OBB edges + name labels only, NO state info.
5. **Frame capture wiring** (`scripts/run_evaluation.py` + `scripts/run_experiment.py`): in `setup_voxposer_episode`, capture `env.render_high_res_static(W,H,fov)` + camera matrix → `render_annotated_overhead(...)` → pass as `scene_image=` to `steering.setup_episode(...)`. Gated on `cfg.steering.scene_grounding.enabled`.
6. **stage_manager plumbing** (`steering/stage_manager.py`):
   - `scene_image` kwarg → call grounding LMP → store result via `lmp_interface.set_scene_context(d)` (when enabled).
   - **Always**: call `format_scene_state(scene_obs, robot_obs)` → store via `lmp_interface.set_scene_state(s)` → format text → pass to `compose_with_repair(scene_state=...)`.
   - The composer prepends scene_state above scene_context (visible deterministic state before vision-derived disambig). Affordance LMPs read both from `lmp_interface`.
7. **voxposer_steering plumbing**: same kwarg pass-through.
8. **Config keys** (`conf/steering/voxposer.yaml`): `scene_grounding.{enabled, model, image_width, image_height, image_fov, annotate, fallback_on_error}`. No new key for scene_state (always on).

**Phase 2 canary:** `cfg.steering.scene_grounding.enabled=false` keeps eval behavior byte-identical to post-Phase-1 state. `cfg.steering.scene_grounding.enabled=true` on 1 task × 1 ep produces a grounding-dict log line + composer prompt with `# Scene state` block visible in DEBUG logs.

**Phase 2 acceptance:**
- ruff clean.
- `enabled=false` smoke: open_drawer 1/1 success, no scene_grounding LMP instantiated, no image capture (Task-5 video path still works). **scene_state block IS injected** even with VLM off (always-on per §4.5) — composer log shows `# Scene state (deterministic, from scene_obs)` block but no `# Scene state (from VLM grounding)` block.
- `enabled=true` smoke: grounding dict emitted with **exactly 2 top-level keys** (`blocks_visible` + `ambiguous_resolutions`, per pivot (1) — no `fixtures_state`). Composer log shows both blocks: deterministic scene_state first, then VLM scene_context. Eval still produces a stage list (correctness not yet measured — that's Phase 4).
- Schema validation: grounding LMP wrapped in `compose_with_repair`-style re-prompt loop; on third bad emission, raises `GroundingValidationError`. Smoke confirms the wrapper fires when an obviously malformed test-injected output is fed back.
- **`format_scene_state` smoke** (independent of VLM): inject 3 known scene_obs vectors (drawer-open + light-on; drawer-closed + light-off; gripper-holding-block) and confirm the helper emits the expected text — drawer state, slider state, light states, gripper width, block-by-fixture grouping all match ground truth.
- No regression on the Phase 1 ship state.

**Phase 2 budget:** integration testing only. ~5 vision calls × ~$0.022 ≈ $0.10.

### Phase 3 — Downstream prompt injection (composer + affordance)

**Goal:** wire the grounding dict into composer + affordance prompts so they actually USE the scene context.

**Steps:**
1. **Composer prompt** (`voxposer/prompts/calvin/composer_prompt.txt`): add the "When `# Scene state` is provided" section per §2 row + 2 new vision-conditioned examples (one P4 disambiguation, one cavity lift).
2. **Affordance prompt** (`voxposer/prompts/calvin/get_affordance_map_prompt.txt`): add a section on using `fixtures_state` for cavity-aware approach points + 1 new example (place_in_slider stage 2: outer-edge approach, not centroid).
3. **`format_scene_context`** helper in `voxposer/lmp.py`: pretty-printer per §4.4. Called from `stage_manager.setup_episode` for composer + auto-injected by affordance LMP via `self._lmp_interface._scene_context` read.

**Phase 3 canary:** same 4 task families as Phase 1 + `place_in_slider` × 3 ep each = 15 ep. Compare scene-grounding-enabled vs Phase 1 ship state.

**Phase 3 acceptance:**
- ≥1/3 success on place_in_slider (currently 0/5 across all evals).
- No regression on the Phase 1 canary tasks vs their Phase 1 results.
- Composer prompt log shows the scene-context block is actually read (per the new examples' phrasing).

**Phase 3 budget:** 15 ep × 1 grounding call + 1-2 composer calls + ~3 affordance calls = ~25 vision calls + ~50 text calls. **~$0.60.**

### Phase 4 — Targeted canary on 4 task families

**Goal:** verify each of the 4 targets shows measurable progress before committing to the full eval.

**Steps:**
1. 6 tasks × 3 ep = 18 ep canary: `push_into_drawer`, `place_in_slider`, `stack_block`, `lift_blue_block_slider`, `lift_red_block_drawer`, `turn_on_lightbulb` (a P4-disambig representative).
2. Run with full scene_grounding stack enabled. Compare to 3b final v3 baseline.

**Phase 4 acceptance:**
- ≥3 of the 6 canary tasks improve vs 3b baseline by ≥1 episode.
- 0 tasks regress to 0 success when their 3b baseline was ≥1/5 (hard rule precursor — narrower window pre-full-eval).
- Cost tracking on track: total Phase 4 spend <$2.

If Phase 4 misses, halt before Phase 5 and either iterate on prompts (canary again, max 2 iter cycles) or escalate to team-lead for scope refinement.

### Phase 5 — Final eval (28×5 canonical + 28×3 P4)

**Goal:** the formal acceptance evals.

**Steps:**
1. **28×5 canonical** with scene_grounding enabled. Same seeds + sampled_episodes.json as 3b iter-5 baseline.
2. **28×3 P4 perturbed** = 28 tasks × 3 ep × P4 instruction variant = 84 ep. Same seed file. P4 variants from `perturbed_language_annotations.json`.
3. Both runs hit hard rule check: no task with 3b baseline ≥3/5 drops below 1/5.

**Phase 5 acceptance gates (locked, team-lead Q9):**

| Gate | Type | Criterion |
|------|------|-----------|
| **Primary (a)** | P4 perturbed | 28×3 P4 sub-eval: ≥80% of P4 tasks emit valid stages (vs current ~57% per 3b §2.1). |
| **Primary (c) — cavity** | Targeted | `place_in_slider` 0/5 → ≥1/5. |
| **Primary (c) — stack** | Targeted | `stack_block` 1/5 → ≥2/5. |
| **Primary (c) — lift** | Targeted | ≥1 task in `lift_*_slider`/`lift_*_drawer` family recovers to ≥2/5 (e.g., `lift_red_block_drawer` 1/5 → ≥2/5). |
| **Tracked (b)** | Canonical | 28×5 canonical pass rate ≥ 76/140 = 54.3%. |
| **Hard rule** | Regression | No task with 3b baseline ≥3/5 drops below 1/5. |

Pass on Primary (a) + (c) + Hard rule = SHIP. Tracked (b) is an aspiration, not a gate.

**Phase 5 budget:** (28×5 = 140 ep) + (28×3 P4 = 84 ep) = 224 ep × 1 grounding call + ~3 text LMP calls = ~$5-7 vision + ~$0.50 text. **~$6-8 total.**

### Iter cadence summary

| Phase | What ships | Budget | Time est |
|-------|-----------|--------|----------|
| Phase 0 ✅ | audit only (shipped 2026-05-19) | ~$0.30 actual | done |
| Phase 1 | Task-4 deferred fixes + **§4.6 preambles for 3 prompts** + secondary regression sweep | ~$0.20 | 1-2 hours |
| Phase 2 | VLM plumbing (flag-off) + **§4.5 `format_scene_state` helper (always-on)** | ~$0.05 | 2-3 hours |
| Phase 3 | downstream injection (composer/affordance read both scene_state + scene_context) | ~$0.10 | 2-3 hours |
| Phase 4 | 6-task canary | ~$0.20 | 1-2 hours |
| Phase 5 | final eval | ~$0.50 | 2-3 hours |
| **Total** | | **~$1.30** (gpt-5.4-mini grounding locked) | **8-13 hours remaining** |

Budget collapse vs original $10-15 plan = (a) gpt-5.4-mini locked instead of gpt-4o (~10× cheaper per vision call), and (b) narrowed 2-key schema halves both input prompt tokens and output tokens. The two scope additions (context preambles + scene_state helper) add ~$0.10 of additional canary spend; the helper itself is free (no LLM call).

---

## 6. Validation

### Per-phase smoke checks

| Phase | Smoke check |
|-------|-------------|
| 0 ✅ | `task7_phase0_audit.json` shipped; pivot (1) approved; gpt-5.4-mini locked; schema narrowed to `blocks_visible` + `ambiguous_resolutions`. |
| 1 | 4×3 primary canary + **6×2 secondary regression canary** (preamble safety); composer log shows 0 ObjectResolutionError + 0 generic-`'block'` emissions; all 3 active prompts (composer/affordance/avoidance) have §4.6 preambles + still produce parseable output. |
| 2 | `enabled=false` smoke: open_drawer 1/1, no grounding LMP touched, **but `# Scene state (deterministic, from scene_obs)` block still appears in composer prompt** (always-on). `enabled=true` smoke: grounding dict in log with **2 top-level keys** (no `fixtures_state`); composer prompt has both scene_state + scene_context blocks; no crashes. `GroundingValidationError` fires on malformed-output canary. `format_scene_state` 3-vector smoke matches ground truth. |
| 3 | 5-task × 3 ep canary; place_in_slider ≥1/3; ruff clean. |
| 4 | 6×3 canary; ≥3 tasks improve vs 3b baseline; total spend < $2. |
| 5 | Final 28×5 + 28×3 P4 hits acceptance gates per §5. |

**VLM responsibility scoping (post-pivot-1):** the grounding LMP is responsible for `blocks_visible` + `ambiguous_resolutions` only. State info (drawer position, light state, slider position, gripper state) is **NOT** a grounding-LMP responsibility — it flows through `scene_obs` deterministically. State changes during an episode therefore do not trigger grounding re-runs; the one episode-start vision call is sufficient. This narrows what we can blame the VLM for and what we can blame the existing scene-state plumbing for when residuals surface.

### Hard-rule regression check (Phase 5)

For every task with 3b iter-5 success rate ≥ 3/5: post-Task-7 success rate must be ≥ 1/5. This is the same regression floor used in 3a, 3b, and prior. Compute from `task7_phase5_canonical.json` after the run; fail and rollback if any task violates.

### Negative leakage grep (Phase 5)

```bash
grep -rn "action_primitive_object_annotations" voxposer/ steering/ policies/ scripts/
```

Must return 0 hits in production paths. Same check as 3a/3b.

### Cost accounting

`task7_log.md` per-phase entry includes: `vision_calls`, `text_calls`, `cost_estimate_usd`. Final total must be ≤ $15 (architecture B ceiling, with audit and re-runs). If exceeded, escalate before continuing.

---

## 7. Risks & gotchas

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Phase 0 reveals both models <70% accurate.** | Medium | Phase 1 still ships independently (text-only). Phase 2-5 paused; escalate to team-lead for prompt-design iteration or alternative grounding strategy. |
| **PyBullet camera matrix exposure breaks `render_high_res_static`.** | Low | The new `_get_static_camera_matrix` helper is purely additive; existing `render_high_res_static` is left unchanged. Smoke test Phase 2's `enabled=false` path to confirm. |
| **Image annotation projection math has off-by-one or wrong-axis bug.** | Medium | Phase 0 audit ships annotated images — refactorer eyeballs the 8 audit scenes' annotations before any model call. Sanity check: red_block label is on the red block, not 100px off. |
| **gpt-4o vision rate limits trip during 28×3 P4 run.** | Low-medium | Per-episode grounding call has exponential backoff (already in `LLMBackend.generate`). Plus the run is checkpointed per-task; rate-limit-induced fail retries cleanly. |
| **Phase 1 lift relaxation regresses lift_*_table tasks.** | Low | Phase 1 acceptance check includes "0 regressions below Task-4 canary value." If lift_red_block_table dips, ship only the specific-color rule + leave the lift rule unchanged. |
| **Image cache invalidation surprises.** | Low | Image hash in cache key is additive (no schema break). Phase 2 smoke confirms text-only paths still hit the existing cache. |
| **Grounding LMP emits malformed dict that breaks composer.** | Medium | `GroundingValidationError` (new) hard-fails the episode with a clear log, same as `VocabValidationError` / `ObjectResolutionError`. `cfg.steering.scene_grounding.fallback_on_error=true` is the documented escape valve — falls back to text-only composer path (acts as if `enabled=false` for that episode). Default `true` for runner robustness. |
| **Phase 5 28×3 P4 eval exposes new failure modes not seen in canaries.** | Medium | Same gate as 3b: hard rule + acceptance gates. If failure modes are isolated to ≤2 tasks, ship; document residuals. If widespread, halt and analyze. |
| **gpt-5.4-mini `ambiguous_resolutions` accuracy 44.4% may bottleneck P4 disambig.** (Phase 0 finding) | Medium | Still strictly better than the hard-fail baseline (ObjectResolutionError throws on P4 today, so improving ~44% of cases is net-positive). Revisit prompt design only if Phase 5 P4 acceptance gate misses. Three known mitigation paths surfaced in §9 future work. |
| **Context preambles (§4.6) may shift LLM behavior on prior-passing tasks.** Adding role-clarifying text near the top of composer/affordance/avoidance prompts could perturb in-context-example matching even when not intended. | Medium | Phase 1 secondary-canary covers 6 previously-passing tasks for ±1 episode regression detection. **Granular rollback path:** if regressions trace to one specific prompt's preamble (e.g., the avoidance preamble breaks something), revert that preamble alone and ship the rest. If broad regression across all 6 secondary-canary tasks, revert all 3 preambles and re-evaluate the preamble template before re-shipping. The Task-4 composer fixes can ship without the preambles. |
| **`format_scene_state` thresholds (drawer joint > 0.05 etc.) misclassify state on edge cases.** | Low | Phase 2 smoke explicitly checks 3 known scene_obs vectors against expected text output. CALVIN's joint thresholds are well-documented; misclassification would surface as wrong "drawer: open" on a closed-drawer scene, which a 1-ep open_drawer smoke would catch immediately. |
| **Per-task scene image hashes diverge across eval runs (sampled state changes block positions).** | Low | Grounding cache key includes image hash, so re-running with same seeds + same starting conditions = cache hit. Different seeds = different starting state = different image hash = expected cache miss. Cost stays bounded per the audit/Phase 5 estimates. |
| **Phase 2's `_get_static_camera_matrix` returns PyBullet's view matrix not the camera matrix and the projection is wrong.** | Medium | Refactorer writes a 5-line standalone test that projects a known world point (e.g., the red_block centroid from the scene_obs slice) to pixel coords and confirms the projection is within 5px of where the block visibly is in the captured frame. Smoke before integration. |

---

## 8. Out of scope (explicit)

- **Anthropic multimodal.** Task description specifies "OpenAI multimodal." Anthropic image-content shape is documented (see §4.1 placeholder); future-work follow-up.
- **Per-stage frame re-capture.** Per Q3 (locked): episode-start only. Future work if cavity tasks need it.
- **Avoidance prompt extension.** Per §2 row: cavity-aware obstacle masking needs primitive-aware EDT (3b shelved both); v1 leaves avoidance LMP text-only.
- **Gripper camera ingestion.** Per Q4: static-overhead only. Multi-view costs 2× for marginal upside before per-stage re-capture is in place.
- **Resolution sweep.** Per Q4 (locked): 600×600 / FOV=20° matches `multistage.yaml` defaults; sweep only if grounding accuracy is unacceptable.
- **New `parse_query_obj` integration.** Per Q2: stays text-only.
- **VLM scene-image renderer for the live tk window.** Task 5's Renderer Protocol left this hook open (§9). The Phase 2 `voxposer/scene_image.py` produces JPEG bytes; a future `SceneImageRenderer` plugged into `VisualizationManager.register(...)` could surface the annotated frame in the live viewer. Not in v1.
- **Self-correction loop** (composer regenerates with image after value-map failure). Phase 4 measures whether this is needed; if it is, separate plan.
- **`scripts/test_voxposer.py` / `scripts/calibrate_voxposer_objects.py`** — offline tools using `ValueMapVisualizer`. Same boundary as Task 5: zero-touch unless `LLMBackend` signature change forces an adapter.

---

## 9. Future-work hook

The architecture leaves four clean extension points:

1. **VLM renderer for the live tk window.** Task 5's `Renderer` Protocol + `Manager.register(...)` are the documented hooks. A `SceneImageRenderer(Renderer)` implementing `update_state(state)` could read `state["scene_image_jpeg"]` (a new state key emitted by `stage_manager.snapshot()` once Task 7 lands) and display the annotated frame in a side-panel of the tk window. ~50 LoC.
2. **Per-stage scene re-capture.** When the eval budget grows and the cavity-task ceiling is hit, re-capturing at each stage-activation could resolve "scene changed mid-episode" cases. The plumbing point already exists: `stage_manager._activate_stage` is the natural place to invoke a new grounding call. Cache hit rate would drop (image hash changes as gripper moves), but for residual tasks the cost may be worth it.
3. **Anthropic multimodal.** `LLMBackend._call_anthropic` placeholder raises `NotImplementedError` for v1. Adding it is ~15 LoC (Anthropic's `content` shape is similar to OpenAI's). Useful if anthropic-Claude becomes a target for benchmark comparison — and the **specific reason it's worth keeping warm** post-Phase-0: gpt-4o's safety-refusal pattern blocked it from production; Anthropic claude is the natural alternative if gpt-5.4-mini's 44.4% `ambiguous_resolutions` accuracy proves insufficient downstream.
4. **Boosting `ambiguous_resolutions` accuracy** (the Phase 0 residual). Three options ordered by cost:
   - **(a) Iterate grounding prompt.** Few-shot expansion with more P4-flavored examples; chain-of-thought scaffolding in the LMP output (force the model to reason about which canonical noun maps to the ambiguous phrase before emitting the dict). Cheapest path; ~$1 to canary.
   - **(b) Higher image resolution.** 600×600 was Phase-0 default. 1024×1024 doubles input tokens (~3000 per call) and increases label-readability for small objects (LED/switch are ~10px wide at 600×600). Cost ~2-3x. Re-run Phase 0 audit at higher res before committing.
   - **(c) Anthropic claude multimodal path** per (3) above. Different model family; different failure modes. Largest plumbing change (~15 LoC) but a fresh model class to test against.

None of these are blocked by Task 7's v1; all extend cleanly through existing interfaces. Recommend running (a) post-Phase-5 only if the P4 acceptance gate is missed.

**Reusable building blocks from Phase 1+2 (for future LMPs).** The preamble template (§4.6) and the `format_scene_state` schema (§4.5) are designed as reusable contracts:
- **Preamble template:** any new LMP (e.g., a future grasping-decision LMP, a contact-physics LMP, a re-plan LMP) gets the same 5-10 line preamble shape — fill in role + upstream/downstream. Keeps system-context disclosure consistent across the LMP pipeline.
- **`format_scene_state` schema:** new LMPs that need deterministic state can read the same text block from `lmp_interface._scene_state` (no LLM call to re-format the same data). The fixed sections (drawer / slider / lights / gripper / blocks-by-fixture) cover the dominant decisions any CALVIN-side LMP would need.

If/when Task 7 ships and a follow-on task introduces new LMPs, these two artifacts cut their plumbing cost — preamble is ~5 lines, scene_state is already injected. Documented here so the reusability isn't lost when the team moves on.

---

## 10. Approval checklist (team-lead before refactorer kickoff)

- [ ] §0 task-status check confirms #4 + #5 done; #7 has no further blockers.
- [ ] §1 task-description verbatim; user kickoff line cited.
- [ ] §2 surveyed surface matches the survey table sent with the A-E reconciliation.
- [ ] §3 file layout matches the architecture B + Phase 1 preflight design.
- [ ] §4 design spec implements architecture B per locked Q1; output schema matches team-lead's sketch; LLMBackend extension is additive.
- [ ] §5 iteration phases (0-5) match team-lead's locked sequence; Phase 1 ships preflight composer fixes independently.
- [ ] §6 validation includes hard-rule regression check + negative leakage grep + cost accounting.
- [ ] §7 risks include the Phase 2 camera-matrix smoke test + Phase 0 fail-back path.
- [ ] §8 out-of-scope items match the locked answers to Q1-Q9 + team-lead's "annotated frames OK" call.
- [ ] §9 future-work hook documents 3 clean extension points without designing them in.
- [ ] Cost budget ≤ $15 ceiling; nominal path is ~$10 on gpt-4o grounding or ~$1 on gpt-5.4-mini grounding.

On approval → refactorer kickoff message includes pointer to this plan + the iter-0 starting condition (`docs/refactor/task7_phase0_audit.json` blocking deliverable). Same per-iter commit cadence as Task 5.
