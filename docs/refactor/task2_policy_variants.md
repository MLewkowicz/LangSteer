# Task 2 — Policy Variant Split

**Status:** ✅ done  
**Plan:** [task2_plan.md](task2_plan.md)  
**Branch:** `refactoring`

---

## Goal

`policies/diffuser_actor.py` was a 520-line monolith housing four conditioning variants (CLIP language, no-language, primitive-id, primitive+object-id) inside a single class guarded by `if use_primitive_id / use_object_id / use_instruction` branches. Any reader wanting to understand one variant had to parse all of them. Task 2 splits the file into a thin shared base plus one file per variant, leaving the trained `nn.Module` graph (`policies/diffuser_actor_components/`) byte-for-byte untouched.

---

## Plan summary

Full plan: [task2_plan.md](task2_plan.md)

- **Hard constraint:** `policies/diffuser_actor_components/` is frozen. Checkpoint state-dict keys (`encoder.instruction_encoder.weight`, `encoder.primitive_embedding.weight`, etc.) must not change. The encoder already dispatches polymorphically on input dtype/shape.
- **Target layout:** `diffuser_actor_base.py` + four per-variant files + slimmed `diffuser_actor.py` façade.
- **Dispatch:** `build_diffuser_actor_policy(cfg)` factory reads the three conditioning flags (`use_instruction`, `use_primitive_id`, `use_object_id`), cross-validates them, and returns the right instance.
- **Back-compat:** `DiffuserActorPolicy = build_diffuser_actor_policy` alias keeps any caller using the historical constructor syntax working unchanged.
- **Yamls unchanged:** All six `conf/policy/*.yaml` files stay byte-identical; `name: diffuser_actor` continues to be the lookup key.
- **Runner change:** `scripts/run_experiment.py` line 41–42 switches from `DiffuserActorPolicy(cfg_dict)` to `build_diffuser_actor_policy(cfg_dict)`.
- **`PrimitiveObject` inherits `Primitive`:** Object conditioning is strictly an extension of primitive conditioning, so `PrimitiveObjectDiffuserActorPolicy` subclasses `PrimitiveDiffuserActorPolicy`, reusing `set_primitive`, `_current_primitive_id`, and id validation.
- **Two variant hooks:** Base defines `_build_instruction(obs)` and `_log_conditioning_diag(instr_emb, obs)` as the only points subclasses override.
- **Class-level flags:** `_use_instruction`, `_use_primitive_id`, `_use_object_id` constants on each class for runner introspection (e.g. `wire_steering`).

---

## Implementation

All changes are uncommitted on branch `refactoring` (five new files untracked, two modified).

### Files added

| File | Lines | Role |
|------|-------|------|
| `policies/diffuser_actor_base.py` | 458 | `DiffuserActorBasePolicy` — model build + checkpoint load, gripper history buffer, RGB/PCD/gripper tensor prep, `forward()` template with `_build_instruction` / `_log_conditioning_diag` hooks, `_build_guidance_fns`, action conversion (quaternion → Euler, openness → binary). `set_primitive` / `set_object` raise `RuntimeError` by default; subclasses override. |
| `policies/diffuser_actor_language.py` | 83 | `LanguageDiffuserActorPolicy` — CLIP text embedding with per-instruction LRU cache, lazy HuggingFace model load. Class flags: `_use_instruction=True`, `_use_primitive_id=False`, `_use_object_id=False`. |
| `policies/diffuser_actor_nolang.py` | 36 | `NolangDiffuserActorPolicy` — `_build_instruction` returns `None`; model's language cross-attention pathway is never entered. Class flags all False. |
| `policies/diffuser_actor_primitive.py` | 63 | `PrimitiveDiffuserActorPolicy` — `set_primitive(id)` with bounds check; `_build_instruction` emits `(1, 1)` long tensor. Class flags: `_use_primitive_id=True`. |
| `policies/diffuser_actor_primitive_object.py` | 72 | `PrimitiveObjectDiffuserActorPolicy(PrimitiveDiffuserActorPolicy)` — adds `set_object(id)` with bounds check; `_build_instruction` emits `(1, 2)` long tensor `[[prim_id, obj_id]]`. `_use_object_id=True`. |

### Files modified

| File | Before → After | Note |
|------|----------------|------|
| `policies/diffuser_actor.py` | 520 → 66 lines | Rewritten as factory façade. `build_diffuser_actor_policy(cfg)` reads flags, cross-validates (`use_object` requires `use_primitive`; `use_primitive` requires `use_instruction`), and returns the right instance via lazy per-variant imports. `DiffuserActorPolicy = build_diffuser_actor_policy` back-compat alias. |
| `policies/__init__.py` | minimal → 23 lines | Re-exports all five public names: `build_diffuser_actor_policy`, `DiffuserActorPolicy`, `DiffuserActorBasePolicy`, `LanguageDiffuserActorPolicy`, `NolangDiffuserActorPolicy`, `PrimitiveDiffuserActorPolicy`, `PrimitiveObjectDiffuserActorPolicy`. |
| `scripts/run_experiment.py` | line 41–42 | `DiffuserActorPolicy(cfg_dict)` → `build_diffuser_actor_policy(cfg_dict)`. |

### Files unchanged

| File | Note |
|------|------|
| `policies/diffuser_actor_components/` | Byte-for-byte frozen. State-dict keys, encoder dispatch logic untouched. |
| `conf/policy/*.yaml` (6 files) | No change. `name: diffuser_actor`, flag fields, checkpoint paths unchanged. |
| `scripts/run_evaluation.py` | No change; inherits factory via `from scripts.run_experiment import instantiate_policy`. |
| `core/policy.py` | `BasePolicy` interface unchanged. |

### Inheritance tree

```
BasePolicy (core/policy.py)
└── DiffuserActorBasePolicy  (diffuser_actor_base.py)
    ├── LanguageDiffuserActorPolicy      (diffuser_actor_language.py)
    ├── NolangDiffuserActorPolicy        (diffuser_actor_nolang.py)
    └── PrimitiveDiffuserActorPolicy     (diffuser_actor_primitive.py)
        └── PrimitiveObjectDiffuserActorPolicy  (diffuser_actor_primitive_object.py)
```

### Key design choices

1. **Wrapper-only split.** `diffuser_actor_components/` (the trained `nn.Module`) is NOT split. The encoder already dispatches polymorphically on input dtype/shape — `(B,2) long → primitive+object`, `(B,1) long → primitive`, `(B,seq,512) float → CLIP`. Splitting it would rename state-dict keys and break all six trained checkpoints. The flag-tangle lives entirely in the wrapper; the wrapper is what was fixed.
2. **Factory dispatch, zero yaml changes.** `build_diffuser_actor_policy(cfg)` reads three flags and instantiates the right subclass. Hydra `_target_:` dispatch was considered but rejected as more invasive for no functional gain — all six yamls would have needed a `_target_:` key added.
3. **`PrimitiveObject` inherits from `Primitive`**, not from base directly. Object conditioning is a strict extension of primitive conditioning (`use_object_id` requires `use_primitive_id`). Inheriting reuses `set_primitive`, `_current_primitive_id`, and bounds validation; `PrimitiveObjectDiffuserActorPolicy` only adds `set_object` and overrides `_build_instruction` to emit `(1,2)` instead of `(1,1)`.
4. **`forward()` template + two hooks.** Base owns the shared flow (rgb prep → pcd prep → gripper prep → diag → model call → action convert → relative-to-abs). Variants override `_build_instruction` and `_log_conditioning_diag` only. Steering hookup extracted to `_build_guidance_fns(steering, obs)`.
5. **Public API frozen.** `set_primitive`/`set_object` exist on every variant — base raises `RuntimeError` with the helpful steering-hint text preserved verbatim from the original. `_use_*` class-level constants let `wire_steering`'s `getattr(policy, "_use_primitive_id", False)` introspection work unchanged.

### Factory dispatch (key excerpt)

```python
# policies/diffuser_actor.py
def build_diffuser_actor_policy(cfg: Any) -> DiffuserActorBasePolicy:
    use_primitive = cfg.get("use_primitive_id", False)
    use_object    = cfg.get("use_object_id", False)
    use_instruction = cfg.get("use_instruction", True)

    if use_primitive and not use_instruction:
        raise ValueError("use_primitive_id=True requires use_instruction=True")
    if use_object and not use_primitive:
        raise ValueError("use_object_id=True requires use_primitive_id=True")

    if use_object:   return PrimitiveObjectDiffuserActorPolicy(cfg)
    if use_primitive: return PrimitiveDiffuserActorPolicy(cfg)
    if use_instruction: return LanguageDiffuserActorPolicy(cfg)
    return NolangDiffuserActorPolicy(cfg)

DiffuserActorPolicy = build_diffuser_actor_policy  # back-compat
```

---

## Behavior preserved / removed / relocated

| Category | Item | Before → After |
|----------|------|----------------|
| **Preserved** | All four conditioning variants | Monolith `if` branches → separate subclass files |
| **Preserved** | CLIP lazy-load + per-instruction embedding cache | `LanguageDiffuserActorPolicy._get_instruction_embedding` |
| **Preserved** | `set_primitive(id)` bounds check + RuntimeError if unset | `PrimitiveDiffuserActorPolicy` + base default raiser |
| **Preserved** | `set_object(id)` bounds check + RuntimeError if unset | `PrimitiveObjectDiffuserActorPolicy` + base default raiser |
| **Preserved** | `(B,1)` / `(B,2)` long-tensor instruction encoding | Per-variant `_build_instruction` |
| **Preserved** | `DiffuserActorPolicy(cfg)` call syntax | `DiffuserActorPolicy = build_diffuser_actor_policy` alias |
| **Preserved** | `_use_instruction` / `_use_primitive_id` / `_use_object_id` flags | Class-level constants on each variant (runner introspection) |
| **Preserved** | Checkpoint compatibility — state-dict keys unchanged | `diffuser_actor_components/` untouched |
| **Preserved** | Cfg yaml schema unchanged | `conf/policy/*.yaml` byte-identical |
| **Removed** | Variant-dispatch `if` blocks inside single class | Replaced by factory + subclasses |
| **Relocated** | Shared plumbing (model build, checkpoint load, action conversion, forward template) | `diffuser_actor.py` (520L) → `diffuser_actor_base.py` (458L) |
| **Relocated** | Per-variant conditioning logic | `diffuser_actor.py` → four per-variant files |

---

## Smoke tests / validation

*Static checks run by refactorer:*

| Check | Result |
|-------|--------|
| `uv run ruff check policies/` | ✅ clean |
| `uv run ruff format --check policies/` | ✅ clean |
| `uv run mypy policies/` | ✅ 2 pre-existing patterns flagged (pybullet import + CLIP `.to().eval()` chain); no new errors introduced |
| Import + factory smoke test | ✅ `build_diffuser_actor_policy` constructs all four variants |
| **CALVIN end-to-end** `--evaluation langsteer_primitive_object --num-episodes 1 open_drawer` | ✅ **1/1 SUCCEEDED in 3 steps, 100%** |

**End-to-end CALVIN smoke test** (post-refactor, `open_drawer`, seed=42)
- Command: `uv run python scripts/run_evaluation.py --evaluation langsteer_primitive_object --num-episodes 1 --tasks /tmp/smoke_task_order.json`
- Log excerpt (confirms full wiring: steering callbacks → variant `set_primitive`/`set_object` → `_build_instruction` → diag):
  ```
  [steering.stage_manager] Primitive-id set: stage 0 -> 'grasp' (id=0)
  [steering.stage_manager] Object-id set: stage 0 -> 'drawer_handle' (id=2)
  [policies.diffuser_actor_primitive_object] [Diag] primitive_id=0 object_id=2
  [steering.stage_manager] Stage transition: 0 → 1
    ✓ Episode 1/1 SUCCEEDED (steps=3, reward=0.00)
  ```
- Confirms: `set_primitive`/`set_object` callbacks wired correctly from `StageManager` → variant subclass; `[Diag]` lines appear from the correct per-variant module (not the old monolith); grasp → pull stage transition visible.

**Soft-target deviations:**
- `diffuser_actor_base.py` is 458 lines vs. plan target ~290–330. Same situation as `stage_manager.py` in Task 1 — single responsibility is preserved, the base genuinely owns all shared plumbing.
- Total ~778 lines across 6 wrapper files vs. 520 in one. Growth is class boilerplate + docstrings; no variant logic is entangled.

---

## Open items

- **Model-side flag tangle** — `DiffuserActor` / `Encoder` / `DiffusionHead` in `diffuser_actor_components/` still carry the flag-dispatch logic internally (e.g. `if self.use_primitive_id:` in the encoder). Cleaning this up would require a state-dict migration to preserve checkpoint loadability — separate future scope.
- **`training/policies/diffuser_actor/trainer.py`** — analogous flag tangle on the training side is out of scope for Task 2. Continues to work via the existing `DiffuserActor` nn.Module directly.
- **`scripts/run_evaluation.py` call site** — still uses `DiffuserActorPolicy` via back-compat alias (only `run_experiment.py` was updated). Could be updated to `build_diffuser_actor_policy` for consistency; low priority.
- **`_use_*` flag convention** — `wire_steering` and runners rely on class-level `_use_*` constants. No ABC enforcement; worth formalising as a documented protocol if variant count grows.
