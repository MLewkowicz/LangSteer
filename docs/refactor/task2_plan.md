# Task 2 — Split `policies/diffuser_actor.py` into per-variant files

**Owner:** planner → refactorer (after approval)
**Branch:** `refactoring`
**Goal:** Replace the 520-line `DiffuserActorPolicy` (one class, three conditioning flags, four code paths) with four focused per-variant policy classes that share a thin base. Public API stays identical so `uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer` still works.

---

## 1. Scope — what changes, what doesn't

### Hard rule (locked by team-lead)
**`policies/diffuser_actor_components/` is byte-for-byte unchanged.** That directory is the trained `nn.Module` graph (`DiffuserActor`, `Encoder`, `DiffusionHead`, attention layers). All six trained checkpoints save state-dicts under attribute paths like `encoder.instruction_encoder.weight`, `encoder.primitive_embedding.weight`, `prediction_head.traj_lang_attention.0.*`. Any restructuring there renames keys and breaks every checkpoint. The encoder's `encode_instruction` already dispatches polymorphically on input dtype/shape (`(B,2) long` → primitive+object, `(B,1) long` → primitive, `(B,seq,512) float` → CLIP) — that's a clean contract. The variant bloat the user is complaining about lives entirely in the wrapper. We fix it there.

### What this task DOES touch
- `policies/diffuser_actor.py` (520 lines) — split into per-variant files.
- `conf/policy/*.yaml` (6 files) — **unchanged**. `name: diffuser_actor` stays; the runner dispatches on the existing flag triple.
- `scripts/run_experiment.py` — one call site (lines 41–42) switches from `DiffuserActorPolicy(cfg_dict)` to `build_diffuser_actor_policy(cfg_dict)`.
- `scripts/run_evaluation.py` — no change; inherits via `from scripts.run_experiment import instantiate_policy`.

### What this task does NOT touch
- `policies/diffuser_actor_components/*.py` (eight files). Byte-for-byte frozen.
- `training/policies/diffuser_actor/*` (trainer + dataset). Training-side flag-plumbing cleanup is its own task if the user wants it later.
- `core/policy.py` (BasePolicy interface). The four variants all subclass `BasePolicy` through the new base.

---

## 2. Target file layout

```
policies/
  diffuser_actor_base.py            # DiffuserActorBasePolicy — shared plumbing
  diffuser_actor_language.py        # LanguageDiffuserActorPolicy — CLIP text + mask_language
  diffuser_actor_nolang.py          # NolangDiffuserActorPolicy — no instruction
  diffuser_actor_primitive.py       # PrimitiveDiffuserActorPolicy — (B,1) primitive-id + set_primitive
  diffuser_actor_primitive_object.py  # PrimitiveObjectDiffuserActorPolicy — (B,2) prim+obj + set_object (subclass of Primitive)
  diffuser_actor.py                 # DEPRECATED back-compat shim — re-exports + flag-sniffing factory for out-of-tree callers
  diffuser_actor_components/        # UNCHANGED
```

Rationale per team-lead's Q2 answer (per-variant file + thin base):
- **One file per variant.** Addresses the user's "not clogging one file" complaint at the file level. A reader opens `diffuser_actor_primitive_object.py` and sees exactly the code that variant runs — no flag dispatch, no other variant's CLIP machinery, no unreachable branches.
- **Thin base, not kitchen-sink.** Base earns its keep by removing real duplication only: RGB/PCD/gripper prep, action conversion, model build, checkpoint loading, the `forward()` template, the steering-hookup helper. ~280–320 lines. Variant-specific behavior lives in subclass overrides of two hooks: `_build_instruction(obs) -> Tensor | None` and `_log_conditioning_diag(instr_emb, obs)`.
- **Primitive+Object inherits from Primitive.** Object conditioning is strictly an extension of primitive conditioning (cfg gate: `use_object_id` requires `use_primitive_id`). Inheriting reuses `set_primitive`, primitive-id validation, and the `_current_primitive_id` field for free; the subclass only adds `set_object` and overrides `_build_instruction` to emit `(B,2)` instead of `(B,1)`.

File-size targets:

| File | Target | Notes |
|---|---:|---|
| `diffuser_actor_base.py` | ~280–320 | RGB/PCD/gripper prep, action conversion, model build, checkpoint loading, `forward()` template, `_build_guidance_fns` |
| `diffuser_actor_language.py` | ~80–110 | CLIP cache + tokenizer + `_get_instruction_embedding` + diag hook |
| `diffuser_actor_nolang.py` | ~30–50 | `_build_instruction` returns None + diag hook |
| `diffuser_actor_primitive.py` | ~80–110 | `set_primitive`, validation, `(B,1)` builder, diag hook |
| `diffuser_actor_primitive_object.py` | ~50–80 | subclass; `set_object`, `(B,2)` builder, diag hook |
| `diffuser_actor.py` | ~40–60 | factory + re-exports + back-compat alias |

Total ~560–730 lines across six files vs. today's 520 in one. The growth is class boilerplate + docstrings; variant logic is no longer entangled.

---

## 3. Dispatch — flag-sniffing factory function

Per team-lead's final answer (factory is the minimum-viable shape; yamls stay free of `_target_:`). The new `policies/diffuser_actor.py` becomes a small façade that exposes a `build_diffuser_actor_policy(cfg)` factory, re-exports the four variant classes, and keeps a `DiffuserActorPolicy = build_diffuser_actor_policy` alias for back-compat with any caller using the historical constructor syntax.

### 3.1 Yamls — unchanged

All six `conf/policy/*.yaml` files stay byte-identical. `name: diffuser_actor` continues to be the lookup key in `instantiate_policy`, and the existing flag fields (`use_instruction`, `use_primitive_id`, `use_object_id`, `num_primitives`, `num_objects`) drive variant selection.

### 3.2 Constructor signature — unchanged from today

Each variant's `__init__` keeps the same dict-arg shape as today's monolith:

```python
def __init__(self, cfg: Any) -> None:
    super().__init__(cfg)
    # cfg.get("backbone", "clip") etc. — same body as today
```

No `**cfg` rewrite. The factory passes the cfg dict positionally; subclasses delegate to `super().__init__(cfg)` so `BasePolicy.__init__` stays happy.

### 3.3 Factory file `policies/diffuser_actor.py`

```python
"""Variant dispatch for DiffuserActor policies.

Replaces the original 520-line monolith. Three things live here:
  1. The flag-sniffing factory `build_diffuser_actor_policy(cfg)`.
  2. Re-exports of the four concrete variant classes (so callers can
     `from policies.diffuser_actor import LanguageDiffuserActorPolicy` etc.).
  3. A `DiffuserActorPolicy = build_diffuser_actor_policy` alias — keeps the
     historical `DiffuserActorPolicy(cfg_dict)` call site working without
     modification. Greps for in-repo callers found only run_experiment.py
     (updated by Step 7) and the re-export in policies/__init__.py.
"""

from typing import Any

from policies.diffuser_actor_base import DiffuserActorBasePolicy
from policies.diffuser_actor_language import LanguageDiffuserActorPolicy
from policies.diffuser_actor_nolang import NolangDiffuserActorPolicy
from policies.diffuser_actor_primitive import PrimitiveDiffuserActorPolicy
from policies.diffuser_actor_primitive_object import PrimitiveObjectDiffuserActorPolicy


def build_diffuser_actor_policy(cfg: Any) -> DiffuserActorBasePolicy:
    """Instantiate the right DiffuserActor variant based on cfg flags.

    Cross-validation gates (use_primitive_id requires use_instruction;
    use_object_id requires use_primitive_id) raise ValueError before any
    model is built, so a malformed yaml fails fast with a clear message.
    """
    use_primitive = cfg.get("use_primitive_id", False)
    use_object = cfg.get("use_object_id", False)
    use_instruction = cfg.get("use_instruction", True)

    if use_primitive and not use_instruction:
        raise ValueError(
            "use_primitive_id=True requires use_instruction=True "
            "(primitive mode reuses the instruction cross-attention pipeline)."
        )
    if use_object and not use_primitive:
        raise ValueError("use_object_id=True requires use_primitive_id=True.")

    if use_object:
        return PrimitiveObjectDiffuserActorPolicy(cfg)
    if use_primitive:
        return PrimitiveDiffuserActorPolicy(cfg)
    if use_instruction:
        return LanguageDiffuserActorPolicy(cfg)
    return NolangDiffuserActorPolicy(cfg)


# Back-compat alias — keeps `DiffuserActorPolicy(cfg)` call syntax working.
# It's intentionally a function (not a class), so callers can keep the
# constructor-style invocation. `isinstance(x, DiffuserActorPolicy)` checks
# fail with this shim, but a repo-wide grep confirmed no such checks exist.
DiffuserActorPolicy = build_diffuser_actor_policy
```

### 3.4 `scripts/run_experiment.py` diff (one call site)

Current (lines 37–50):

```python
def instantiate_policy(cfg: DictConfig) -> BasePolicy:
    """Factory function to instantiate policy based on config."""
    policy_name = cfg.policy.name
    if policy_name == "diffuser_actor":
        from policies.diffuser_actor import DiffuserActorPolicy
        policy = DiffuserActorPolicy(OmegaConf.to_container(cfg.policy, resolve=True))
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    if hasattr(cfg.policy, "ckpt_path") and cfg.policy.ckpt_path:
        policy.load_checkpoint(cfg.policy.ckpt_path)

    return policy
```

New (preferred form — explicit factory call):

```python
def instantiate_policy(cfg: DictConfig) -> BasePolicy:
    """Factory function to instantiate policy based on config."""
    policy_name = cfg.policy.name
    if policy_name == "diffuser_actor":
        from policies.diffuser_actor import build_diffuser_actor_policy
        policy = build_diffuser_actor_policy(
            OmegaConf.to_container(cfg.policy, resolve=True)
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    if hasattr(cfg.policy, "ckpt_path") and cfg.policy.ckpt_path:
        policy.load_checkpoint(cfg.policy.ckpt_path)

    return policy
```

The behavior is identical with or without renaming `DiffuserActorPolicy` → `build_diffuser_actor_policy` at the call site (the alias keeps the old form working). The rename is a clarity nudge for future readers — variant dispatch is now explicit.

### 3.5 `scripts/run_evaluation.py` — no change

`run_evaluation.py:38` imports `instantiate_policy` from `run_experiment` (`from scripts.run_experiment import instantiate_env, instantiate_policy, instantiate_steering`). Only one call site to update; the eval script inherits via that import. Team-lead's correction noted (prior plan version overcounted as "two callers").

### 3.6 `policies/__init__.py` re-exports

Per team-lead's Refinement 2: expand the `__init__.py` re-export so the concrete variant classes are reachable from `from policies import ...`:

```python
# policies/__init__.py
from policies.diffuser_actor import (
    DiffuserActorPolicy,            # alias for build_diffuser_actor_policy — back-compat
    build_diffuser_actor_policy,    # preferred entry point
    LanguageDiffuserActorPolicy,
    NolangDiffuserActorPolicy,
    PrimitiveDiffuserActorPolicy,
    PrimitiveObjectDiffuserActorPolicy,
)

__all__ = [
    "DiffuserActorPolicy",
    "build_diffuser_actor_policy",
    "LanguageDiffuserActorPolicy",
    "NolangDiffuserActorPolicy",
    "PrimitiveDiffuserActorPolicy",
    "PrimitiveObjectDiffuserActorPolicy",
]
```

The variant classes flow through `policies.diffuser_actor`'s re-exports (§3.3), so `policies/__init__.py` imports them from there. Tests and notebooks can do `from policies import PrimitiveObjectDiffuserActorPolicy` directly.

---

## 4. Public-API contract (frozen)

What external code touches today and what each variant must expose post-refactor:

```python
# Used by every caller
policy.reset()
policy.load_checkpoint(path)
policy.forward(obs, steering)

# Used by steering wiring (run_evaluation.py:191-202, run_experiment.py:143-153)
hasattr(policy, "set_primitive")
hasattr(policy, "set_object")
policy._use_primitive_id   # bool attribute read for guard
policy._use_object_id      # bool attribute read for guard
policy._model.position_noise_scheduler   # read directly by wire_steering (run_evaluation.py:176)
policy._model.rotation_noise_scheduler   # read directly by wire_steering (run_evaluation.py:179)

# Used by training entry point (scripts/train_diffuser_actor.py)
# → imports `DiffuserActor` (the nn.Module) from policies.diffuser_actor_components, NOT the wrapper.
# Untouched.
```

Per-variant matrix:

| Variant | `set_primitive` | `set_object` | `_use_primitive_id` | `_use_object_id` | `_use_instruction` | model's `instruction` arg |
|---|---|---|---|---|---|---|
| Language | RuntimeError | RuntimeError | False | False | True | `(1, seq_len, 512)` float |
| Nolang | RuntimeError | RuntimeError | False | False | False | None |
| Primitive | OK | RuntimeError | True | False | True | `(1, 1)` long |
| Primitive+Object | OK | OK | True | True | True | `(1, 2)` long |

The three private flag attributes (`_use_primitive_id`, `_use_object_id`, `_use_instruction`) are set as class-level constants on each subclass so `wire_steering`'s `getattr(policy, "_use_primitive_id", False)` introspection keeps working without changes.

`policy._model` (the underlying `DiffuserActor` nn.Module) lives on the base class, built once in `DiffuserActorBasePolicy.__init__`. `wire_steering` reads `policy._model.position_noise_scheduler` / `policy._model.rotation_noise_scheduler` — same path as today.

---

## 5. New module signatures

### `diffuser_actor_base.py`

```python
class DiffuserActorBasePolicy(BasePolicy):
    """Shared plumbing for the four DiffuserActor variants.

    Owns: model build, gripper history buffer, RGB/PCD/gripper prep, action
    conversion, checkpoint loading, the top-level forward() shape, the
    diagnostic-logging frame, and the steering hookup helper. Variant-specific
    behavior is delegated to two abstract hooks:
        _build_instruction(obs)              -> Tensor | None
        _log_conditioning_diag(instr_emb, obs)
    """

    # Subclasses override these as class-level constants:
    _use_instruction: bool = True
    _use_primitive_id: bool = False
    _use_object_id: bool = False

    def __init__(self, cfg: Any) -> None: ...
    def load_checkpoint(self, path: str) -> None: ...
    def reset(self) -> None: ...
    def forward(self, obs: Observation, steering: Optional[BaseSteering] = None) -> Action: ...

    # Hooks — must be overridden by variants
    def _build_instruction(self, obs: Observation) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def _log_conditioning_diag(
        self, instr_emb: Optional[torch.Tensor], obs: Observation,
    ) -> None:
        """Append variant-specific lines inside the first-2-forwards diagnostic block."""
        raise NotImplementedError

    # Shared helpers (verbatim from today's wrapper)
    def _prepare_rgb(self, obs): ...
    def _prepare_pcd(self, obs): ...
    def _prepare_gripper(self, obs): ...
    @staticmethod
    def _convert_quat_to_euler(quat): ...
    def _convert_action(self, trajectory): ...

    # New helper extracted from today's forward() L449-468
    def _build_guidance_fns(
        self, steering: Optional[BaseSteering], obs: Observation,
    ) -> tuple[Optional[Callable], Optional[Callable]]:
        """Wire steering's gripper-pos/rotation setters, then return
        (guidance_fn, dps_guidance_fn) for the model call. One is None."""
        ...

    # Default raisers for set_primitive / set_object
    def set_primitive(self, primitive_id: int) -> None:
        raise RuntimeError(
            "set_primitive() called but policy is not primitive-conditioned. "
            "Use policy=diffuser_actor_primitive or policy=diffuser_actor_primitive_object."
        )

    def set_object(self, object_id: int) -> None:
        raise RuntimeError(
            "set_object() called but policy is not object-conditioned. "
            "Use policy=diffuser_actor_primitive_object."
        )
```

### `diffuser_actor_language.py`

```python
class LanguageDiffuserActorPolicy(DiffuserActorBasePolicy):
    """Original 3D Diffuser Actor — CLIP text embeddings.

    Owns: HuggingFace CLIPTextModel lazy load, instruction cache, mask_language
    handling. Used by configs: diffuser_actor.yaml, diffuser_actor_maskedlang.yaml.
    """
    _use_instruction = True
    _use_primitive_id = False
    _use_object_id = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.text_max_length: int = cfg.get("text_max_length", 16)
        self._instruction_cache: Dict[str, torch.Tensor] = {}
        self._clip_text_model = None
        self._clip_tokenizer = None

    def _build_instruction(self, obs):
        emb = self._get_instruction_embedding(obs.instruction)
        return emb.unsqueeze(0)  # (1, seq_len, 512)

    def _get_instruction_embedding(self, instruction_text: str) -> torch.Tensor: ...
    def _log_conditioning_diag(self, instr_emb, obs): ...
```

### `diffuser_actor_nolang.py`

```python
class NolangDiffuserActorPolicy(DiffuserActorBasePolicy):
    """No instruction conditioning. Configs: diffuser_actor_nolang.yaml,
    diffuser_actor_nolang_abcd.yaml.
    """
    _use_instruction = False

    def _build_instruction(self, obs):
        return None

    def _log_conditioning_diag(self, instr_emb, obs):
        logger.info("[Diag] no language conditioning (use_instruction=False)")
```

### `diffuser_actor_primitive.py`

```python
class PrimitiveDiffuserActorPolicy(DiffuserActorBasePolicy):
    """Primitive-id conditioning. Replaces CLIP with nn.Embedding(num_primitives, D).

    Config: diffuser_actor_primitive.yaml.
    """
    _use_instruction = True
    _use_primitive_id = True
    _use_object_id = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._num_primitives: int = cfg.get("num_primitives", 4)
        self._current_primitive_id: Optional[int] = None

    def set_primitive(self, primitive_id: int) -> None:
        if not (0 <= primitive_id < self._num_primitives):
            raise ValueError(
                f"primitive_id={primitive_id} out of range [0, {self._num_primitives})"
            )
        self._current_primitive_id = int(primitive_id)

    def _build_instruction(self, obs):
        if self._current_primitive_id is None:
            raise RuntimeError(
                "Primitive-id mode active but no primitive set. The policy "
                "expects a steering module (e.g. steering=voxposer) to drive "
                "set_primitive(idx) at every stage transition. Either add "
                "steering=voxposer to your run, or call policy.set_primitive(idx) "
                "manually before each forward()."
            )
        return torch.tensor(
            [[self._current_primitive_id]],
            dtype=torch.long, device=self._device,
        )  # (1, 1)

    def _log_conditioning_diag(self, instr_emb, obs):
        logger.info(f"[Diag] primitive_id={self._current_primitive_id}")
```

### `diffuser_actor_primitive_object.py`

```python
class PrimitiveObjectDiffuserActorPolicy(PrimitiveDiffuserActorPolicy):
    """Primitive + object conditioning — production variant.

    Adds parallel nn.Embedding(num_objects, D) on top of the primitive
    embedding. Config: diffuser_actor_primitive_object.yaml.
    """
    _use_object_id = True

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._num_objects: int = cfg.get("num_objects", 8)
        self._current_object_id: Optional[int] = None

    def set_object(self, object_id: int) -> None:
        if not (0 <= object_id < self._num_objects):
            raise ValueError(
                f"object_id={object_id} out of range [0, {self._num_objects})"
            )
        self._current_object_id = int(object_id)

    def _build_instruction(self, obs):
        if self._current_primitive_id is None:
            raise RuntimeError(
                "Primitive-id mode active but no primitive set. "
                "Add steering=voxposer or call policy.set_primitive(idx) manually."
            )
        if self._current_object_id is None:
            raise RuntimeError(
                "Object-id mode active but no object set. "
                "Add steering=voxposer (whose composer emits object slots in every "
                "stage tuple) or call policy.set_object(idx) manually before forward()."
            )
        return torch.tensor(
            [[self._current_primitive_id, self._current_object_id]],
            dtype=torch.long, device=self._device,
        )  # (1, 2)

    def _log_conditioning_diag(self, instr_emb, obs):
        logger.info(
            f"[Diag] primitive_id={self._current_primitive_id} "
            f"object_id={self._current_object_id}"
        )
```

---

## 6. Top-level `forward()` shape (in base class)

The whole flow factors as a single method, with two variant hooks and one helper:

```python
def forward(self, obs, steering=None):
    with torch.no_grad():
        rgb_obs = self._prepare_rgb(obs)
        pcd_obs = self._prepare_pcd(obs)
        curr_gripper = self._prepare_gripper(obs)

        instr_emb = self._build_instruction(obs)   # ← variant hook

        # Diagnostic logging frame — shared lines, then variant hook.
        _diag = self._log_count < 2
        if _diag:
            logger.info(f"[Diag] rgb: shape={rgb_obs.shape}, range=[{rgb_obs.min():.3f}, {rgb_obs.max():.3f}]")
            logger.info(f"[Diag] pcd: shape={pcd_obs.shape}, range=[{pcd_obs.min():.3f}, {pcd_obs.max():.3f}]")
            g = curr_gripper[0, -1, :]
            logger.info(f"[Diag] gripper[-1]: pos={g[:3].cpu().numpy()}, quat={g[3:7].cpu().numpy()}, "
                        f"quat_norm={g[3:7].norm().item():.4f}")
            self._log_conditioning_diag(instr_emb, obs)   # ← variant hook

        trajectory_mask = torch.ones(1, self.pred_horizon, device=self._device)
        guidance_fn, dps_guidance_fn = self._build_guidance_fns(steering, obs)

        trajectory = self._model(
            gt_trajectory=None,
            trajectory_mask=trajectory_mask,
            rgb_obs=rgb_obs,
            pcd_obs=pcd_obs,
            curr_gripper=curr_gripper,
            instruction=instr_emb,
            run_inference=True,
            mask_language=self._mask_language,    # harmless when use_instruction=False
            guidance_fn=guidance_fn,
            dps_guidance_fn=dps_guidance_fn,
            corrector_steps=self._corrector_steps,
            corrector_step_size=self._corrector_step_size,
        )

        if _diag:
            t = trajectory[0, 0]
            logger.info(f"[Diag] raw_traj[0]: pos={t[:3].cpu().numpy()}, "
                        f"quat={t[3:7].cpu().numpy()}, openness={t[7].item():.3f}")

        action_np = self._convert_action(trajectory)

        if _diag:
            logger.info(f"[Diag] after_convert[0]: {action_np[0, 0]}")

        if self._relative:
            gripper_last = curr_gripper[:, [-1], :]
            gripper_padded = torch.cat([gripper_last, torch.zeros_like(gripper_last[..., :1])], dim=-1)
            gripper_euler = self._convert_action(gripper_padded)
            if _diag:
                logger.info(f"[Diag] gripper_euler (base pose): {gripper_euler[0, 0]}")
            action_np[..., :3] += gripper_euler[..., :3]
            action_np[..., 3:6] += gripper_euler[..., 3:6]

        if _diag:
            logger.info(f"[Diag] after_rel2abs[0]: {action_np[0, 0]}")
            self._log_count += 1

        action_np = action_np.squeeze(0)

    gripper = float(action_np[0, 6])
    return Action(trajectory=action_np, gripper=gripper)
```

`_build_guidance_fns` is a small helper holding today's L449–468 block: gripper-pos/rotation setter dispatch on the steering object, the `_steering_fn` closure, and the epsilon-vs-dps mode selection via `steering.guidance_mode`.

---

## 7. Ordered refactor steps (for `refactorer`)

Each step leaves the codebase importable. Run after each step:

```bash
uv run python -c "from policies.diffuser_actor_base import DiffuserActorBasePolicy; print('ok')"
```

After Step 7:

```bash
uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer
```

### Step 0 — Scaffold (no behavior change)
Create five new empty files with module docstrings only:
- `policies/diffuser_actor_base.py`
- `policies/diffuser_actor_language.py`
- `policies/diffuser_actor_nolang.py`
- `policies/diffuser_actor_primitive.py`
- `policies/diffuser_actor_primitive_object.py`

Leave existing `policies/diffuser_actor.py` and yamls untouched.

### Step 1 — `diffuser_actor_base.py` (extract shared plumbing)
Move from `policies/diffuser_actor.py` verbatim:
- `__init__` setup EXCEPT the four variant flags (`_use_instruction`, `_use_primitive_id`, `_num_primitives`, `_use_object_id`, `_num_objects`), the cross-validation gates, and the variant state fields (`_current_primitive_id`, `_current_object_id`).
- Keep `__init__(self, cfg: Any) -> None:` (same dict-arg shape as today). No signature change; `cfg.get(...)` calls in the body are unchanged. Subclasses call `super().__init__(cfg)`.
- `load_checkpoint`, `reset`.
- `_prepare_rgb`, `_prepare_pcd`, `_prepare_gripper`.
- `_convert_quat_to_euler` (staticmethod), `_convert_action`.
- Top-level `forward()` body restructured per §6, with `_build_instruction` and `_log_conditioning_diag` as abstract hooks.
- Extract today's steering-hookup block (L449–468) into `_build_guidance_fns(steering, obs) -> (guidance_fn, dps_guidance_fn)`.
- Default `set_primitive` / `set_object` that raise RuntimeError with the same shape as today's exceptions.
- Class-level constants `_use_instruction=True`, `_use_primitive_id=False`, `_use_object_id=False` (subclasses override).
- Top-of-file import guard for `DiffuserActor` (today at policies/diffuser_actor.py:12–18) moves here — this is where it's actually constructed.

After this step, `policies/diffuser_actor.py` still exists and still works (the original `DiffuserActorPolicy` class is untouched).

### Step 2 — `diffuser_actor_nolang.py`
Simplest variant. Implements:
- Class-level `_use_instruction = False`
- `_build_instruction(obs)` → `return None`
- `_log_conditioning_diag(...)` → logs `"[Diag] no language conditioning (use_instruction=False)"`
- `__init__` calls `super().__init__(cfg)` only

### Step 3 — `diffuser_actor_language.py`
Move from `policies/diffuser_actor.py`:
- `text_max_length`, `_instruction_cache`, `_clip_text_model`, `_clip_tokenizer` init
- `_get_instruction_embedding(text)` method (verbatim from L200–232)
- `_build_instruction(obs)` returns `_get_instruction_embedding(obs.instruction).unsqueeze(0)`
- `_log_conditioning_diag(...)` → logs CLIP shape/norm + instruction text + "(masked)" suffix when `_mask_language`

### Step 4 — `diffuser_actor_primitive.py`
Move from `policies/diffuser_actor.py`:
- `_num_primitives`, `_current_primitive_id` init
- `set_primitive(primitive_id)` — drop the `if not self._use_primitive_id` guard (class is unconditionally primitive-only); keep the range check (`0 <= primitive_id < self._num_primitives`)
- `_build_instruction(obs)` builds `(1, 1)` long tensor; raises the exact RuntimeError text from today's L390–397 if `_current_primitive_id is None`
- `_log_conditioning_diag(...)` → logs primitive_id

### Step 5 — `diffuser_actor_primitive_object.py`
Subclass of `PrimitiveDiffuserActorPolicy`:
- Class-level `_use_object_id = True`
- `_num_objects`, `_current_object_id` init in `__init__` (after `super().__init__(cfg)`)
- `set_object(object_id)` — verbatim range check; no `_use_object_id` guard
- Override `_build_instruction(obs)` to build `(1, 2)` long tensor; raises exact text from today's L398–409 when either id is None
- Override `_log_conditioning_diag(...)` → logs both ids

### Step 6 — Rewrite `policies/diffuser_actor.py` as the factory + alias façade
Replace the original 520-line file with the contents from §3.3:
- Module docstring describing the three roles (factory, re-exports, alias).
- Re-exports of `DiffuserActorBasePolicy` + the four concrete variant classes.
- `build_diffuser_actor_policy(cfg)` factory with both cross-validation gates.
- `DiffuserActorPolicy = build_diffuser_actor_policy` back-compat alias.

### Step 7 — Update `policies/__init__.py` and `scripts/run_experiment.py`
**`policies/__init__.py`** (per team-lead's Refinement 2): expand the re-export to surface the four concrete variant classes alongside `DiffuserActorPolicy` and `build_diffuser_actor_policy`. See §3.6 for the exact import + `__all__` block.

**`scripts/run_experiment.py`** (single call site — team-lead's correction): update lines 41–42 to use `build_diffuser_actor_policy` explicitly (§3.4). `scripts/run_evaluation.py:38` inherits the change via `from scripts.run_experiment import instantiate_policy`; no direct edit there.

Yamls stay untouched.

### Step 8 — Smoke test (production path)
```bash
uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer
```
Must run `open_drawer` end-to-end (Task 1's smoke baseline). Compare logs:
- `[Diag] primitive_id=2 object_id=2` should appear (verbatim text).
- Stage transitions and grasp-gate logs unchanged.

### Step 9 — Per-variant class-constants spot-check (no instantiation)
Per team-lead's Refinement 3: instantiating each yaml would build the full `DiffuserActor` (CLIP backbone + encoder + diffusion head) six times — slow and GPU-bound. The flag triple lives on each subclass as class-level constants, so we can verify dispatch without constructing instances:

```python
from policies.diffuser_actor import (
    LanguageDiffuserActorPolicy,
    NolangDiffuserActorPolicy,
    PrimitiveDiffuserActorPolicy,
    PrimitiveObjectDiffuserActorPolicy,
)

for cls in (
    LanguageDiffuserActorPolicy,
    NolangDiffuserActorPolicy,
    PrimitiveDiffuserActorPolicy,
    PrimitiveObjectDiffuserActorPolicy,
):
    print(cls.__name__,
          cls._use_instruction, cls._use_primitive_id, cls._use_object_id)
```

Expected output:
| class | inst | prim | obj | yamls that dispatch to it |
|---|---|---|---|---|
| LanguageDiffuserActorPolicy | T | F | F | `diffuser_actor`, `diffuser_actor_maskedlang` |
| NolangDiffuserActorPolicy | F | F | F | `diffuser_actor_nolang`, `diffuser_actor_nolang_abcd` |
| PrimitiveDiffuserActorPolicy | T | T | F | `diffuser_actor_primitive` |
| PrimitiveObjectDiffuserActorPolicy | T | T | T | `diffuser_actor_primitive_object` |

Step 8's full smoke test is the real dispatch verification — this spot-check just catches "class-level flag accidentally set wrong" before launching the heavy run.

### Step 10 — Back-compat alias spot-check
```python
from policies.diffuser_actor import DiffuserActorPolicy, build_diffuser_actor_policy
# Alias is a direct reference, not a deprecation wrapper — works without warning.
assert DiffuserActorPolicy is build_diffuser_actor_policy
# Old `DiffuserActorPolicy(cfg)` call syntax still works.
p = DiffuserActorPolicy({"use_primitive_id": True, "use_object_id": True, "use_instruction": True, ...})
assert type(p).__name__ == "PrimitiveObjectDiffuserActorPolicy"
```

### Step 11 — Pre-PR cleanup
- `uv run ruff format policies/`
- `uv run ruff check policies/`
- `uv run mypy policies/` (note pre-existing errors in `policies/diffuser_actor_components/` are out of scope; only flag new ones introduced by the wrapper split)
- Each new file: top-of-file docstring describing the variant's role.

---

## 8. Behavior preserved / removed / relocated

### Preserved (semantics identical)
- All public methods on the wrapper: `__init__(cfg)`, `load_checkpoint(path)`, `reset()`, `forward(obs, steering)`, `set_primitive(id)`, `set_object(id)` — same signatures, same exceptions (user-action hint preserved verbatim; only the leading class-context phrase may shift).
- All cfg knobs read from yaml (no field rename).
- `name: diffuser_actor` field stays — read by `run_experiment.py:108` (PCD-images guard).
- Internal flag attributes (`_use_primitive_id`, `_use_object_id`, `_use_instruction`) — read by `wire_steering` in `run_evaluation.py:191–202` and `run_experiment.py:143–153`.
- Both cross-validation gates (`use_primitive_id requires use_instruction`, `use_object_id requires use_primitive_id`). Enforced in `build_diffuser_actor_policy(cfg)` at yaml-time and re-enforced as class-level invariants on each subclass via `_use_instruction`/`_use_primitive_id`/`_use_object_id` constants.
- All diagnostic log messages: exact text, exact "first 2 forward passes" cadence (`_log_count`), exact line ordering (shared rgb/pcd/gripper lines → variant-specific lines).
- All steering hookup: `set_current_gripper_pos`/`set_current_gripper_rotation` dispatch, `_steering_fn` closure, epsilon-vs-dps mode selection via `steering.guidance_mode`.
- Relative-to-absolute pose conversion at the end of `forward()`.
- `trajectory_mask = ones(1, pred_horizon)` stub.
- `mask_language`, `corrector_steps`, `corrector_step_size` plumbing into the model call.
- Checkpoint loading semantics: 3DA format detection (`weight` vs. `state_dict`), DDP `module.` prefix stripping, missing/unexpected key logging.
- CLIP lazy load with `text_max_length`.
- Instruction cache by text string.
- `policy._model` attribute and `policy._model.position_noise_scheduler` / `rotation_noise_scheduler` paths (read by `wire_steering`).

### Removed
- **Nothing.** Minimum-viable structural change.

### Relocated (origin → destination)

| Origin (today's `policies/diffuser_actor.py`) | Destination |
|---|---|
| `__init__` lines 35–93 (shared knobs + model build + gripper history + log counter) | `DiffuserActorBasePolicy.__init__` |
| `__init__` lines 60–63 (flag reads), 81–82 (`_current_*` fields), 78–80 (`_num_primitives`/`_num_objects`) | Distributed to subclasses; class-level constants for the booleans |
| `__init__` lines 64–72 (cross-validation gates) | `build_diffuser_actor_policy` factory (yaml-time) + class-level constants on each subclass |
| `load_checkpoint`, `reset`, `_prepare_rgb`, `_prepare_pcd`, `_prepare_gripper`, `_convert_quat_to_euler`, `_convert_action` | `DiffuserActorBasePolicy` |
| `set_primitive` (L170–185), `set_object` (L187–198) | Default RuntimeError raisers on base; real implementations on `Primitive*` subclasses |
| `_get_instruction_embedding` (L200–232), CLIP lazy-load state | `LanguageDiffuserActorPolicy` |
| `forward()` L388–417 (the four-way `instr_emb` builder) | `_build_instruction` hook overridden per subclass |
| `forward()` L427–440 (variant-specific diag) | `_log_conditioning_diag` hook overridden per subclass |
| `forward()` L449–468 (steering hookup) | `_build_guidance_fns(steering, obs)` helper on base |
| `forward()` outer flow (RGB/PCD/gripper prep → model call → action conversion → relative-to-abs) | `DiffuserActorBasePolicy.forward()` template |
| Top-of-file `_IMPORT_ERROR` guard for `DiffuserActor` (L12–18) | `DiffuserActorBasePolicy` (top of `diffuser_actor_base.py`) |
| Variant dispatch (today implicit in flags) | `build_diffuser_actor_policy(cfg)` factory in `policies/diffuser_actor.py`, called from `scripts/run_experiment.py:41-42` |

---

## 9. Risks / gotchas

1. **Checkpoint compatibility — sacred.** `policies/diffuser_actor_components/*` is byte-for-byte unchanged (§1). Each variant constructs the same `DiffuserActor(...)` with its own flag values — the same call as today. State-dict keys are unchanged. ✅

2. **Constructor signature unchanged: `__init__(self, cfg)`.** Each subclass takes the same dict the factory was handed; `BasePolicy.__init__(cfg)` is called via `super().__init__(cfg)` — same as today. No signature change, no Hydra-kwargs adapter needed.

3. **`DiffuserActorPolicy` symbol preservation.** Today `from policies.diffuser_actor import DiffuserActorPolicy` is used at `scripts/run_experiment.py:41`. After Step 7 callers prefer `build_diffuser_actor_policy`, but `DiffuserActorPolicy = build_diffuser_actor_policy` keeps the legacy call form working byte-for-byte for any out-of-tree code.

4. **`policies/__init__.py` re-exports.** Expand to surface the four concrete variant classes + the alias + the factory. See §3.6 for the exact block.

5. **`wire_steering` introspection.** Reads `policy._use_primitive_id` and `policy._use_object_id` via `getattr(..., False)`. Each subclass MUST set these as class-level constants. Plan above does this. Verified by the Step 9 matrix.

6. **Exception text preservation (team-lead's Refinement 1).** `set_primitive` / `set_object` / `_build_instruction` error strings preserve the user-action hint verbatim (the part telling users to add `steering=voxposer` or call `policy.set_primitive(idx)` manually). Only the leading class-context phrase may shift ("...policy was built with use_primitive_id=False" → "...policy is not primitive-conditioned"). Any external log-grep workflow keying on the hint keeps working.

7. **`mask_language` for Nolang.** The model's encoder branch `if self.use_instruction:` is False for Nolang, so `mask_language` is never read inside the model. Base class can pass it unconditionally.

8. **Variant-independent cfg knobs stay on base.** `pred_horizon`, `corrector_steps`, `corrector_step_size`, `_relative`, `_quaternion_format`, `_gripper_loc_bounds_np`, `_mask_language` — all set in `DiffuserActorBasePolicy.__init__`.

9. **CLIP machinery only built for Language variant.** Lazy-load is per-instance. Subclasses that don't need it (Nolang / Primitive / Primitive+Object) simply don't have the attributes. ✅ Memory win: no idle CLIP tokenizer in primitive runs.

10. **The diag block's `_log_count` increment.** Today `self._log_count += 1` runs at the end of the `_diag` block (L515). Preserve in the base class — all four log lines (rgb/pcd/gripper + variant) must come from the same forward pass.

11. **Step 7 grep.** Refactorer runs `grep -rn "DiffuserActorPolicy" --include="*.py"` to confirm no surprise callers. Expected hits: `policies/diffuser_actor.py` (new file containing the alias), `policies/__init__.py` (re-export), `scripts/run_experiment.py` (updated call site). Anything else is a hit to investigate.

12. **Single call site (team-lead's correction).** Only `scripts/run_experiment.py:41-42` needs the import/call swap. `scripts/run_evaluation.py:38` inherits via `from scripts.run_experiment import instantiate_policy`. The earlier draft's "two callers, two lines each" was a miscount.

13. **Training entry point** (`scripts/train_diffuser_actor.py` → `training/policies/diffuser_actor/trainer.py`) imports `DiffuserActor` (the nn.Module) from `policies.diffuser_actor_components`, NOT the wrapper. Untouched by this task.

14. **`cfg.policy.name == "diffuser_actor"` guard at `run_experiment.py:108`.** PCD-images guard. Reads the yaml's `name:` field, which stays in every yaml. ✅ Unchanged.

15. **`OmegaConf.to_container(cfg.policy, resolve=True)` conversion at L42** stays. The factory takes the resulting plain dict and variant classes access fields via `cfg.get("xxx", default)`. No change to interpolation semantics.

16. **Diag log line ordering.** Today's forward prints `[Diag] rgb` → `[Diag] pcd` → `[Diag] gripper[-1]` → variant-specific. The refactored template calls the hook after the shared lines, preserving order.

17. **`policies/diffuser_actor.py` import side effects.** The factory file re-imports all four subclass modules at module-load time, triggering their module-level imports (transformers etc.) at first import. Cost is one-time at startup, equivalent to today's monolith. Lazy imports inside the factory function are an option if startup time becomes an issue — not done by default.

---

## 10. Acceptance criteria

- `uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer` runs `open_drawer` end-to-end and produces the same success/log pattern as Task 1's post-refactor smoke run.
- All six yaml configs resolve to the right subclass per the Step 9 matrix.
- `policies/diffuser_actor.py` ≤ ~60 lines (factory + re-exports + alias).
- Each variant file is single-purpose and within its target range from §2.
- `ruff check policies/` and `ruff format --check policies/` clean.
- No new mypy errors introduced in `policies/`.
- `scripts/train_diffuser_actor.py` import surface unchanged (`from policies.diffuser_actor_components import DiffuserActor` still works).
- `policies/diffuser_actor_components/` is byte-for-byte unchanged (`git diff --stat policies/diffuser_actor_components/` shows zero lines).
- `from policies.diffuser_actor import DiffuserActorPolicy` still works (direct alias for `build_diffuser_actor_policy`; no warning).
- `from policies import LanguageDiffuserActorPolicy` (and the other three) works via `policies/__init__.py` re-exports.

---

## 11. Out of scope for Task 2

- **`policies/diffuser_actor_components/*` (all eight files)** — byte-for-byte frozen. The encoder's dtype-dispatch on `encode_instruction` stays as the polymorphic boundary into the model. Future scope, not blocking anything.
- **Trainer / dataset refactor** — `training/policies/diffuser_actor/trainer.py` has analogous flag entanglement but is its own task. Not touched. Training entry point keeps instantiating `DiffuserActor` directly from `policies.diffuser_actor_components`.
- **Steering** (Task 1 territory) — callbacks reach `set_primitive`/`set_object` via the same duck-typing pattern; nothing changes there.
- **VoxPoser prompts** (Task 3), grasp-gate verification (Task 4), visualization cleanup (Task 5).
- **Yaml schema changes** — `name`, `ckpt_path`, all hyperparams stay in place; the six yamls are byte-identical post-refactor.

### Captured as future scope (not blocking any current task)
- Cleaning up the `use_*` flag plumbing inside `DiffuserActor` / `Encoder` / `DiffusionHead`. Would require either a checkpoint-key-preserving refactor or a state-dict migration step. Not free; not urgent — the model file is well-organized and dtype-dispatches already.
- Dropping the `name: diffuser_actor` yaml field entirely if `run_experiment.py:108`'s PCD-images guard moves to an `isinstance(policy, DiffuserActorBasePolicy)` check.
- Switching the dispatch later to Hydra `_target_:` per-yaml. Considered for Task 2 and rejected as more invasive than the factory; revisit if/when other policies start growing variants and a uniform Hydra-instantiate pattern becomes preferable.
