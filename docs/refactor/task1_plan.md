# Task 1 — Refactor `steering/voxposer_steering.py`

**Owner:** planner → refactorer
**Branch:** `refactoring`
**Goal:** Split the 1583-line monolith into focused modules separating trajectory-space steering, rotation-space steering, adaptive scalers, and stage management. Preserve all existing behavior. Public API stays identical so `uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer` still works.

---

## 1. Current state — what's tangled

`VoxPoserSteering` bundles 7 concerns into one class:

| # | Concern | Today's location |
|---|---|---|
| 1 | StageSpec + composer-tuple parsing + vocabs | L43–115, L542–628, L1383–1429 |
| 2 | LMP system lifecycle + composer call | L423–509 |
| 3 | Stage state, transitions, refresh, loop-back, grasp gate, primitive/object callbacks | L240–315, L630–954 |
| 4 | Adaptive scalers (timestep / distance / step / rot-align) — inline conditionals | L1306–1377, L1466–1498 |
| 5 | Position-branch math (Tweedie x₀ + world coord conversion + value-map gradient lookup + ε build) | L1182–1304, L994–1086 |
| 6 | Rotation-branch math (x₀-rot + per-horizon SLERP + ε build) | L1088–1175, L1448–1566 |
| 7 | Composing the per-step guidance tensor (`get_guidance`) | L994–1176 |

Adaptive scalers and value-map math both live as private methods. Stage-management state (loop-back counters, grasp gate latch, LMP lifecycle, primitive/object dispatch) is intermixed with steering math via `_current_*` private fields.

---

## 2. Target file layout

```
steering/
  voxposer_steering.py     # slim orchestrator (~250–280 lines) — BaseSteering impl
  diffusion_utils.py       # Tweedie x₀, alpha_bar fetch, ε/dps coefficients
  coordinates.py           # PositionTransform: model↔world + voxel-gradient lookup
  position_field.py        # PositionFieldGuidance — value-map gradient → ε branch
  rotation_field.py        # RotationFieldGuidance — SO(3) target → ε branch (FIRST-CLASS)
  scalers.py               # ScalerContext + Timestep/Distance/Step/RotationAlignment + compose()
  stage_spec.py            # StageSpec dataclass, vocabs, parse_composer_stages, normalize_rot_target
  stage_manager.py         # StageManager — stages, transitions, grasp gate, loop-back, refresh, LMPs, callbacks
  tweedie.py               # UNCHANGED. May reuse diffusion_utils.py later (Task-out-of-scope follow-up).
```

`voxposer/lmp.py` is unchanged — composer remains a low-level LLM-program producer. The composer→steering handoff goes through `StageManager`, which both invokes the composer and parses its raw stage tuples via `parse_composer_stages` (a pure function in `stage_spec.py`).

Rationale: rotation-space machinery sits in its own first-class module (`rotation_field.py`) with a generic `compute(stage, ctx)` interface. Today the only "target source" is a single fixed rotation per stage; Task 3 can add a `RotationValueMap` target without touching the orchestrator. The scalers form a tiny strategy pattern (each is a `BaseScaler` with `compute(ctx) -> float`) so adding a new scaler is a one-class change. Stage management leaves the steering module entirely.

---

## 3. Public-API contract (what stays externally identical)

`scripts/run_experiment.py`, `scripts/run_evaluation.py`, and `policies/diffuser_actor.py` call these on a `VoxPoserSteering` instance. Every one must keep working unchanged:

```python
# Construction
VoxPoserSteering(cfg)                                    # accepts the same cfg dict
steering.guidance_mode                                   # 'epsilon' | 'dps'  (read by policy)
steering._value_map                                      # read by run_evaluation.py:218 + run_experiment.py:215
                                                         #   → expose via @property delegating to stage_manager
steering._lmp_config                                     # mutated by run_experiment.py:104 to set visualization_save_dir
                                                         #   → keep as plain attr on orchestrator, share dict with StageManager

# Schedulers
steering.set_position_scheduler(scheduler)
steering.set_rotation_scheduler(scheduler)

# Per-step state setters (called by policy.forward)
steering.set_current_gripper_pos(np.ndarray)
steering.set_current_gripper_rotation(np.ndarray)        # forward-compat plumbing, kept

# Callbacks (wired once by runner)
steering.set_primitive_callback(fn)
steering.set_object_callback(fn)

# Episode lifecycle
steering.setup_episode(task_name, instruction=..., robot_obs=..., scene_obs=...,
                       fixture_positions=..., block_aabbs=...)  -> (None, None)

# Per-env-step callbacks (called from step_callback in runners)
steering.increment_step()
steering.refresh_costmap(robot_obs, scene_obs, fixture_positions=..., block_aabbs=...)
steering.check_stage_transition(ee_pos, gripper_width) -> bool

# Guidance call (called from diffuser_actor_model.py's denoising loop)
steering.get_guidance(current_sample, timestep, obs_embedding, model_output) -> tensor

# Live viewer
steering.get_costmap_state(ee_pos) -> dict | None
```

The cfg schema in `conf/steering/voxposer.yaml` is **not changed**. All knobs land in the same place externally; they're just consumed by different internal modules.

---

## 4. New module signatures (public contracts)

### `stage_spec.py`

```python
PRIMITIVE_VOCAB: dict[str, int]
OBJECT_VOCAB: dict[str, int]
ARTICULATED_TARGET_TASKS: frozenset[str]
VALID_STAGE_MODES: set[str]

@dataclass
class StageSpec:
    aff_fn: Optional[Callable]
    avoid_fn: Optional[Callable]
    mode: str = "static"
    primitive: Optional[str] = None
    object: Optional[str] = None
    rot_target: Any = None
    # Mutated by StageManager on first activation (static caches):
    cached_affordance: Optional[np.ndarray] = field(default=None, repr=False)
    cached_avoidance: Optional[np.ndarray] = field(default=None, repr=False)
    cached_target: Optional[np.ndarray] = field(default=None, repr=False)
    cached_rotation: Optional[np.ndarray] = field(default=None, repr=False)

def parse_composer_stages(
    raw_result, *, default_mode: str = "static",
) -> list[StageSpec]:
    """Tuple/list-of-tuples from composer → validated StageSpec list.
    Drops invalid stages with a warning. Returns [] on completely unparseable input."""

def normalize_rot_target(value, *, idx: int) -> Optional[np.ndarray]:
    """Resolve composer rot_target into canonical (6,) ortho-6D. None on failure."""
```

### `diffusion_utils.py`

```python
def get_alpha_bar(scheduler, timestep, *, device,
                  clamp_min: float = 1e-6, clamp_max: float = 1.0 - 1e-6) -> torch.Tensor:
    """ᾱ_t for any DDPM-style scheduler. Clamped for numerical safety."""

def predict_x0(x_t: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor,
               *, prediction_type: str = "epsilon") -> torch.Tensor:
    """Tweedie's formula. Supports 'epsilon' and 'v_prediction'."""

def epsilon_coeff(alpha_bar: torch.Tensor) -> torch.Tensor:
    """sqrt(ᾱ) / sqrt(1-ᾱ) — multiplier converting Δx₀ (model space) → Δε."""

def dps_coeff(alpha_bar: torch.Tensor) -> torch.Tensor:
    """1 / sqrt(ᾱ) — multiplier for DPS-style correction on x_{t-1}."""
```

### `coordinates.py`

```python
class PositionTransform:
    """Diffuser Actor's gripper-relative + normalized → world coordinate frame.
    Holds the per-step gripper pos. One instance per VoxPoserSteering."""

    def __init__(self, *, gripper_loc_bounds: Optional[np.ndarray],
                 workspace_min: np.ndarray, workspace_max: np.ndarray,
                 is_relative: bool, device: str): ...

    def set_gripper_pos(self, gripper_pos: np.ndarray) -> None: ...

    @property
    def current_gripper_pos(self) -> Optional[torch.Tensor]: ...

    def model_to_world(self, model_pos: torch.Tensor) -> torch.Tensor:
        """(B,H,3) model [-1,1] → (B,H,3) absolute world (meters)."""

    def world_gradient_to_model(self, grad_world: torch.Tensor) -> torch.Tensor:
        """Chain rule: ∂model/∂world is constant per-axis."""

    def lookup_voxel_gradient(self, positions_world: torch.Tensor,
                              gradient_field: torch.Tensor,
                              map_size: int) -> torch.Tensor:
        """(B,H,3) world positions → (B,H,3) gradient by nearest-voxel lookup."""
```

### `scalers.py`

```python
@dataclass
class ScalerContext:
    """All inputs any scaler might need. Built once per get_guidance call."""
    timestep: int
    num_train_timesteps: Optional[int]
    ee_pos: Optional[torch.Tensor]              # (3,) — for DistanceScaler
    stage_target: Optional[np.ndarray]          # (3,) — for DistanceScaler
    steps_in_stage: int                          # for StepScaler
    rot_pred_6d: Optional[torch.Tensor] = None  # (B,H,6) — for RotationAlignmentScaler
    rot_target_6d: Optional[torch.Tensor] = None  # (6,)

class BaseScaler:
    enabled: bool
    def compute(self, ctx: ScalerContext) -> float: ...

class TimestepScaler(BaseScaler):
    """min_scale + (1-min_scale) * t/T."""

class DistanceScaler(BaseScaler):
    """Linear ramp on ||ee_pos - stage_target||."""

class StepScaler(BaseScaler):
    """Linear ramp on env-steps spent in the current stage."""

class RotationAlignmentScaler(BaseScaler):
    """Linear ramp on Frobenius distance ||R_pred - R_target||."""

def compose(scalers: Iterable[BaseScaler], ctx: ScalerContext) -> float:
    """Multiply enabled scalers' outputs. Returns 1.0 when nothing enabled."""
```

### `position_field.py`

```python
class PositionFieldGuidance:
    """Trajectory-position branch: value-map gradient → ε guidance.

    Reads StageActivation.{value_map, gradient_field}; predicts x₀ via Tweedie,
    converts to world, samples the gradient, and folds it into ε.
    """
    def __init__(self, *, horizon: int, guidance_strength: float,
                 prediction_type: str, guidance_mode: str,
                 start_guidance_timestep: int,
                 coordinates: PositionTransform,
                 scalers: list[BaseScaler]): ...

    def compute(self, *, x_t: torch.Tensor, eps: torch.Tensor,
                timestep: int, alpha_bar: torch.Tensor,
                stage: "StageActivation", ctx: ScalerContext,
                episode_step: int) -> Optional[torch.Tensor]:
        """Returns (B,H,3) delta or None when stage has no value map / timestep gate."""
```

### `rotation_field.py`

```python
class RotationFieldGuidance:
    """SO(3) branch: per-horizon SLERP target → ε guidance.

    Today's only target is a single 6D rotation per stage (StageActivation.rotation_target_6d).
    Generic enough to host a value-map-on-rotations target in a future task without
    touching this class — the only contract is "stage.rotation_target_6d exists".
    """
    def __init__(self, *, horizon: int, guidance_strength: float,
                 guidance_mode: str, start_guidance_timestep: int,
                 rot_horizon_floor: float, rot_horizon_alpha_max: float,
                 scalers: list[BaseScaler]): ...

    def compute(self, *, x_t_rot: torch.Tensor, eps_rot: torch.Tensor,
                timestep: int, alpha_bar_rot: torch.Tensor,
                stage: "StageActivation", ctx: ScalerContext,
                episode_step: int) -> Optional[torch.Tensor]:
        """Returns (B,H,6) delta or None when stage has no rot_target / timestep gate."""

    @staticmethod
    def slerp_targets_per_horizon(x0_rot_pred: torch.Tensor,
                                  target_6d: torch.Tensor,
                                  alphas: torch.Tensor) -> torch.Tensor: ...
```

### `stage_manager.py`

```python
@dataclass
class StageActivation:
    """Snapshot of what the active stage exposes to the guidance branches."""
    value_map: Optional[ValueMap]
    gradient_field: Optional[torch.Tensor]        # (M,M,M,3)
    stage_target_world: Optional[np.ndarray]      # (3,)
    rotation_target_6d: Optional[torch.Tensor]    # (6,) on device
    primitive: Optional[str]
    object: Optional[str]
    stage_idx: int
    num_stages: int
    steps_in_stage: int

class StageManager:
    """Owns stage list, transitions, grasp gate, loop-back, LMP lifecycle,
    composer-stage parsing, and primitive/object callback dispatch."""

    def __init__(self, cfg, *, device: str, map_size: int,
                 workspace_min: np.ndarray, workspace_max: np.ndarray,
                 visualize: bool): ...

    # composer + lifecycle
    def setup_episode(self, task_name: str, *,
                      instruction: Optional[str],
                      robot_obs: Optional[np.ndarray],
                      scene_obs: Optional[np.ndarray],
                      fixture_positions: Optional[dict],
                      block_aabbs: Optional[dict]) -> bool:
        """Run composer, parse stages, activate stage 0. Returns False on failure."""

    def refresh(self, robot_obs: np.ndarray, scene_obs: np.ndarray, *,
                fixture_positions: Optional[dict],
                block_aabbs: Optional[dict]) -> None: ...

    # gates / transitions
    def check_transition(self, ee_pos: np.ndarray, gripper_width: float) -> bool: ...
    def increment_step(self) -> None: ...

    # accessors
    def current(self) -> StageActivation: ...
    def snapshot(self, ee_pos: np.ndarray) -> Optional[dict]:
        """Live-viewer schema — matches today's get_costmap_state output."""

    # callbacks
    def set_primitive_callback(self, fn: Callable[[int], None]) -> None: ...
    def set_object_callback(self, fn: Callable[[int], None]) -> None: ...
```

### `voxposer_steering.py` (slim orchestrator)

```python
class VoxPoserSteering(BaseSteering):
    def __init__(self, cfg):
        # Parse global knobs.
        # Build PositionTransform, StageManager, scalers, PositionFieldGuidance,
        # RotationFieldGuidance. Inject shared scaler instances where applicable.

    # — Setters delegate —
    def set_position_scheduler(self, s): ...
    def set_rotation_scheduler(self, s): ...
    def set_current_gripper_pos(self, p): self._coords.set_gripper_pos(p)
    def set_current_gripper_rotation(self, r): ...  # store, forward-compat
    def set_primitive_callback(self, fn): self._stage_manager.set_primitive_callback(fn)
    def set_object_callback(self, fn): self._stage_manager.set_object_callback(fn)

    # — Episode lifecycle —
    def setup_episode(self, task_name, instruction=None, robot_obs=None,
                      scene_obs=None, fixture_positions=None, block_aabbs=None):
        self._episode_step = 0
        self._stage_manager.setup_episode(task_name, instruction=instruction,
                                          robot_obs=robot_obs, scene_obs=scene_obs,
                                          fixture_positions=fixture_positions,
                                          block_aabbs=block_aabbs)
        return None, None

    def check_stage_transition(self, ee_pos, gripper_width):
        return self._stage_manager.check_transition(ee_pos, gripper_width)

    def refresh_costmap(self, robot_obs, scene_obs, fixture_positions=None,
                        block_aabbs=None):
        self._stage_manager.refresh(robot_obs, scene_obs,
                                    fixture_positions=fixture_positions,
                                    block_aabbs=block_aabbs)

    def increment_step(self):
        self._episode_step += 1
        self._stage_manager.increment_step()

    def get_costmap_state(self, ee_pos):
        return self._stage_manager.snapshot(ee_pos)

    @property
    def _value_map(self):
        """Compat shim — external code reads steering._value_map."""
        return self._stage_manager.current().value_map

    # — Guidance —
    def get_guidance(self, current_sample, timestep, obs_embedding, model_output):
        stage = self._stage_manager.current()
        ctx = ScalerContext(
            timestep=int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep),
            num_train_timesteps=self._position_scheduler.config.num_train_timesteps
                if self._position_scheduler else None,
            ee_pos=self._coords.current_gripper_pos,
            stage_target=stage.stage_target_world,
            steps_in_stage=stage.steps_in_stage,
            rot_target_6d=stage.rotation_target_6d,
        )
        alpha_bar_pos = get_alpha_bar(self._position_scheduler, timestep, device=self._device)
        alpha_bar_rot = get_alpha_bar(self._rotation_scheduler, timestep, device=self._device)

        guidance = torch.zeros_like(model_output)

        pos_delta = self._position_field.compute(
            x_t=current_sample, eps=model_output, timestep=timestep,
            alpha_bar=alpha_bar_pos, stage=stage, ctx=ctx,
            episode_step=self._episode_step,
        )
        if pos_delta is not None:
            H = pos_delta.shape[1]
            guidance[:, :H, :3] = pos_delta

        rot_delta = self._rotation_field.compute(
            x_t_rot=current_sample[..., 3:9], eps_rot=model_output[..., 3:9],
            timestep=timestep, alpha_bar_rot=alpha_bar_rot, stage=stage, ctx=ctx,
            episode_step=self._episode_step,
        )
        if rot_delta is not None:
            H = rot_delta.shape[1]
            guidance[:, :H, 3:9] = rot_delta

        return guidance.detach()
```

---

## 5. Ordered refactor steps (for `refactorer`)

Each step is self-contained and leaves the codebase running. Run the test command after every step:

```bash
uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer
```

(or, faster, import-check: `uv run python -c "from steering.voxposer_steering import VoxPoserSteering"`)

### Step 0 — Scaffold (no behavior change)
- Create empty modules: `stage_spec.py`, `diffusion_utils.py`, `coordinates.py`, `scalers.py`, `position_field.py`, `rotation_field.py`, `stage_manager.py`. Each gets a module docstring describing its role.

### Step 1 — `stage_spec.py` (pure data + parsing)
- Move `_PRIMITIVE_VOCAB` → `PRIMITIVE_VOCAB`.
- Move `_OBJECT_VOCAB` → `OBJECT_VOCAB`.
- Move `_ARTICULATED_TARGET_TASKS` → `ARTICULATED_TARGET_TASKS`.
- Move `_VALID_STAGE_MODES` → `VALID_STAGE_MODES`.
- Move `StageSpec` dataclass (unchanged).
- Convert `_parse_stage` method → free function `parse_composer_stages(raw_result, *, default_mode)`. Includes the tuple-vs-list-of-tuples wrap that today lives at L496–504. Returns `list[StageSpec]` (skips invalid stages with `logger.warning`). Returns `[]` when result type is unrecognized.
- Move `_normalize_rot_target` → free function `normalize_rot_target(value, *, idx)`.
- The "primitive missing" / "object missing" `ValueError`s currently raised in `setup_episode` (L514–533) stay in the orchestrator/StageManager — they need `_set_primitive_fn` etc. to decide whether to enforce.
- Update `voxposer_steering.py` imports.

### Step 2 — `diffusion_utils.py` (shared Tweedie math)
- Implement `get_alpha_bar`, `predict_x0`, `epsilon_coeff`, `dps_coeff`.
- `predict_x0` handles `'epsilon'` and `'v_prediction'` (same logic as today's `_predict_x0` at L1268–1304).
- Replace `_predict_x0`, `_get_alpha_bar`, `_get_alpha_bar_rot`, `_predict_x0_rot` call-sites in voxposer_steering.py with the new free functions. Delete the four methods.
- Do **not** touch `tweedie.py` even though it has equivalent logic — out of scope.

### Step 3 — `coordinates.py` (`PositionTransform`)
- Move `_model_to_world`, `_lookup_gradient` and the world-gradient-to-model Jacobian (currently inline at L1055–1060).
- Pull in state: `_gripper_loc_bounds`, `_workspace_min`, `_workspace_max`, `_is_relative`, `_current_gripper_pos`.
- Add `set_gripper_pos`, `current_gripper_pos` property.
- Wire the orchestrator's `set_current_gripper_pos` to call `self._coords.set_gripper_pos`.

### Step 4 — `scalers.py`
- Define `ScalerContext`, `BaseScaler`.
- Port `_compute_timestep_scale` → `TimestepScaler`.
- Port `_compute_distance_scale` → `DistanceScaler`.
- Port `_compute_step_scale` → `StepScaler`.
- Port `_compute_rotation_alignment_scale` → `RotationAlignmentScaler`. Note: today this method probes `x0_rot_pred[0, mid]` against `self._current_stage_target_rotation`; in the refactor those come from `ctx.rot_pred_6d` and `ctx.rot_target_6d`. The orchestrator will populate them (rotation branch fills `rot_pred_6d` after computing x₀_rot; for simplicity the rotation field can compute its own align-scale internally using ctx.rot_target_6d + the locally-computed prediction).
- Add `compose(scalers, ctx)`.
- Update voxposer_steering.py to instantiate scalers from cfg.

### Step 5 — `position_field.py` (`PositionFieldGuidance`)
- Class holds: horizon, strength, prediction_type, guidance_mode, start_guidance_timestep, `coordinates` ref, list of scalers (timestep + distance + step).
- `compute(...)` reproduces lines 1019–1086 verbatim semantically:
  - `t > start_guidance_timestep` → return `None`.
  - `stage.value_map is None or stage.gradient_field is None` → return `None`.
  - Predict x₀ via `predict_x0` with `alpha_bar`.
  - `model_to_world` on x₀'s position slice.
  - `lookup_voxel_gradient` on world positions.
  - `world_gradient_to_model` via PositionTransform.
  - Multiply by `compose(scalers, ctx)` × `strength` × (`epsilon_coeff` or `dps_coeff`).
  - Returns `(B, H, 3)`.
- Move the position-branch logger message (L1080–1086) here. Preserve "step % 10 == 0" frequency.

### Step 6 — `rotation_field.py` (`RotationFieldGuidance`)
- Class holds: horizon, strength, mode, start_guidance_timestep, rot_horizon_floor, rot_horizon_alpha_max, list of scalers (timestep + rot_alignment).
- `slerp_targets_per_horizon` moves as a `@staticmethod` (currently `_slerp_targets_per_horizon`, L1500–1566; verbatim).
- `compute(...)` reproduces L1088–1175 semantically:
  - `stage.rotation_target_6d is None` or `t > start_guidance_timestep` → return `None`.
  - Predict x₀_rot via `predict_x0` with `alpha_bar_rot`.
  - Compute per-horizon SLERP targets.
  - `grad_6d = x0_rot_pred - target_6d_per_h` (sign-flip preserved).
  - Multiply by scaler product × strength × (`epsilon_coeff` or `dps_coeff`).
  - Returns `(B, H, 6)`.
- Preserve the rotation-branch logger message (L1156–1174) including the chord-distance probe.

### Step 7 — `stage_manager.py`
- Class with all state: `_stages`, `_current_stage_idx`, `_value_map`, `_gradient_field`, `_current_stage_target`, `_current_stage_target_rotation`, `_current_primitive_name`, `_steps_in_stage`, `_steps_since_refresh`, `_loop_back_count`, `_steps_in_last_stage_basin`, `_task_uses_grasp_gate`, `_robot_obs`, `_lmps`, `_lmp_interface`, `_visualizer`, plus grasp-gate state and config knobs.
- Move: `_init_lmp_system`, the composer-run portion of `setup_episode` (L457–540), `_activate_stage` (L630–783), `check_stage_transition` → renamed `check_transition` (L790–850), `_maybe_loop_back` (L852–884), `_is_grasp_complete` (L886–915), `refresh_costmap` → renamed `refresh` (L917–954), `_eval_map` (L1568–1583).
- Move the visualizer instantiation + per-activation `visualize(...)` call (L429, L776–783) here.
- Move primitive/object callback dispatch (L745–759) here.
- Build `StageActivation` snapshots in `current()`.
- Implement `snapshot(ee_pos)` that reproduces the `get_costmap_state` dict schema (L967–988):
  ```python
  {
      'value_map', 'ee_pos', 'target', 'target_rotation',
      'objects', 'step' (episode-level — pass through?),
      'stage_idx', 'num_stages', 'instruction', 'primitive',
  }
  ```
  Note: `step` today is `self.current_episode_step` (orchestrator-owned). Pass it in via parameter, or have the orchestrator merge it into the dict returned from `snapshot`. **Choose: orchestrator merges** — keeps StageManager free of episode-step concerns.
- StageManager constructor takes `cfg` plus `device`, `map_size`, `workspace_min/max`, `visualize`. The cfg dict is also held by reference so external code that mutates `_lmp_config['visualization_save_dir']` (run_experiment.py:104) still works — both orchestrator and StageManager share the same dict.

### Step 8 — Slim `voxposer_steering.py`
- Final shape: orchestrator only, per the signatures above.
- Builds: `PositionTransform`, `StageManager`, four scalers, `PositionFieldGuidance` (with timestep + distance + step scalers), `RotationFieldGuidance` (with timestep + rot_alignment scalers).
- Holds: `_episode_step`, `_position_scheduler`, `_rotation_scheduler`, `_device`, `_current_gripper_rotation` (forward-compat).
- Provides `_value_map` property (compat shim).
- Keeps `_lmp_config` as a plain attribute (the same dict passed to StageManager).

### Step 9 — Update `steering/__init__.py`
- Probably no-op; `VoxPoserSteering` import path stays the same.
- Optionally export `StageManager`, `PositionFieldGuidance`, `RotationFieldGuidance` for downstream tooling — defer unless something else needs them.

### Step 10 — Smoke test
- `uv run python -c "from steering.voxposer_steering import VoxPoserSteering; print(VoxPoserSteering.__mro__)"`
- `uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer` — run a single task end-to-end. Compare log output sections (stage activations, grasp gate, position-branch + rotation-branch debug lines) against a pre-refactor reference run; numbers should match to within float jitter.

### Step 11 — Pre-PR cleanup
- Ensure each new module has a top-of-file docstring describing its responsibility + a short note about its public surface.
- `ruff format steering/`, `ruff check steering/`, `mypy steering/`.

---

## 6. Behavior preserved / removed / relocated

### Preserved (semantics identical)
- All public API method signatures.
- Tweedie x₀ formula and ε-space chain rule (`Δε = strength × scale × sqrt(ᾱ)/sqrt(1-ᾱ) × (x₀ - ref)` for position; analogous with rotation scheduler ᾱ_rot for rotation).
- `guidance_mode` switch ('epsilon' uses `epsilon_coeff`, 'dps' uses `dps_coeff`) in both branches.
- All four scaler ramps (linear, with `_full` / `_near` / `_floor` knobs).
- `start_guidance_timestep` / `start_guidance_timestep_rot` upper-bound gating.
- Per-horizon SLERP with quadratic alpha ramp from `rot_horizon_floor` to `rot_horizon_alpha_max`.
- Grasp gate: width history deque, min/max/spread check, latch-once-true, articulated-task disable, gate-blocked log throttle.
- Loop-back: dwell counter, radius, max-loops cap, restart-from-stage-0 reuse-cached behavior.
- Static-vs-track stage caching of affordance / avoidance / target / rotation arrays.
- Composer-tuple shape parsing: 2-/3-/4-/5-/6-tuple forms with the 5-tuple disambiguation on `raw[2]`.
- Primitive-id / object-id callback firing on each activation; the "missing primitive/object on stage N" `ValueError` at setup time.
- Per-episode reset of grasp history, stage step counter, loop-back count, basin counter, step-since-refresh.
- HTML visualizer call on non-refresh activations (suppressed during refresh).
- `get_costmap_state` dict schema (used by `visualization/manager.py`).
- All `logger.info` / `logger.debug` / `logger.warning` messages — preserve their content (text and the "step % 10 == 0" throttle), so log-grep workflows keep working.

### Removed
- **Nothing.** Brief is explicit: minimum-viable structural changes.

### Relocated (origin → destination)
| Origin (today) | Destination |
|---|---|
| `_PRIMITIVE_VOCAB`, `_OBJECT_VOCAB`, `_ARTICULATED_TARGET_TASKS`, `_VALID_STAGE_MODES` | `stage_spec.py` |
| `StageSpec` dataclass | `stage_spec.py` |
| `_parse_stage`, `_normalize_rot_target` | `stage_spec.py` (free functions) |
| `_init_lmp_system`, composer-call portion of `setup_episode` | `stage_manager.py` |
| `_activate_stage`, `check_stage_transition`, `_maybe_loop_back`, `_is_grasp_complete`, `refresh_costmap`, `increment_step` (stage-side counter), `_eval_map` | `stage_manager.py` |
| Grasp state (`_grasp_*`, `_gripper_width_history`, `_grasp_latched`, etc.), loop-back state, `_steps_in_stage`, `_robot_obs`, `_lmps`, `_lmp_interface`, `_visualizer`, `_task_uses_grasp_gate` | `stage_manager.py` |
| Primitive/object callback registration + dispatch | `stage_manager.py` |
| `get_costmap_state` (dict schema) | `stage_manager.snapshot(ee_pos)` + orchestrator merges episode-level `step` |
| `_model_to_world`, `_lookup_gradient`, `_gripper_loc_bounds`, `_workspace_min/max`, `_is_relative`, `_current_gripper_pos` | `coordinates.py` |
| `_predict_x0`, `_predict_x0_rot`, `_get_alpha_bar`, `_get_alpha_bar_rot` | `diffusion_utils.py` (free functions) |
| `_compute_timestep_scale`, `_compute_distance_scale`, `_compute_step_scale`, `_compute_rotation_alignment_scale` | `scalers.py` (classes) |
| Position-branch ε build (currently inside `get_guidance`, L1045–1086) | `position_field.py` |
| Rotation-branch ε build + per-horizon SLERP target construction (L1088–1175 + L1500–1566) | `rotation_field.py` |

---

## 7. Risks / gotchas

1. **`steering._value_map` is read externally** in `scripts/run_evaluation.py:218` and `scripts/run_experiment.py:215` to verify the composer succeeded. Solution: expose a `@property` on `VoxPoserSteering` that returns `self._stage_manager.current().value_map`. Don't break this contract.

2. **`steering._lmp_config['visualization_save_dir']` is mutated** by `scripts/run_experiment.py:104` *after* construction, *before* `setup_episode`. The lazy `_init_lmp_system` reads the dict on first episode setup. Solution: both orchestrator and StageManager hold the same dict reference. After-construction mutation propagates correctly.

3. **`ScalerContext` cycle on rotation alignment**: `RotationAlignmentScaler` reads predicted rotation, but the prediction is computed inside `RotationFieldGuidance.compute`. The clean answer is to **have `RotationFieldGuidance` compute its own align-scale internally** rather than pre-populating `ctx.rot_pred_6d` in the orchestrator. The scaler can live inside `rotation_field.py` as a private helper if it doesn't need to be reusable. (Decision: keep `RotationAlignmentScaler` as a public scaler but let `RotationFieldGuidance.compute` build the per-call context including `rot_pred_6d` after x₀-rot is computed.)

4. **`_steps_in_stage` is owned by `StageManager`** but read by `DistanceScaler` / `StepScaler`. The orchestrator pulls it via `stage_manager.current().steps_in_stage` and passes through `ScalerContext.steps_in_stage`. Same for `stage_target_world` (DistanceScaler).

5. **Device plumbing**. Many submodules need `device`. Pass at construction. Don't pass at call time.

6. **`episode_step` for log throttling**. Today `self.current_episode_step % 10 == 0` decides whether to emit the position/rotation debug line. Pass as `episode_step` kwarg to `compute(...)`. Don't make it a `ScalerContext` field — it's purely for logging.

7. **`_current_gripper_rotation` is plumbed in but unused** in v1. Keep the setter wired (forward-compat for a future align-decay on real EE rotation, per the comment at L386–392). Store on orchestrator.

8. **Forward references in module-level types**. `StageActivation` is defined in `stage_manager.py` but referenced in `position_field.py` / `rotation_field.py` signatures. Use `from __future__ import annotations` or `TYPE_CHECKING` to avoid cycles — `stage_manager.py` imports nothing from `position_field` / `rotation_field`, so the dependency arrow points the right way already.

9. **Composer stage-emission ownership** (open question raised in brief): The composer LLM emits raw stage tuples; today `VoxPoserSteering.setup_episode` parses them and stores `StageSpec`. After refactor: `StageManager.setup_episode` runs the LMP, then calls `parse_composer_stages` (pure helper) to validate & normalize. Composer (`voxposer/lmp.py`) is unchanged. Handoff direction stays one-way: composer → StageManager.

10. **`tweedie.py` overlap**. `TweedieSteering` has its own `_get_reference_window`, `_compute_timestep_scale`, etc. — duplicated logic. Out of scope: do not touch `tweedie.py`. After Task 1, `tweedie.py` can be refactored to share `diffusion_utils.py` + the `TimestepScaler` from `scalers.py` (~50 lines of dedup) — capture as a follow-up note, not part of this task.

11. **`run_evaluation.py` step-callback peculiarity**: lines 342–346 call `check_stage_transition` twice if `update_dash` is present, but only `update_dash` is gated on `hasattr`. This is upstream of the steering refactor — leave it alone, but be aware the second call is harmless because the stage state advances monotonically.

12. **Float jitter**. The orchestrator now applies the `epsilon_coeff` via a shared helper that clamps `1 - ᾱ` to `≥ 1e-6`. Today's inline code uses the same clamp value in both branches, so output should match bit-for-bit (or close to it). The smoke-test comparison should tolerate float jitter around the last 1–2 decimals.

13. **`set_lmp_objects` side-effect**. Called inside today's `setup_episode` (L484). Make sure it migrates to `StageManager.setup_episode` immediately after `update_state`.

14. **Static stage caching of `_steps_in_stage` reset**. Today `_activate_stage` resets `_steps_in_stage = 0` only when `not is_refresh` (L733). Preserve that branch in StageManager.

---

## 8. Out of scope for Task 1 (do not touch)

- Diffuser actor variant split (Task 2).
- Voxposer prompt + value-map redesign (Task 3) — including adding a `RotationValueMap` target. The first-class `RotationFieldGuidance` interface enables it, but Task 3 implements it.
- Multi-stage grasp-gate verification runs (Task 4).
- Visualization cleanup (Task 5).
- `tweedie.py` refactor — only note follow-up opportunity for shared `diffusion_utils.py` reuse.
- Any cfg-schema change in `conf/steering/voxposer.yaml`.

---

## 9. Acceptance criteria

- `uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer` runs end-to-end on at least one task.
- `voxposer_steering.py` ≤ ~300 lines.
- Each new module ≤ ~350 lines and has a single responsibility describable in one sentence.
- `ruff check steering/` and `ruff format --check steering/` pass.
- A pre-refactor and post-refactor run of the same task produce the same stage-transition log sequence, the same number of activations, and quantitatively similar guidance norms (float jitter ≤ ~1e-5).
- No regression in `scripts/run_experiment.py` (the other entry point also reads `steering._value_map`).
