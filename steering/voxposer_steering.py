"""VoxPoser value-map-based steering for diffusion policies.

Uses LLM-synthesized 3D value maps to guide the diffusion denoising
process via spatial gradients. At each guided denoising step, predicts the
clean trajectory x_0 via Tweedie's formula, denormalizes predicted positions
to world space, queries the precomputed gradient field of the value map, then
applies the gradient as guidance in either epsilon space or DPS (post-step)
style.

Two guidance modes are supported (toggled via guidance_mode config):
  - 'epsilon': Convert world-space gradient → epsilon space via Jacobian and
    add to ε before the scheduler step. No denoising loop changes required.
  - 'dps': Return gradient correction in model trajectory space; the
    denoising loop applies it to x_{t-1} after the scheduler step.

Steering is skipped entirely for timesteps above start_guidance_timestep,
preventing guidance from acting on pure-noise early steps where Tweedie
x_0 predictions are meaningless.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

from core.steering import BaseSteering
from policies.diffuser_actor_components.rotation_utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from voxposer.calvin_interface import voxel2pc
from voxposer.lmp import setup_lmp, set_lmp_objects
from voxposer.value_map import ValueMap
from voxposer.visualizer import ValueMapVisualizer

# Canonical primitive vocabulary (must match PrimitiveEmbedding training order).
# `rotate` is only meaningful in 6-tuple form (always paired with rot_target).
_PRIMITIVE_VOCAB = {"grasp": 0, "push": 1, "pull": 2, "place": 3, "rotate": 4}

# Canonical object vocabulary — ALPHABETICAL ORDER. Must match
# trainer.OBJECT_VOCAB (and the ordering used to train the
# primitive_object_ABCD checkpoint). Do not reorder without retraining.
_OBJECT_VOCAB = {
    "block": 0,
    "blue_block": 1,
    "drawer_handle": 2,
    "led_button": 3,
    "lightbulb_switch": 4,
    "pink_block": 5,
    "red_block": 6,
    "slider_handle": 7,
}

_VALID_STAGE_MODES = {"static", "track"}

# CALVIN tasks whose manipulation target is an articulated joint (slider,
# drawer, switch lever, button) rather than a movable block. The grasp-
# completion gate is disabled for these — the gripper doesn't actually close
# *around* the target, it contacts a handle, so the (min, max, stable) width
# criterion isn't a meaningful "grasp succeeded" signal here. Tasks that
# touch articulated objects but still grasp a block first (e.g. place_in_drawer,
# stack_block) are deliberately NOT in this set.
_ARTICULATED_TARGET_TASKS = frozenset({
    "open_drawer", "close_drawer",
    "move_slider_left", "move_slider_right",
    "turn_on_lightbulb", "turn_off_lightbulb",
    "turn_on_led", "turn_off_led",
    "push_into_drawer",
})

logger = logging.getLogger(__name__)


@dataclass
class StageSpec:
    """A single composer-emitted stage with optional refresh-mode metadata.

    `mode='static'` evaluates aff_fn/avoid_fn once at first activation and
    pins the resulting voxel arrays + target centroid for the rest of the
    stage's lifetime — refresh_costmap skips re-eval. Use this for affordances
    that compute a fixed offset from an object's pose (e.g. "15cm to the left
    of the slider handle"); re-evaluating with live state causes the target
    to chase the moving object.

    `mode='track'` re-evaluates each refresh tick, letting the affordance
    follow a moving object. Use for affordances anchored *at* an object.

    `primitive` is the motion primitive the base policy should be conditioned
    on for this stage. Constrained to the 4-symbol vocabulary
    {grasp, push, pull, place}; any other value is rejected at parse time.
    None is allowed only when the policy isn't primitive-id-conditioned.
    """

    aff_fn: Optional[Callable]
    avoid_fn: Optional[Callable]
    mode: str = "static"
    primitive: Optional[str] = None
    # Object name from _OBJECT_VOCAB. Forwarded to policy.set_object() when the
    # policy is built with use_object_id=True. None when the policy is
    # primitive-only (or CLIP-conditioned).
    object: Optional[str] = None
    # Optional rotation guidance target. Accepts a callable (resolved at
    # activation), a (3,3) matrix, a (6,) ortho-6D row, a (9,) flattened
    # matrix, or a (4,) wxyz quaternion. Normalized to canonical (6,) by
    # `_normalize_rot_target`. None disables rotation guidance for the stage.
    rot_target: Any = None
    cached_affordance: Optional[np.ndarray] = field(default=None, repr=False)
    cached_avoidance: Optional[np.ndarray] = field(default=None, repr=False)
    cached_target: Optional[np.ndarray] = field(default=None, repr=False)
    cached_rotation: Optional[np.ndarray] = field(default=None, repr=False)


class VoxPoserSteering(BaseSteering):
    """Steer diffusion policies using LLM-generated spatial value maps.

    At episode start, the LLM composer decomposes the task instruction into
    affordance/avoidance/gripper maps over a 3D voxel grid. During denoising,
    Tweedie's formula predicts x_0 from the current noisy trajectory, positions
    are denormalized to world space, the precomputed gradient field is queried,
    and the result is converted to either epsilon-space or DPS-style guidance.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.guidance_strength = cfg.get('guidance_strength', 1.0)
        self.horizon = cfg.get('horizon', 16)
        # Must be 'epsilon' for DiffuserActor (which uses epsilon prediction)
        self.prediction_type = cfg.get('prediction_type', 'epsilon')
        self.device = cfg.get('device', 'cuda')
        self.map_size = cfg.get('map_size', 100)

        # Guidance mode: 'epsilon' (modify ε before scheduler) or
        # 'dps' (correct x_{t-1} after scheduler step)
        self.guidance_mode = cfg.get('guidance_mode', 'epsilon')

        # Only apply guidance for timesteps <= this value (out of max_timesteps).
        # Steps above the threshold are pure noise — Tweedie predictions are
        # unreliable there.
        self.start_guidance_timestep = cfg.get('start_guidance_timestep', 50)

        # Timestep scaling (soft ramp within the guided window)
        self.use_timestep_scaling = cfg.get('use_timestep_scaling', True)
        self.min_timestep_scale = cfg.get('min_timestep_scale', 0.1)

        # Adaptive guidance decay — fades guidance as the EE enters the
        # affordance basin so the primitive-conditioned policy can drive
        # through. Both modes are opt-out.
        # Distance-decay: linear ramp on ||ee - stage_target||.
        self.use_distance_scaling = cfg.get('use_distance_scaling', False)
        self.distance_full = cfg.get('distance_full', 0.12)  # m: full guidance
        self.distance_near = cfg.get('distance_near', 0.04)  # m: floor
        self.distance_floor = cfg.get('distance_floor', 0.05)
        # Step-decay: linear ramp on env-steps spent in the current stage.
        self.use_step_scaling = cfg.get('use_step_scaling', False)
        self.step_full = cfg.get('step_full', 0)
        self.step_decay = cfg.get('step_decay', 80)
        self.step_floor = cfg.get('step_floor', 0.05)

        # Rotation guidance — opt-in per stage (composer must emit rot_target
        # in the 5-tuple stage form). All knobs are no-ops when no stage has a
        # rotation target, so existing rollouts behave identically until used.
        self.guidance_strength_rot = cfg.get(
            'guidance_strength_rot', self.guidance_strength
        )
        self.start_guidance_timestep_rot = cfg.get(
            'start_guidance_timestep_rot', self.start_guidance_timestep
        )
        # Alignment-decay on Tweedie x_0 rotation: linear ramp on the
        # Frobenius distance between predicted and target rotation matrices.
        self.use_rot_alignment_scaling = cfg.get('use_rot_alignment_scaling', False)
        self.rot_align_full = cfg.get('rot_align_full', 0.5)   # ≈ 30°
        self.rot_align_near = cfg.get('rot_align_near', 0.1)   # ≈ 6°
        self.rot_align_floor = cfg.get('rot_align_floor', 0.05)

        # Per-horizon SLERP fraction range. The rotation guidance computes a
        # per-horizon target as SLERP(R_pred_h, R_target, alpha_h) along the
        # SO(3) geodesic, with alpha_h ramping quadratically from `floor` at
        # h=0 up to `alpha_max` at h=H-1. The endpoint alpha controls HOW
        # FAR ALONG THE GEODESIC the predicted trajectory should reach
        # within a single policy call:
        #   alpha_max=1.0 → "complete the full rotation in this trajectory"
        #     — the diffuser has to fit a 90° wrist sweep across H horizon
        #     steps, which is much faster than training-distribution demos
        #     and the IK can't track it smoothly (visually pitchy).
        #   alpha_max=0.2 → "advance ~20% of the remaining rotation per
        #     trajectory" — closer to training-distribution motion speed.
        #     Convergence accumulates across env steps; smoother in sim.
        # `floor` keeps a small pull on the very first horizon step so the
        # trajectory starts rotating immediately rather than holding the
        # current pose for one step.
        self.rot_horizon_floor = cfg.get('rot_horizon_floor', 0.0)
        self.rot_horizon_alpha_max = cfg.get('rot_horizon_alpha_max', 0.3)

        # Workspace bounds (world coords, meters) — for value-map lookup
        self._workspace_min = np.array(
            cfg.get('workspace_bounds_min', [-0.35, -0.40, 0.40]),
            dtype=np.float32,
        )
        self._workspace_max = np.array(
            cfg.get('workspace_bounds_max', [0.35, 0.15, 0.85]),
            dtype=np.float32,
        )

        # DiffuserActor position normalization bounds (gripper_loc_bounds).
        # These define the normalization from gripper-relative world coords
        # to model-internal [-1, 1] space.
        glb = cfg.get('gripper_loc_bounds', None)
        self._gripper_loc_bounds: Optional[torch.Tensor] = (
            torch.tensor(glb, dtype=torch.float32) if glb is not None else None
        )
        self._is_relative = cfg.get('relative', True)

        # Current absolute gripper position (set per-step by policy wrapper)
        self._current_gripper_pos: Optional[torch.Tensor] = None
        # Current EE rotation (3x3) — plumbed for forward compatibility; v1
        # rotation guidance uses the Tweedie x_0 prediction, not real EE state.
        self._current_gripper_rotation: Optional[torch.Tensor] = None

        # Visualization
        self._visualize = cfg.get('visualize', False)
        self._visualizer: Optional[ValueMapVisualizer] = None

        # Lazy-init LMP system
        self._lmp_config = cfg
        self._lmps = None
        self._lmp_interface = None

        # Per-episode state (current active stage)
        self._value_map: Optional[ValueMap] = None
        self._gradient_field: Optional[torch.Tensor] = None  # (M,M,M,3)
        # Primitive name of the active stage (for visualization). Stays None
        # when the task isn't in the schema or the schema couldn't be loaded.
        self._current_primitive_name: Optional[str] = None

        # Multi-stage state
        self._stages: list[StageSpec] = []  # parsed from composer output
        self._current_stage_idx: int = 0
        self._current_stage_target: Optional[np.ndarray] = None  # (3,) world pos
        # Active rotation target as canonical (6,) ortho-6D row, on device.
        # None means rotation guidance is off for the current stage.
        self._current_stage_target_rotation: Optional[torch.Tensor] = None
        self._stage_proximity_threshold: float = cfg.get(
            'stage_proximity_threshold', 0.05
        )

        # Grasp-completion gate. For grasp stages we additionally require the
        # gripper width to settle above `grasp_min_width` for `grasp_stability_window`
        # consecutive steps before allowing the proximity-based transition to
        # fire. Filters out the common failure mode where the EE drifts into the
        # basin while the policy is still mid-grasp-attempt and the next stage
        # (lift/place) launches around an empty gripper.
        self._grasp_min_width: float = cfg.get('grasp_min_width', 0.02)
        # Upper bound — rules out "gripper still fully open hovering near the
        # object." Without this, a stable open gripper (width ≈ 0.08m on the
        # Franka) trivially passes the stability + min-width checks.
        self._grasp_max_width: float = cfg.get('grasp_max_width', 0.06)
        self._grasp_stability_window: int = cfg.get('grasp_stability_window', 5)
        self._grasp_stability_eps: float = cfg.get('grasp_stability_eps', 0.002)
        self._gripper_width_history: deque = deque(
            maxlen=self._grasp_stability_window
        )
        # Latch: once a grasp completes within a stage, keep returning True for
        # the rest of that stage. Without this, normal post-grasp dynamics
        # (lifting, micro-slip, controller wiggle) push width fluctuation past
        # `grasp_stability_eps` and the gate re-engages, leaving the stage
        # stuck oscillating around the basin forever.
        self._grasp_latched: bool = False
        # Per-episode flag: disable the grasp gate for tasks whose target is
        # an articulated joint (slider/drawer/switch/button) rather than a
        # graspable block. Set in setup_episode() from task_name. Defaults to
        # True so unknown tasks fall back to gating ON (the safe default for
        # the typical movable-block case).
        self._task_uses_grasp_gate: bool = True
        # Throttle "gate-blocked" log to once per N steps so a sustained block
        # doesn't spam the rollout output.
        self._grasp_block_log_interval: int = 20
        self._grasp_block_log_counter: int = 0

        # Primitive-ID conditioning (for primitive-mode Diffuser Actor).
        # The composer LLM is the source of truth: each stage tuple it emits
        # carries its own primitive name (grasp / push / pull / place / rotate)
        # plus optionally an object name. When the corresponding `_set_*_fn`
        # is registered, steering invokes it at each stage activation with
        # the integer id so the policy swaps its conditioning token.
        self._set_primitive_fn: Optional[Callable[[int], None]] = None
        self._set_object_fn: Optional[Callable[[int], None]] = None

        # Schedulers — set by the policy wrapper before inference
        self.position_scheduler = None
        self.rotation_scheduler = None

        # Dynamic costmap refresh
        self._refresh_enabled = cfg.get('refresh_costmap', True)
        self._refresh_interval = cfg.get('refresh_interval', 1)
        self._steps_since_refresh = 0

        # Last-stage loop-back: if the EE dwells in the final stage's basin
        # without the env signalling success, restart from stage 0. The cached
        # StageSpec list and (for static stages) cached aff/avoid/target arrays
        # are reused — no new LLM call. Opt-in; default off keeps prior behavior.
        self._loop_back_enabled: bool = cfg.get('loop_back_on_last_stage', False)
        self._loop_back_dwell_steps: int = cfg.get('last_stage_dwell_steps', 15)
        self._loop_back_radius: float = cfg.get(
            'last_stage_dwell_radius', self._stage_proximity_threshold
        )
        self._max_loop_backs: int = cfg.get('max_loop_backs', 3)
        self._loop_back_count: int = 0
        self._steps_in_last_stage_basin: int = 0

        # Default stage mode for legacy 2-tuple LLM outputs (no 3rd element).
        # 'static' = freeze affordance at first activation (safe — never chases
        #   moving offsets). 'track' = re-evaluate each refresh.
        self._default_stage_mode = cfg.get('default_stage_mode', 'static')
        if self._default_stage_mode not in _VALID_STAGE_MODES:
            logger.warning(
                f"Invalid default_stage_mode '{self._default_stage_mode}', "
                f"falling back to 'static'"
            )
            self._default_stage_mode = 'static'

        self.current_episode_step = 0
        # Stage-local step counter — resets on _activate_stage. Used by the
        # step-decay scale so freshly activated stages start with full guidance.
        self._steps_in_stage = 0
        self._robot_obs: Optional[np.ndarray] = None  # cached for visualization

        logger.info(
            f"VoxPoserSteering: mode={self.guidance_mode}, "
            f"strength={self.guidance_strength}, "
            f"start_t={self.start_guidance_timestep}, "
            f"map_size={self.map_size}, prediction_type={self.prediction_type}"
        )

    # ------------------------------------------------------------------
    # Scheduler setters
    # ------------------------------------------------------------------

    def set_position_scheduler(self, scheduler):
        """Store reference to position noise scheduler."""
        self.position_scheduler = scheduler

    def set_rotation_scheduler(self, scheduler):
        """Store reference to rotation noise scheduler."""
        self.rotation_scheduler = scheduler

    def set_primitive_callback(self, fn: Callable[[int], None]) -> None:
        """Register a callback invoked at each stage transition with the primitive id.

        Wire with `voxposer.set_primitive_callback(policy.set_primitive)` when
        using the primitive-id-conditioned Diffuser Actor. Leaving this unset
        is a no-op (e.g. for CLIP or no-language policies).
        """
        self._set_primitive_fn = fn

    def set_object_callback(self, fn: Callable[[int], None]) -> None:
        """Register a callback invoked at each stage transition with the object id.

        Wire with `voxposer.set_object_callback(policy.set_object)` when using
        the primitive+object-conditioned Diffuser Actor. Requires the LLM
        composer to emit an `object` slot in every stage tuple.
        """
        self._set_object_fn = fn

    def set_current_gripper_pos(self, gripper_pos: np.ndarray):
        """Set current absolute gripper position for relative coordinate conversion.

        Called by policy wrapper before each forward pass.

        Args:
            gripper_pos: (3,) absolute gripper XYZ position
        """
        self._current_gripper_pos = torch.tensor(
            gripper_pos, dtype=torch.float32, device=self.device
        )

    def set_current_gripper_rotation(self, gripper_rot) -> None:
        """Set current EE rotation. Stored for forward compatibility.

        v1 rotation guidance computes alignment-decay from the Tweedie x_0
        prediction, so this state isn't read inside `get_guidance`. Kept
        plumbed so a v2 decay variant on real EE rotation requires no wiring
        changes upstream.

        Args:
            gripper_rot: (3,) Euler XYZ, (4,) wxyz quaternion, or (3,3) matrix.
        """
        rot = np.asarray(gripper_rot, dtype=np.float32)
        if rot.shape == (3,):
            # Euler XYZ → 3x3, pytorch3d "XYZ" intrinsic convention
            # (Rx @ Ry @ Rz) — must match Diffuser Actor's convert_rotation.
            rx, ry, rz = float(rot[0]), float(rot[1]), float(rot[2])
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
            mat = Rx @ Ry @ Rz
        elif rot.shape == (4,):
            w, x, y, z = rot
            mat = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ], dtype=np.float32)
        elif rot.shape == (3, 3):
            mat = rot.astype(np.float32)
        else:
            logger.warning(f"set_current_gripper_rotation: unexpected shape {rot.shape}")
            return
        self._current_gripper_rotation = torch.from_numpy(mat).to(self.device)

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------

    def _init_lmp_system(self):
        """Lazily initialize the LMP hierarchy."""
        if self._lmps is not None:
            return
        self._lmps, self._lmp_interface = setup_lmp(self._lmp_config)
        if self._visualize:
            self._visualizer = ValueMapVisualizer(self._lmp_config)
        logger.info("Initialized VoxPoser LMP system")

    def setup_episode(self, task_name: str, instruction: str = None,
                      robot_obs: np.ndarray = None,
                      scene_obs: np.ndarray = None,
                      fixture_positions: dict = None,
                      block_aabbs: dict = None):
        """Generate value maps for a new episode via LLM composer.

        The composer returns either:
          - A list of stage tuples: [(aff, avoid), ...]
          - A single tuple: (aff, avoid)

        Each stage gets its own ValueMap + gradient field. Stage transitions
        happen at runtime via check_stage_transition().

        Args:
            task_name: CALVIN task name
            instruction: Natural language instruction (defaults to task_name)
            robot_obs: (15,) robot state for object detection
            scene_obs: (24,) scene state for object detection
            fixture_positions: Live fixture positions from PyBullet (optional)
            block_aabbs: Live orientation-aware block AABBs from PyBullet (optional)

        Returns:
            (None, None) for compatibility with TweedieSteering interface
        """
        self.current_episode_step = 0
        self._stages = []
        self._current_stage_idx = 0
        self._current_stage_target = None
        self._steps_since_refresh = 0
        self._loop_back_count = 0
        self._steps_in_last_stage_basin = 0
        self._robot_obs = robot_obs
        self._task_uses_grasp_gate = task_name not in _ARTICULATED_TARGET_TASKS
        if not self._task_uses_grasp_gate:
            logger.info(
                f"Grasp-completion gate disabled for articulated-target task "
                f"'{task_name}'"
            )
        self._init_lmp_system()

        if instruction is None:
            instruction = task_name.replace('_', ' ')

        # Update scene state
        if robot_obs is not None and scene_obs is not None:
            self._lmp_interface.update_state(
                robot_obs, scene_obs, fixture_positions, block_aabbs
            )

        # Set object context for LMPs
        object_names = self._lmp_interface.get_object_names()
        set_lmp_objects(self._lmps, object_names)

        # Run composer to generate value maps
        logger.info(f"Running VoxPoser composer for: '{instruction}'")
        try:
            result = self._lmps['composer'](instruction)
        except Exception as e:
            logger.error(f"Composer failed for '{instruction}': {e}")
            self._value_map = None
            return None, None

        # Parse composer result into list of StageSpec
        if isinstance(result, list):
            raw_stages = result
        elif isinstance(result, tuple):
            raw_stages = [result]
        else:
            logger.warning(f"Unexpected composer result type: {type(result)}")
            self._value_map = None
            return None, None

        self._stages = [self._parse_stage(s, i) for i, s in enumerate(raw_stages)]
        self._stages = [s for s in self._stages if s is not None]
        if not self._stages:
            self._value_map = None
            return None, None

        # If the policy needs primitives, every stage must have one. Catch
        # this here so the failure mode is "LLM forgot a primitive" rather
        # than a delayed crash on the first forward().
        if self._set_primitive_fn is not None:
            missing = [i for i, s in enumerate(self._stages) if s.primitive is None]
            if missing:
                raise ValueError(
                    f"Primitive-id mode is active but the composer omitted "
                    f"primitives on stages {missing}. Each stage must be a "
                    f"4-tuple (aff_fn, avoid_fn, mode, primitive) where "
                    f"primitive ∈ {sorted(_PRIMITIVE_VOCAB)}. Update the "
                    f"composer prompt or fix the LLM output."
                )
        if self._set_object_fn is not None:
            missing = [i for i, s in enumerate(self._stages) if s.object is None]
            if missing:
                raise ValueError(
                    f"Object-id mode is active but the composer omitted "
                    f"objects on stages {missing}. Each stage must be a "
                    f"5-tuple (..., mode, primitive, object) or 6-tuple "
                    f"(..., rot_target, mode, primitive, object) where "
                    f"object ∈ {sorted(_OBJECT_VOCAB)}. Update the composer prompt."
                )

        logger.info(f"Composer returned {len(self._stages)} stage(s)")

        # Activate the first stage
        self._activate_stage(0)

        return None, None

    def _parse_stage(self, raw, idx: int) -> Optional[StageSpec]:
        """Normalize a composer stage entry into a StageSpec.

        Accepts:
          - (aff_fn, avoid_fn)                                                  → no primitive
          - (aff_fn, avoid_fn, 'static'|'track')                                → explicit mode, no primitive
          - (aff_fn, avoid_fn, 'static'|'track', primitive)                     → primitive
          - (aff_fn, avoid_fn, 'static'|'track', primitive, object)             → primitive + object
          - (aff_fn, avoid_fn, rot_target, 'static'|'track', primitive)         → primitive + rotation
          - (aff_fn, avoid_fn, rot_target, 'static'|'track', primitive, object) → primitive + object + rotation

        `primitive`, when present, MUST be in _PRIMITIVE_VOCAB.
        `object`, when present, MUST be in _OBJECT_VOCAB.
        Invalid values cause the stage to be dropped — we don't want the LLM
        inventing tokens the policy can't decode.
        """
        if not isinstance(raw, (tuple, list)):
            logger.warning(f"Stage {idx}: expected tuple/list, got {type(raw)}")
            return None

        def _check_primitive(p):
            if not (isinstance(p, str) and p in _PRIMITIVE_VOCAB):
                logger.error(
                    f"Stage {idx}: invalid primitive {p!r}; must be one "
                    f"of {sorted(_PRIMITIVE_VOCAB)}. Dropping stage."
                )
                return False
            return True

        def _check_object(o):
            if not (isinstance(o, str) and o in _OBJECT_VOCAB):
                logger.error(
                    f"Stage {idx}: invalid object {o!r}; must be one "
                    f"of {sorted(_OBJECT_VOCAB)}. Dropping stage."
                )
                return False
            return True

        def _normalize_mode(m):
            if isinstance(m, str) and m in _VALID_STAGE_MODES:
                return m
            logger.warning(
                f"Stage {idx}: invalid mode {m!r}, "
                f"falling back to default '{self._default_stage_mode}'"
            )
            return self._default_stage_mode

        if len(raw) == 2:
            return StageSpec(raw[0], raw[1], mode=self._default_stage_mode)
        if len(raw) == 3:
            return StageSpec(raw[0], raw[1], mode=_normalize_mode(raw[2]))
        if len(raw) == 4:
            mode, primitive = _normalize_mode(raw[2]), raw[3]
            if not _check_primitive(primitive):
                return None
            return StageSpec(raw[0], raw[1], mode=mode, primitive=primitive)
        if len(raw) == 5:
            # Two valid 5-tuple shapes:
            #   (aff, avoid, mode, primitive, object)         — primitive+object, no rotation
            #   (aff, avoid, rot_target, mode, primitive)     — primitive + rotation
            # Disambiguate on the type of raw[2]: a string mode vs anything else (rot target).
            if isinstance(raw[2], str) and raw[2] in _VALID_STAGE_MODES:
                mode, primitive, obj = _normalize_mode(raw[2]), raw[3], raw[4]
                if not _check_primitive(primitive) or not _check_object(obj):
                    return None
                return StageSpec(
                    raw[0], raw[1], mode=mode, primitive=primitive, object=obj,
                )
            rot_target, mode, primitive = raw[2], _normalize_mode(raw[3]), raw[4]
            if not _check_primitive(primitive):
                return None
            return StageSpec(
                raw[0], raw[1], mode=mode, primitive=primitive,
                rot_target=rot_target,
            )
        if len(raw) == 6:
            # (aff, avoid, rot_target, mode, primitive, object)
            rot_target, mode = raw[2], _normalize_mode(raw[3])
            primitive, obj = raw[4], raw[5]
            if not _check_primitive(primitive) or not _check_object(obj):
                return None
            return StageSpec(
                raw[0], raw[1], mode=mode, primitive=primitive, object=obj,
                rot_target=rot_target,
            )
        logger.warning(f"Stage {idx}: expected 2- to 6-tuple, got len={len(raw)}")
        return None

    def _activate_stage(self, idx: int, is_refresh: bool = False):
        """Build ValueMap and gradient field for stage `idx`.

        Evaluates the lazy map functions, creates a ValueMap, smooths it,
        precomputes gradients, and computes the stage's target position
        (centroid of raw affordance voxels in world space).

        Args:
            idx: Stage index to activate.
            is_refresh: If True, skip the (expensive) static HTML visualizer.
                Set by refresh_costmap() to avoid writing hundreds of HTML
                files per episode.
        """
        if idx >= len(self._stages):
            logger.warning(f"Stage {idx} out of range (have {len(self._stages)})")
            return

        spec = self._stages[idx]

        # Static stages: re-use frozen arrays after first activation.
        if spec.mode == 'static' and spec.cached_affordance is not None:
            affordance = spec.cached_affordance
            avoidance = spec.cached_avoidance
            logger.info(f"stage {idx}: static (cached)")
        else:
            affordance = self._eval_map(spec.aff_fn)
            avoidance = self._eval_map(spec.avoid_fn)
            if spec.mode == 'static' and affordance is not None:
                spec.cached_affordance = affordance
                spec.cached_avoidance = avoidance
                logger.info(f"stage {idx}: static (freezing now)")
            else:
                logger.info(f"stage {idx}: tracking (re-evaluated)")

        if affordance is None:
            logger.warning(f"Stage {idx}: no affordance map, steering disabled")
            self._value_map = None
            self._gradient_field = None
            self._current_stage_target = None
            self._current_stage_target_rotation = None
            self._current_stage_idx = idx
            return

        self._value_map = ValueMap(
            affordance=affordance,
            avoidance=avoidance,
            workspace_bounds_min=self._workspace_min,
            workspace_bounds_max=self._workspace_max,
            map_size=self.map_size,
            instruction=f"stage_{idx}",
        )
        self._value_map.smooth()
        self._value_map.precompute_gradients()

        # Precompute gradient field as torch tensor
        self._gradient_field = torch.from_numpy(
            np.stack([
                self._value_map._grad_x,
                self._value_map._grad_y,
                self._value_map._grad_z,
            ], axis=-1)
        ).float().to(self.device)

        # Stage target: cached for static, recomputed for track.
        if spec.mode == 'static' and spec.cached_target is not None:
            self._current_stage_target = spec.cached_target
        else:
            raw_aff = self._value_map._raw_affordance
            if raw_aff is not None and raw_aff.max() > 0:
                target_voxels = np.argwhere(raw_aff > 0)
                centroid_voxel = target_voxels.mean(axis=0).astype(int)
                target = voxel2pc(
                    centroid_voxel[np.newaxis],
                    self._workspace_min, self._workspace_max, self.map_size,
                )[0]
            else:
                target = None
            if spec.mode == 'static' and target is not None:
                spec.cached_target = target
            self._current_stage_target = target

        # Rotation target: same static/track caching pattern as position
        # target. Static stages cache after the first successful resolution;
        # track stages re-evaluate on refresh. None disables rotation guidance.
        if spec.rot_target is None:
            self._current_stage_target_rotation = None
        elif spec.mode == 'static' and spec.cached_rotation is not None:
            self._current_stage_target_rotation = torch.from_numpy(
                spec.cached_rotation
            ).float().to(self.device)
        else:
            rot6d = self._normalize_rot_target(spec.rot_target, idx)
            if rot6d is None:
                self._current_stage_target_rotation = None
            else:
                if spec.mode == 'static':
                    spec.cached_rotation = rot6d
                self._current_stage_target_rotation = torch.from_numpy(
                    rot6d
                ).float().to(self.device)

        self._current_stage_idx = idx
        if not is_refresh:
            self._steps_in_stage = 0
            self._gripper_width_history.clear()
            self._grasp_block_log_counter = 0
            self._grasp_latched = False

        # The composer LLM writes a primitive name into each stage; we trust
        # the parser to have already validated it against _PRIMITIVE_VOCAB.
        self._current_primitive_name = spec.primitive

        # Primitive-id conditioning: stage primitive -> int id -> policy.
        # If the policy needs a primitive but the LLM didn't emit one, that's
        # a prompt bug — the per-stage error is surfaced in setup_episode().
        if self._set_primitive_fn is not None and spec.primitive is not None:
            pid = _PRIMITIVE_VOCAB[spec.primitive]
            self._set_primitive_fn(pid)
            logger.info(
                f"Primitive-id set: stage {idx} -> '{spec.primitive}' (id={pid})"
            )

        # Object-id conditioning (parallel to primitive). Fired only when the
        # policy registered a callback AND the composer emitted an object slot.
        if self._set_object_fn is not None and spec.object is not None:
            oid = _OBJECT_VOCAB[spec.object]
            self._set_object_fn(oid)
            logger.info(
                f"Object-id set: stage {idx} -> '{spec.object}' (id={oid})"
            )

        logger.info(
            f"Activated stage {idx}/{len(self._stages) - 1}: "
            f"affordance max={affordance.max():.2f}, "
            f"non-zero={np.count_nonzero(affordance)}, "
            f"target={self._current_stage_target}"
        )
        if self._current_stage_target_rotation is not None:
            R_target = compute_rotation_matrix_from_ortho6d(
                self._current_stage_target_rotation.unsqueeze(0)
            ).squeeze(0).cpu().numpy()
            logger.info(
                f"Stage {idx} rotation target (3x3):\n{R_target}"
            )

        # Visualize if enabled (skip on refresh to avoid HTML spam)
        if self._visualizer is not None and not is_refresh:
            ee_pos = (
                self._robot_obs[:3] if self._robot_obs is not None else None
            )
            detections = self._lmp_interface.get_all_detections()
            self._visualizer.visualize(
                self._value_map, ee_pos_world=ee_pos, objects=detections
            )

    def increment_step(self):
        """Advance episode step counter."""
        self.current_episode_step += 1
        self._steps_in_stage += 1

    def check_stage_transition(
        self, ee_pos: np.ndarray, gripper_width: float
    ) -> bool:
        """Check if EE is close enough to current target to advance stage.

        Called at each environment step from the step callback.

        For grasp primitives, additionally gates the transition on grasp
        completion (gripper width has settled above `grasp_min_width`) so the
        next stage doesn't launch around an empty gripper while the policy is
        still mid-attempt.

        Args:
            ee_pos: (3,) absolute world-frame end-effector position
            gripper_width: scalar gripper opening (meters); from obs.ee_pose[6].

        Returns:
            True if stage was advanced
        """
        # Track width every call regardless of stage — keeps the buffer warm
        # for the moment a grasp stage starts gating against it.
        self._gripper_width_history.append(float(gripper_width))

        if self._current_stage_target is None:
            return False
        if self._current_stage_idx >= len(self._stages) - 1:
            return self._maybe_loop_back(ee_pos)

        dist = np.linalg.norm(ee_pos - self._current_stage_target)
        if dist >= self._stage_proximity_threshold:
            return False

        # Proximity satisfied. Apply the grasp-completion gate (skipped for
        # articulated-target tasks where the gripper contacts a handle rather
        # than closing around a block — see _ARTICULATED_TARGET_TASKS).
        if (
            self._task_uses_grasp_gate
            and self._current_primitive_name == "grasp"
            and not self._is_grasp_complete()
        ):
            self._grasp_block_log_counter += 1
            if self._grasp_block_log_counter % self._grasp_block_log_interval == 1:
                widths = list(self._gripper_width_history)
                logger.info(
                    f"Stage {self._current_stage_idx} transition gated by grasp "
                    f"check: dist={dist:.3f}m < {self._stage_proximity_threshold}m "
                    f"but gripper not closed on object "
                    f"(width history={['%.3f' % w for w in widths]}, "
                    f"min_width={self._grasp_min_width}, "
                    f"max_width={self._grasp_max_width}, "
                    f"stability_eps={self._grasp_stability_eps})"
                )
            return False

        next_idx = self._current_stage_idx + 1
        logger.info(
            f"Stage transition: {self._current_stage_idx} → {next_idx} "
            f"(dist={dist:.3f}m < threshold={self._stage_proximity_threshold}m)"
        )
        self._activate_stage(next_idx)
        return True

    def _maybe_loop_back(self, ee_pos: np.ndarray) -> bool:
        """If enabled, restart from stage 0 after a sustained last-stage dwell.

        Triggers when the EE has stayed within `_loop_back_radius` of the
        last stage's target for `_loop_back_dwell_steps` consecutive env
        steps. The runner only keeps stepping while the env hasn't fired
        success, so dwelling on the last stage = "we reached the goal
        location but the env disagrees" = retry the whole staged plan.
        Reuses cached StageSpec aff/avoid/target — no new LLM call.
        """
        if not self._loop_back_enabled or len(self._stages) < 2:
            return False
        if self._current_stage_target is None:
            return False
        dist = np.linalg.norm(ee_pos - self._current_stage_target)
        if dist < self._loop_back_radius:
            self._steps_in_last_stage_basin += 1
        else:
            self._steps_in_last_stage_basin = 0
        if (
            self._steps_in_last_stage_basin >= self._loop_back_dwell_steps
            and self._loop_back_count < self._max_loop_backs
        ):
            self._loop_back_count += 1
            self._steps_in_last_stage_basin = 0
            logger.info(
                f"Last-stage basin dwell ≥ {self._loop_back_dwell_steps} "
                f"steps without success — looping back to stage 0 "
                f"(loop {self._loop_back_count}/{self._max_loop_backs})"
            )
            self._activate_stage(0)
            return True
        return False

    def _is_grasp_complete(self) -> bool:
        """True when the gripper has settled around an object.

        Requires a full window of width samples that all fall in the open
        interval (`grasp_min_width`, `grasp_max_width`) — the lower bound
        rejects "closed on nothing" (width ≈ 0), the upper bound rejects
        "still fully open hovering near the object" (width ≈ Franka's ~0.08m
        open) — and whose spread is under `grasp_stability_eps` (rejects
        "still closing" frames where width is transiently mid-range).

        Latches once true for the current stage — see `_grasp_latched`.
        """
        if self._grasp_latched:
            return True
        if len(self._gripper_width_history) < self._grasp_stability_window:
            return False
        w_min = min(self._gripper_width_history)
        w_max = max(self._gripper_width_history)
        complete = (
            (w_min > self._grasp_min_width)
            and (w_max < self._grasp_max_width)
            and ((w_max - w_min) < self._grasp_stability_eps)
        )
        if complete:
            self._grasp_latched = True
            logger.info(
                f"Grasp latched on stage {self._current_stage_idx} "
                f"(width range=[{w_min:.3f}, {w_max:.3f}]m)"
            )
        return complete

    def refresh_costmap(self, robot_obs: np.ndarray, scene_obs: np.ndarray,
                        fixture_positions: dict = None,
                        block_aabbs: dict = None):
        """Re-evaluate current stage's costmap with updated scene state.

        Called at a fixed interval from the step callback. Updates the LMP
        interface with current object positions, then re-evaluates the
        cached LLM-generated map code (no new LLM call needed).

        Args:
            robot_obs: (15,) current robot proprioception
            scene_obs: (24,) current scene state (block positions, joint states)
            fixture_positions: Live fixture positions from PyBullet (optional)
            block_aabbs: Live orientation-aware block AABBs from PyBullet (optional)
        """
        if not self._refresh_enabled:
            return
        if self._value_map is None or not self._stages:
            return

        self._steps_since_refresh += 1
        if self._steps_since_refresh < self._refresh_interval:
            return
        self._steps_since_refresh = 0

        # Always update LMP state — the next stage's first activation needs
        # fresh positions even when the current stage is frozen.
        self._lmp_interface.update_state(
            robot_obs, scene_obs, fixture_positions, block_aabbs
        )
        self._robot_obs = robot_obs

        # Static stages are pinned at activation; skip re-evaluation entirely.
        if self._stages[self._current_stage_idx].mode == 'static':
            return

        # Track mode: re-evaluate cached LLM code with updated object positions.
        self._activate_stage(self._current_stage_idx, is_refresh=True)

    # ------------------------------------------------------------------
    # Live costmap state
    # ------------------------------------------------------------------

    def get_costmap_state(self, ee_pos: np.ndarray) -> Optional[dict]:
        """Snapshot the active costmap state for an external live viewer.

        Returns None when there's no value map yet (e.g. composer failed
        or before setup_episode). The returned dict matches the kwargs
        accepted by `VisualizationManager.update_costmap`.
        """
        if self._value_map is None:
            return None
        objects = (
            self._lmp_interface.get_all_detections()
            if self._lmp_interface else None
        )
        target_rot = (
            self._current_stage_target_rotation.detach().cpu().numpy()
            if self._current_stage_target_rotation is not None else None
        )
        return {
            'value_map': self._value_map,
            'ee_pos': ee_pos,
            'target': self._current_stage_target,
            'target_rotation': target_rot,
            'objects': objects,
            'step': self.current_episode_step,
            'stage_idx': self._current_stage_idx,
            'num_stages': len(self._stages),
            'instruction': getattr(self._value_map, 'instruction', '') or '',
            'primitive': self._current_primitive_name,
        }

    # ------------------------------------------------------------------
    # Core guidance computation
    # ------------------------------------------------------------------

    def get_guidance(self, current_sample: torch.Tensor, timestep: int,
                     obs_embedding: Any, model_output: torch.Tensor) -> torch.Tensor:
        """Compute value-map gradient guidance.

        Two independent branches both write into the returned guidance tensor:
          - Position branch: value-map gradient → model space → ε via Tweedie.
            Writes dims [0:3]. Active when a stage's affordance map exists.
          - Rotation branch: linear 6D delta toward the stage's target rotation
            → ε via Tweedie with the rotation scheduler. Writes dims [3:9].
            Active only when the stage carries a rot_target.

        Either branch is a no-op if its precondition isn't met. The openness
        slice [9:10] is never written.

        Args:
            current_sample: Noisy trajectory x_t, shape (B, L, D)
            timestep: Current diffusion timestep
            obs_embedding: Observation features (unused; fixed_inputs for DA)
            model_output: Model's epsilon prediction ε, shape (B, L, D)

        Returns:
            Guidance tensor, shape matching model_output.
            In 'epsilon' mode: added to ε before scheduler step.
            In 'dps' mode: added to x_{t-1} after scheduler step.
        """
        guidance = torch.zeros_like(model_output)

        t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        B, L, D = model_output.shape
        H = min(self.horizon, L)

        pos_active = (
            self._value_map is not None
            and self._gradient_field is not None
            and t <= self.start_guidance_timestep
        )
        rot_active = (
            self._current_stage_target_rotation is not None
            and t <= self.start_guidance_timestep_rot
        )

        if not pos_active and not rot_active:
            return guidance

        # Tweedie x_0 (full state) — used by the position branch directly and
        # by the rotation branch (only the rotation slice is read there).
        # The shared call avoids forward-pass duplication; the rotation slice
        # of x_0 is recomputed below with the rotation scheduler's ᾱ_rot
        # because the position scheduler's ᾱ would yield a wrong x_0_rot.
        x_0_pred = self._predict_x0(current_sample, timestep, model_output)

        # ---------------- Position branch ----------------
        if pos_active:
            model_pos = x_0_pred[:, :H, :3]
            world_pos = self._model_to_world(model_pos)
            grad_world = self._lookup_gradient(world_pos)

            # Convert world-space gradient → model (x_0) space via chain rule.
            # normalize_pos: model = (world_rel - pos_min) / (pos_max - pos_min) * 2 - 1
            # Jacobian: dmodel/dworld = 2 / (pos_max - pos_min)
            # For gradient: grad_model = grad_world * (pos_max - pos_min) / 2
            if self._gripper_loc_bounds is not None:
                bounds = self._gripper_loc_bounds.to(grad_world.device)
                scale_factor = (bounds[1] - bounds[0]) / 2.0
                grad_model = grad_world * scale_factor.unsqueeze(0).unsqueeze(0)
            else:
                grad_model = grad_world

            scale = self._compute_timestep_scale(timestep)
            distance_scale = self._compute_distance_scale()
            step_scale = self._compute_step_scale()
            adaptive_scale = distance_scale * step_scale
            alpha_bar = self._get_alpha_bar(timestep)

            if self.guidance_mode == 'epsilon':
                coeff = torch.sqrt(alpha_bar) / torch.sqrt(
                    torch.clamp(1.0 - alpha_bar, min=1e-6)
                )
                delta = self.guidance_strength * scale * adaptive_scale * coeff * grad_model
            else:
                coeff = 1.0 / torch.sqrt(torch.clamp(alpha_bar, min=1e-6))
                delta = self.guidance_strength * scale * adaptive_scale * coeff * grad_model

            guidance[:, :H, :3] = delta

            if self.current_episode_step % 10 == 0:
                logger.debug(
                    f"[VoxPoser/{self.guidance_mode}/pos] step={self.current_episode_step}, "
                    f"t={t}: norm={torch.norm(delta).item():.4f}, "
                    f"coeff={coeff.item():.4f}, scale={scale:.4f}, "
                    f"dist_scale={distance_scale:.4f}, step_scale={step_scale:.4f}, "
                    f"alpha_bar={alpha_bar.item():.4f}"
                )

        # ---------------- Rotation branch ----------------
        if rot_active:
            # Recompute the rotation x_0 slice with the *rotation* scheduler's
            # ᾱ_rot — the squaredcos schedule diverges from position's
            # scaled_linear, so reusing x_0_pred[:,:,3:9] would be incorrect.
            x_t_rot = current_sample[:, :H, 3:9]
            eps_rot = model_output[:, :H, 3:9]
            x0_rot_pred = self._predict_x0_rot(x_t_rot, timestep, eps_rot)

            # Per-horizon SLERP targets on the SO(3) geodesic.
            #
            # Each horizon step h gets target_h = SLERP(R_pred_h, R_target,
            # alpha_h). With a quadratic alpha schedule from `rot_horizon_floor`
            # at h=0 up to 1.0 at h=H-1, early predicted steps stay near the
            # current rotation and late steps reach the target — the predicted
            # trajectory naturally traces the great circle. CALVIN executes
            # the trajectory in order, so the wrist physically rotates along
            # the geodesic instead of being slammed to the target (which the
            # IK resolves via wrist pitch — the chaotic motion you saw).
            #
            # Linear (target_for_all_h - pred) is replaced by per-horizon
            # (pred_h - target_h_on_geodesic). The "go through R^6 interior"
            # failure mode of the linear delta cannot occur here: SLERP
            # endpoints are on SO(3) by construction, and the chord between
            # them is along the geodesic.
            target_6d_single = self._current_stage_target_rotation.to(
                x0_rot_pred.device
            )  # (6,)
            h_idx = torch.arange(
                H, device=x0_rot_pred.device, dtype=x0_rot_pred.dtype
            )
            # Quadratic ramp from `floor` at h=0 to `alpha_max` at h=H-1.
            # alpha_max < 1 means each policy call only nudges the trajectory
            # part way along the SO(3) geodesic; convergence accumulates
            # across env steps at the diffuser's training-distribution speed
            # rather than slamming the full rotation into one trajectory.
            alpha_span = self.rot_horizon_alpha_max - self.rot_horizon_floor
            alphas = self.rot_horizon_floor + alpha_span * (
                h_idx / max(H - 1, 1)
            ) ** 2
            target_6d_per_h = self._slerp_targets_per_horizon(
                x0_rot_pred, target_6d_single, alphas
            )

            # Sign convention (matches the position branch): construct a
            # "cost gradient" that points AWAY from the per-horizon target.
            # Tweedie's negative dx0/deps then flips it so realized δx_0
            # points TOWARD the target.
            grad_6d = x0_rot_pred - target_6d_per_h  # (B, H, 6)

            alpha_bar_rot = self._get_alpha_bar_rot(timestep)
            timestep_scale_rot = self._compute_timestep_scale(timestep)
            align_scale = self._compute_rotation_alignment_scale(x0_rot_pred)

            if self.guidance_mode == 'epsilon':
                coeff_rot = torch.sqrt(alpha_bar_rot) / torch.sqrt(
                    torch.clamp(1.0 - alpha_bar_rot, min=1e-6)
                )
            else:
                coeff_rot = 1.0 / torch.sqrt(torch.clamp(alpha_bar_rot, min=1e-6))

            delta_rot = (
                self.guidance_strength_rot * timestep_scale_rot * align_scale
                * coeff_rot * grad_6d
            )

            guidance[:, :H, 3:9] = delta_rot

            if self.current_episode_step % 10 == 0:
                # Log a sample x_0 chord-distance to target so we can see
                # whether the predicted rotation is moving toward the target
                # over episode steps. Probe at horizon midpoint.
                R_target = compute_rotation_matrix_from_ortho6d(
                    self._current_stage_target_rotation.unsqueeze(0)
                )
                R_pred = compute_rotation_matrix_from_ortho6d(
                    x0_rot_pred[0, x0_rot_pred.shape[1] // 2].unsqueeze(0)
                )
                chord_d = torch.norm(R_pred - R_target).item()
                logger.info(
                    f"[VoxPoser/{self.guidance_mode}/rot] step={self.current_episode_step}, "
                    f"t={t}: norm={torch.norm(delta_rot).item():.4f}, "
                    f"coeff_rot={coeff_rot.item():.4f}, "
                    f"align_scale={align_scale:.4f}, "
                    f"alpha_bar_rot={alpha_bar_rot.item():.4f}, "
                    f"chord_d(R_pred,R_target)={chord_d:.4f}"
                )

        return guidance.detach()

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------

    def _model_to_world(self, model_pos: torch.Tensor) -> torch.Tensor:
        """Convert model-internal normalized position → absolute world position.

        Reverses DiffuserActor's two-stage normalization:
          1. normalize_pos: model = (world_rel - pos_min) / (pos_max - pos_min) * 2 - 1
          2. convert2rel:   world_rel = world_abs - gripper_pos

        Args:
            model_pos: (B, H, 3) positions in model [-1, 1] space

        Returns:
            (B, H, 3) absolute world positions (meters)
        """
        if self._gripper_loc_bounds is not None:
            bounds = self._gripper_loc_bounds.to(model_pos.device)
            pos_min = bounds[0]
            pos_max = bounds[1]
            # Undo normalize_pos
            world_rel = (model_pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min
        else:
            world_rel = model_pos

        # Undo gripper-relative conversion
        if self._is_relative and self._current_gripper_pos is not None:
            gripper = self._current_gripper_pos.to(model_pos.device)
            world_abs = world_rel + gripper.view(1, 1, 3)
        else:
            world_abs = world_rel

        return world_abs

    def _lookup_gradient(self, positions: torch.Tensor) -> torch.Tensor:
        """Look up precomputed value-map gradient at world positions.

        Args:
            positions: (B, H, 3) absolute world-frame XYZ positions (meters)

        Returns:
            (B, H, 3) gradient vectors pointing toward increasing affordance
        """
        B, H, _ = positions.shape
        M = self.map_size

        ws_min = torch.tensor(self._workspace_min, device=positions.device, dtype=positions.dtype)
        ws_max = torch.tensor(self._workspace_max, device=positions.device, dtype=positions.dtype)

        # Normalize to [0, M-1] voxel indices
        clamped = torch.clamp(positions, ws_min, ws_max)
        voxel_float = (clamped - ws_min) / (ws_max - ws_min) * (M - 1)
        voxel_idx = torch.clamp(voxel_float.long(), 0, M - 1)

        flat_idx = voxel_idx.reshape(-1, 3)
        ix = flat_idx[:, 0]
        iy = flat_idx[:, 1]
        iz = flat_idx[:, 2]

        # Gradient field: (M, M, M, 3) — gradients of the value map in voxel space
        grad_flat = self._gradient_field[ix, iy, iz]  # (B*H, 3)

        # Scale from voxel-space gradient → world-space gradient (1/m)
        resolution = torch.tensor(
            (self._workspace_max - self._workspace_min) / M,
            device=positions.device, dtype=positions.dtype,
        )
        grad_world = grad_flat / resolution.unsqueeze(0)

        return grad_world.reshape(B, H, 3)

    # ------------------------------------------------------------------
    # Diffusion helpers
    # ------------------------------------------------------------------

    def _get_alpha_bar(self, timestep) -> torch.Tensor:
        """Get cumulative noise schedule value ᾱ_t for the position scheduler."""
        sched = self.position_scheduler
        if sched is None:
            return torch.tensor(0.5, device=self.device)

        if isinstance(timestep, torch.Tensor):
            t_idx = timestep.long()
        else:
            t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)

        alpha_bar = sched.alphas_cumprod[t_idx]
        return torch.clamp(alpha_bar, min=1e-6, max=1.0 - 1e-6)

    def _predict_x0(self, x_t: torch.Tensor, timestep,
                    model_output: torch.Tensor) -> torch.Tensor:
        """Apply Tweedie's formula to predict clean sample x_0 from x_t and ε.

        For epsilon prediction (DiffuserActor):
            x_0 = (x_t - sqrt(1-ᾱ) · ε) / sqrt(ᾱ)

        Uses position_scheduler for alpha_bar if available.
        """
        if self.prediction_type == 'sample':
            return model_output

        sched = self.position_scheduler
        if sched is None:
            logger.warning("No scheduler set, returning model_output as x_0")
            return model_output

        if isinstance(timestep, torch.Tensor):
            t_idx = timestep.long()
        else:
            t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)

        alpha_bar = torch.clamp(sched.alphas_cumprod[t_idx], min=1e-6)
        alpha_bar = alpha_bar.view(-1, 1, 1)

        # model_output may have more dims than x_t (e.g. openness appended)
        D = x_t.shape[-1]
        eps = model_output[..., :D]

        if self.prediction_type == 'epsilon':
            x_0 = (x_t - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        elif self.prediction_type == 'v_prediction':
            x_0 = torch.sqrt(alpha_bar) * x_t - torch.sqrt(1 - alpha_bar) * eps
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return x_0

    def _compute_timestep_scale(self, timestep) -> float:
        """Compute soft guidance scale based on diffusion timestep.

        Ramps from min_timestep_scale at t=0 to 1.0 at t=max_timesteps.
        Only applied within the guided window (t <= start_guidance_timestep).
        """
        if not self.use_timestep_scaling:
            return 1.0

        t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep

        if self.position_scheduler is None:
            return 1.0

        max_timestep = self.position_scheduler.config.num_train_timesteps
        normalized_t = t / max_timestep
        return self.min_timestep_scale + (1.0 - self.min_timestep_scale) * normalized_t

    def _compute_distance_scale(self) -> float:
        """Linear-ramp guidance scale based on EE → stage_target distance.

        Returns 1.0 (no decay) when the EE is far from the basin and ramps
        down to `distance_floor` once it has arrived. Lets the primitive-
        conditioned policy take over the actual contact motion (push, etc.)
        instead of being pinned at the centroid by ongoing guidance.
        """
        if not self.use_distance_scaling:
            return 1.0
        if self._current_stage_target is None or self._current_gripper_pos is None:
            return 1.0

        target = torch.tensor(
            self._current_stage_target,
            dtype=self._current_gripper_pos.dtype,
            device=self._current_gripper_pos.device,
        )
        d = torch.norm(self._current_gripper_pos - target).item()

        d_full = self.distance_full
        d_near = self.distance_near
        floor = self.distance_floor
        if d_full <= d_near:  # malformed config — treat as off
            return 1.0
        if d >= d_full:
            return 1.0
        if d <= d_near:
            return floor
        # Linear ramp between (d_near, floor) and (d_full, 1.0).
        return floor + (1.0 - floor) * (d - d_near) / (d_full - d_near)

    def _compute_step_scale(self) -> float:
        """Linear-ramp guidance scale based on env-steps spent in the current stage.

        Catches cases where distance-decay misfires (e.g., basin centroid is
        offset from the contact point and the EE never gets close enough to
        trigger distance decay).
        """
        if not self.use_step_scaling:
            return 1.0

        s = self._steps_in_stage
        s_full = self.step_full
        s_decay = self.step_decay
        floor = self.step_floor
        if s_decay <= 0:
            return 1.0
        if s <= s_full:
            return 1.0
        if s >= s_full + s_decay:
            return floor
        # Linear ramp from 1.0 (at s_full) down to floor (at s_full + s_decay).
        return 1.0 - (1.0 - floor) * (s - s_full) / s_decay

    # ------------------------------------------------------------------
    # Rotation guidance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_rot_target(value, idx: int) -> Optional[np.ndarray]:
        """Resolve a composer rot_target into a canonical (6,) ortho-6D row.

        Accepts:
          - callable: invoked with no args; result is fed back through this
            method so a callable returning a 3x3 (or any other shape below)
            works transparently.
          - (3,3) rotation matrix: Gram-Schmidt-orthonormalized via the
            existing ortho6d roundtrip, then sliced to 6D. Off-manifold inputs
            (LLM emits a non-orthonormal matrix) are silently corrected.
          - (6,) ortho-6D row: assumed valid; passed through.
          - (9,) flattened 3x3: reshaped, then treated as a 3x3.
          - (4,) wxyz quaternion: converted via quaternion_to_matrix.

        Returns None on failure (and logs a warning); the stage proceeds with
        rotation guidance disabled rather than crashing the rollout.
        """
        if value is None:
            return None
        try:
            if callable(value):
                value = value()
            arr = np.asarray(value, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Stage {idx}: rot_target eval failed: {e}")
            return None

        if arr.shape == (6,):
            return arr.astype(np.float32)
        if arr.shape == (9,):
            arr = arr.reshape(3, 3)
        if arr.shape == (4,):
            quat_t = torch.from_numpy(arr).float().unsqueeze(0)
            mat = quaternion_to_matrix(quat_t).squeeze(0).numpy()
            arr = mat.astype(np.float32)
        if arr.shape == (3, 3):
            mat_t = torch.from_numpy(arr).float().unsqueeze(0)
            ortho_t = compute_rotation_matrix_from_ortho6d(
                get_ortho6d_from_rotation_matrix(mat_t)
            )
            return get_ortho6d_from_rotation_matrix(ortho_t).squeeze(0).numpy().astype(np.float32)
        logger.warning(
            f"Stage {idx}: rot_target has unexpected shape {arr.shape}; "
            f"expected (3,3), (6,), (9,), or (4,)"
        )
        return None

    def _get_alpha_bar_rot(self, timestep) -> torch.Tensor:
        """Get cumulative noise schedule value ᾱ_t for the rotation scheduler.

        Required because rotation uses a different beta schedule (squaredcos)
        than position (scaled_linear); reusing the position scheduler's ᾱ here
        would yield an incorrect Tweedie x_0 estimate for the rotation slice.
        """
        sched = self.rotation_scheduler
        if sched is None:
            return torch.tensor(0.5, device=self.device)
        if isinstance(timestep, torch.Tensor):
            t_idx = timestep.long()
        else:
            t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)
        alpha_bar = sched.alphas_cumprod[t_idx]
        return torch.clamp(alpha_bar, min=1e-6, max=1.0 - 1e-6)

    def _predict_x0_rot(self, x_t_rot: torch.Tensor, timestep,
                        eps_rot: torch.Tensor) -> torch.Tensor:
        """Tweedie's formula on the rotation slice using the rotation scheduler.

        For epsilon prediction:
            x_0_rot = (x_t_rot - sqrt(1-ᾱ_rot) · ε_rot) / sqrt(ᾱ_rot)
        """
        sched = self.rotation_scheduler
        if sched is None:
            logger.warning("No rotation scheduler set, returning eps as x_0_rot")
            return eps_rot
        if isinstance(timestep, torch.Tensor):
            t_idx = timestep.long()
        else:
            t_idx = torch.tensor([timestep], device=self.device, dtype=torch.long)
        alpha_bar = torch.clamp(sched.alphas_cumprod[t_idx], min=1e-6).view(-1, 1, 1)
        return (x_t_rot - torch.sqrt(1 - alpha_bar) * eps_rot) / torch.sqrt(alpha_bar)

    def _compute_rotation_alignment_scale(self, x0_rot_pred: torch.Tensor) -> float:
        """Linear-ramp scale based on chordal distance ||R_pred − R_target||_F.

        Returns 1.0 when the predicted rotation is far from the target (full
        guidance) and `rot_align_floor` when it has aligned. Operates on the
        first batch element / first horizon step as a proxy for the whole
        trajectory; rotation targets in v1 are constant across the horizon.
        """
        if not self.use_rot_alignment_scaling:
            return 1.0
        if self._current_stage_target_rotation is None:
            return 1.0

        # Decode 6D → orthonormal 3x3 for both predicted and target.
        rot_target_6d = self._current_stage_target_rotation.unsqueeze(0)  # (1,6)
        R_target = compute_rotation_matrix_from_ortho6d(rot_target_6d)  # (1,3,3)

        # Probe trajectory midpoint as a cheap representative; chordal distance
        # at that index correlates well with whole-trajectory alignment.
        probe = x0_rot_pred[0, x0_rot_pred.shape[1] // 2].unsqueeze(0)  # (1,6)
        R_pred = compute_rotation_matrix_from_ortho6d(probe)  # (1,3,3)

        d = torch.norm(R_pred - R_target).item()  # Frobenius distance, ≤ 2*sqrt(2)
        d_full = self.rot_align_full
        d_near = self.rot_align_near
        floor = self.rot_align_floor
        if d_full <= d_near:
            return 1.0
        if d >= d_full:
            return 1.0
        if d <= d_near:
            return floor
        return floor + (1.0 - floor) * (d - d_near) / (d_full - d_near)

    @staticmethod
    def _slerp_targets_per_horizon(
        x0_rot_pred: torch.Tensor,
        target_6d_single: torch.Tensor,
        alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Per-horizon SLERP targets along the SO(3) geodesic.

        For each horizon step h, computes a 6D target = SLERP(R_pred_h,
        R_target, alphas[h]). The interpolation goes through the great-circle
        geodesic on SO(3), so the gradient (target - pred) points along the
        manifold rather than cutting through R^6.

        Quaternion SLERP is well-defined even at the 180° antipode (where
        chord_d hits 2*sqrt(2)): orthogonal quaternions, sin(π/2)=1, no
        singularity. Hemisphere fix ensures we always go the short way.

        Args:
            x0_rot_pred: (B, H, 6) predicted rotations in 6D.
            target_6d_single: (6,) the stage's final target rotation.
            alphas: (H,) per-horizon SLERP fractions in [0, 1]. 0 = stay at
                R_pred, 1 = reach R_target.

        Returns:
            (B, H, 6) per-horizon SLERP targets in 6D.
        """
        B, H, _ = x0_rot_pred.shape

        # Decode 6D → 3x3 → quaternion (wxyz convention from rotation_utils).
        R_pred = compute_rotation_matrix_from_ortho6d(
            x0_rot_pred.reshape(-1, 6)
        )  # (B*H, 3, 3)
        q_pred = matrix_to_quaternion(R_pred)  # (B*H, 4)

        R_target = compute_rotation_matrix_from_ortho6d(
            target_6d_single.unsqueeze(0)
        )  # (1, 3, 3)
        q_target = matrix_to_quaternion(R_target).squeeze(0)  # (4,)
        q_target = q_target.unsqueeze(0).expand(B * H, 4).contiguous()

        # Hemisphere fix: q and -q are the same rotation. Pick the sign that
        # gives a positive dot with q_pred so SLERP takes the short way.
        dot = (q_pred * q_target).sum(dim=-1, keepdim=True)
        q_target = torch.where(dot < 0, -q_target, q_target)
        dot = dot.abs().clamp(min=-1.0, max=1.0)

        theta = torch.acos(dot)  # (B*H, 1) — half the rotation angle
        sin_theta = torch.sin(theta)

        alphas_flat = alphas.view(1, H, 1).expand(B, H, 1).reshape(-1, 1)

        # SLERP with lerp fallback for tiny θ (q_pred ≈ q_target).
        SMALL = 1e-6
        use_slerp = sin_theta.abs() > SMALL
        sin_theta_safe = torch.where(
            use_slerp, sin_theta, torch.ones_like(sin_theta)
        )
        s1 = torch.sin((1.0 - alphas_flat) * theta) / sin_theta_safe
        s2 = torch.sin(alphas_flat * theta) / sin_theta_safe
        q_slerp = s1 * q_pred + s2 * q_target
        q_lerp = (1.0 - alphas_flat) * q_pred + alphas_flat * q_target
        q_interp = torch.where(use_slerp, q_slerp, q_lerp)
        q_interp = q_interp / q_interp.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        R_interp = quaternion_to_matrix(q_interp)  # (B*H, 3, 3)
        target_6d_per_h = get_ortho6d_from_rotation_matrix(R_interp)  # (B*H, 6)
        return target_6d_per_h.reshape(B, H, 6)

    @staticmethod
    def _eval_map(map_fn) -> Optional[np.ndarray]:
        """Evaluate a voxel map, handling callables and VoxelIndexingWrapper."""
        if map_fn is None:
            return None
        try:
            if callable(map_fn):
                result = map_fn()
            else:
                result = map_fn
            if hasattr(result, 'array'):
                return result.array
            return np.asarray(result)
        except Exception as e:
            logger.warning(f"Failed to evaluate map: {e}")
            return None
