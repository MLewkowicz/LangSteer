"""Per-episode stage lifecycle for VoxPoser steering.

Owns everything that lives between two `get_guidance` calls:

  * LMP system lazy init + composer invocation + stage parsing,
  * stage activation (value-map smoothing, gradient precompute, target centroid),
  * static / track caching of affordance, avoidance, target, rotation,
  * per-env-step refresh (re-eval track stages only),
  * EE-proximity stage transitions with the grasp-completion gate,
  * last-stage loop-back failure recovery,
  * primitive- and object-id callback dispatch to the policy,
  * HTML visualizer hookup,
  * live-viewer state snapshot.

The guidance branches (`position_field`, `rotation_field`) read a
`StageActivation` snapshot via `current()`; they never touch the manager's
mutable state directly.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

from policies.diffuser_actor_components.rotation_utils import (
    compute_rotation_matrix_from_ortho6d,
)
from steering.stage_spec import (
    ARTICULATED_TARGET_TASKS,
    OBJECT_VOCAB,
    PRIMITIVE_VOCAB,
    VALID_STAGE_MODES,
    StageSpec,
    normalize_rot_target,
    parse_composer_stages,
)
from voxposer.calvin_interface import ObjectResolutionError, pc2voxel, voxel2pc
from voxposer.lmp import (
    VocabValidationError,
    compose_with_repair,
    set_lmp_objects,
    setup_lmp,
)
from voxposer.value_map import ValueMap

logger = logging.getLogger(__name__)


@dataclass
class StageActivation:
    """Snapshot of the active stage's outputs, consumed by guidance branches."""

    value_map: Optional[ValueMap]
    gradient_field: Optional[torch.Tensor]  # (M, M, M, 3)
    stage_target_world: Optional[np.ndarray]  # (3,) absolute world
    rotation_target_6d: Optional[torch.Tensor]  # (6,) on device
    primitive: Optional[str]
    object: Optional[str]
    stage_idx: int
    num_stages: int
    steps_in_stage: int


class StageManager:
    """Owns multi-stage state, transitions, refresh, and callbacks."""

    def __init__(
        self,
        cfg: Any,
        *,
        device: str,
        map_size: int,
        workspace_min: np.ndarray,
        workspace_max: np.ndarray,
    ) -> None:
        # `cfg` is held by reference so external code that mutates it
        # post-construction (e.g. shared LMP config) is picked up at lazy
        # LMP-init time.
        self._cfg = cfg
        self._device = device
        self._map_size = map_size
        self._workspace_min = workspace_min
        self._workspace_max = workspace_max

        # Lazy LMP / composer system.
        self._lmps: Optional[dict] = None
        self._lmp_interface: Any = None

        # Active-stage state.
        self._stages: list[StageSpec] = []
        self._current_stage_idx: int = 0
        self._value_map: Optional[ValueMap] = None
        self._gradient_field: Optional[torch.Tensor] = None
        self._current_stage_target: Optional[np.ndarray] = None
        self._current_stage_target_rotation: Optional[torch.Tensor] = None
        self._current_primitive_name: Optional[str] = None

        # Callbacks.
        self._set_primitive_fn: Optional[Callable[[int], None]] = None
        self._set_object_fn: Optional[Callable[[int], None]] = None

        # Transition / refresh / loop-back config.
        self._stage_proximity_threshold: float = cfg.get(
            "stage_proximity_threshold", 0.05
        )
        self._refresh_enabled: bool = cfg.get("refresh_costmap", True)
        self._refresh_interval: int = cfg.get("refresh_interval", 1)
        self._steps_since_refresh: int = 0
        # Loop-back: when the policy has spent N policy-iters at the last stage
        # without the success oracle terminating the episode, restart from
        # stage 0. No proximity check — the oracle is the source of truth for
        # "actually succeeded"; we only count time-at-last-stage.
        self._loop_back_enabled: bool = cfg.get("loop_back_on_last_stage", False)
        self._loop_back_dwell_steps: int = cfg.get("last_stage_dwell_steps", 3)
        self._max_loop_backs: int = cfg.get("max_loop_backs", 3)
        self._loop_back_count: int = 0
        self._steps_in_last_stage: int = 0

        # Grasp-completion gate.
        self._grasp_min_width: float = cfg.get("grasp_min_width", 0.02)
        self._grasp_max_width: float = cfg.get("grasp_max_width", 0.06)
        self._grasp_stability_window: int = cfg.get("grasp_stability_window", 5)
        self._grasp_stability_eps: float = cfg.get("grasp_stability_eps", 0.002)
        self._gripper_width_history: deque = deque(maxlen=self._grasp_stability_window)
        self._grasp_latched: bool = False
        self._task_uses_grasp_gate: bool = True
        self._grasp_block_log_interval: int = 20
        self._grasp_block_log_counter: int = 0

        # Default stage mode for legacy 2-tuple LLM outputs.
        self._default_stage_mode: str = cfg.get("default_stage_mode", "static")
        if self._default_stage_mode not in VALID_STAGE_MODES:
            logger.warning(
                f"Invalid default_stage_mode '{self._default_stage_mode}', "
                f"falling back to 'static'"
            )
            self._default_stage_mode = "static"

        # Stage-local step counter — resets on stage activation. Read by
        # StepScaler via the orchestrator-built ScalerContext.
        self._steps_in_stage: int = 0
        # Cached robot_obs for the HTML visualizer (ee_pos overlay).
        self._robot_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def set_primitive_callback(self, fn: Callable[[int], None]) -> None:
        self._set_primitive_fn = fn

    def set_object_callback(self, fn: Callable[[int], None]) -> None:
        self._set_object_fn = fn

    # ------------------------------------------------------------------
    # LMP lifecycle / episode setup
    # ------------------------------------------------------------------

    def _init_lmp_system(self) -> None:
        if self._lmps is not None:
            return
        self._lmps, self._lmp_interface = setup_lmp(self._cfg)
        logger.info("Initialized VoxPoser LMP system")

    def setup_episode(
        self,
        task_name: str,
        *,
        instruction: Optional[str] = None,
        robot_obs: Optional[np.ndarray] = None,
        scene_obs: Optional[np.ndarray] = None,
        fixture_positions: Optional[dict] = None,
        block_aabbs: Optional[dict] = None,
    ) -> bool:
        """Run the composer, parse stages, activate stage 0.

        Returns True on success; False when the composer fails or emits no
        valid stages.
        """
        self._stages = []
        self._current_stage_idx = 0
        self._current_stage_target = None
        self._steps_since_refresh = 0
        self._loop_back_count = 0
        self._steps_in_last_stage = 0
        self._robot_obs = robot_obs
        self._task_uses_grasp_gate = task_name not in ARTICULATED_TARGET_TASKS
        if not self._task_uses_grasp_gate:
            logger.info(
                f"Grasp-completion gate disabled for articulated-target task "
                f"'{task_name}'"
            )
        self._init_lmp_system()

        if instruction is None:
            instruction = task_name.replace("_", " ")

        if robot_obs is not None and scene_obs is not None:
            self._lmp_interface.update_state(
                robot_obs, scene_obs, fixture_positions, block_aabbs
            )

        object_names = self._lmp_interface.get_object_names()
        set_lmp_objects(self._lmps, object_names)

        logger.info(f"Running VoxPoser composer for: '{instruction}'")
        try:
            result = compose_with_repair(
                self._lmps["composer"],
                instruction,
            )
        except (VocabValidationError, ObjectResolutionError):
            # Hard fail — surface vocab-exhausted / object-unresolvable cases
            # to the runner instead of silently disabling steering. Either case
            # means the composer + linter + held-block fallback could not
            # produce a valid stage list for this episode.
            raise
        except Exception as e:
            logger.error(f"Composer failed for '{instruction}': {e}")
            self._value_map = None
            return False

        self._stages = parse_composer_stages(
            result, default_mode=self._default_stage_mode
        )
        if not self._stages:
            self._value_map = None
            return False

        # If the policy needs primitives / objects, every stage must carry
        # them. Catch at setup time so the failure mode is "LLM forgot a
        # primitive" rather than a delayed crash on the first forward().
        if self._set_primitive_fn is not None:
            missing = [i for i, s in enumerate(self._stages) if s.primitive is None]
            if missing:
                raise ValueError(
                    f"Primitive-id mode is active but the composer omitted "
                    f"primitives on stages {missing}. Each stage must be a "
                    f"4-tuple (aff_fn, avoid_fn, mode, primitive) where "
                    f"primitive ∈ {sorted(PRIMITIVE_VOCAB)}. Update the "
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
                    f"object ∈ {sorted(OBJECT_VOCAB)}. Update the composer prompt."
                )

        logger.info(f"Composer returned {len(self._stages)} stage(s)")
        self._activate_stage(0)
        return True

    # ------------------------------------------------------------------
    # Stage activation
    # ------------------------------------------------------------------

    def _activate_stage(self, idx: int, is_refresh: bool = False) -> None:
        if idx >= len(self._stages):
            logger.warning(f"Stage {idx} out of range (have {len(self._stages)})")
            return

        spec = self._stages[idx]

        # Static stages: re-use frozen arrays after first activation.
        if spec.mode == "static" and spec.cached_affordance is not None:
            affordance = spec.cached_affordance
            avoidance = spec.cached_avoidance
            logger.info(f"stage {idx}: static (cached)")
        else:
            affordance = self._eval_map(spec.aff_fn)
            avoidance = self._eval_map(spec.avoid_fn)
            if spec.mode == "static" and affordance is not None:
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
            map_size=self._map_size,
            instruction=f"stage_{idx}",
        )
        self._value_map.smooth()
        self._value_map.precompute_gradients()

        self._gradient_field = (
            torch.from_numpy(
                np.stack(
                    [
                        self._value_map._grad_x,
                        self._value_map._grad_y,
                        self._value_map._grad_z,
                    ],
                    axis=-1,
                )
            )
            .float()
            .to(self._device)
        )

        # Stage target: cached for static, recomputed for track.
        # For {place, push, pull} primitives the target is the closest SURFACE
        # voxel of the affordance volume to the current EE — so a bbox-filled
        # affordance terminates at a reachable face, not the (interior) centroid.
        # {grasp, rotate} keep the centroid (we want to drive *into* the object).
        if spec.mode == "static" and spec.cached_target is not None:
            self._current_stage_target = spec.cached_target
        else:
            raw_aff = self._value_map._raw_affordance
            if raw_aff is not None and raw_aff.max() > 0:
                target_voxels = np.argwhere(raw_aff > 0)
                use_surface = spec.primitive in {"place", "push", "pull"}
                if use_surface and self._robot_obs is not None:
                    ee_voxel = pc2voxel(
                        self._robot_obs[:3][np.newaxis],
                        self._workspace_min,
                        self._workspace_max,
                        self._map_size,
                    )[0]
                    dists = np.linalg.norm(
                        target_voxels - ee_voxel, axis=-1
                    )
                    target_voxel = target_voxels[int(dists.argmin())]
                else:
                    target_voxel = target_voxels.mean(axis=0).astype(int)
                target = voxel2pc(
                    target_voxel[np.newaxis],
                    self._workspace_min,
                    self._workspace_max,
                    self._map_size,
                )[0]
            else:
                target = None
            if spec.mode == "static" and target is not None:
                spec.cached_target = target
            self._current_stage_target = target

        # Rotation target: same static/track caching as position target.
        if spec.rot_target is None:
            self._current_stage_target_rotation = None
        elif spec.mode == "static" and spec.cached_rotation is not None:
            self._current_stage_target_rotation = (
                torch.from_numpy(spec.cached_rotation).float().to(self._device)
            )
        else:
            rot6d = normalize_rot_target(spec.rot_target, idx=idx)
            if rot6d is None:
                self._current_stage_target_rotation = None
            else:
                if spec.mode == "static":
                    spec.cached_rotation = rot6d
                self._current_stage_target_rotation = (
                    torch.from_numpy(rot6d).float().to(self._device)
                )

        self._current_stage_idx = idx
        if not is_refresh:
            self._steps_in_stage = 0
            self._gripper_width_history.clear()
            self._grasp_block_log_counter = 0
            self._grasp_latched = False

        # The composer LLM writes a primitive name into each stage; we trust
        # the parser to have already validated it against PRIMITIVE_VOCAB.
        self._current_primitive_name = spec.primitive

        if self._set_primitive_fn is not None and spec.primitive is not None:
            pid = PRIMITIVE_VOCAB[spec.primitive]
            self._set_primitive_fn(pid)
            logger.info(
                f"Primitive-id set: stage {idx} -> '{spec.primitive}' (id={pid})"
            )
        if self._set_object_fn is not None and spec.object is not None:
            oid = OBJECT_VOCAB[spec.object]
            self._set_object_fn(oid)
            logger.info(f"Object-id set: stage {idx} -> '{spec.object}' (id={oid})")

        logger.info(
            f"Activated stage {idx}/{len(self._stages) - 1}: "
            f"affordance max={affordance.max():.2f}, "
            f"non-zero={np.count_nonzero(affordance)}, "
            f"target={self._current_stage_target}"
        )
        if self._current_stage_target_rotation is not None:
            R_target = (
                compute_rotation_matrix_from_ortho6d(
                    self._current_stage_target_rotation.unsqueeze(0)
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )
            logger.info(f"Stage {idx} rotation target (3x3):\n{R_target}")

        # (Task 5 iter 3 removed the direct ValueMapVisualizer call here.
        # HTML output is now produced by `StageHtmlRenderer` via the
        # `VisualizationManager` dispatch path — same `ValueMapVisualizer`
        # under the hood, but routed through the Renderer Protocol so
        # `Manager.register(...)` is the single extension point.)

    # ------------------------------------------------------------------
    # Per-env-step driver hooks
    # ------------------------------------------------------------------

    def increment_step(self) -> None:
        self._steps_in_stage += 1

    def check_transition(self, ee_pos: np.ndarray, gripper_width: float) -> bool:
        """Advance the stage when the EE has reached the current target."""
        # Track width every call regardless of stage — keeps the buffer warm
        # for the moment a grasp stage starts gating against it.
        self._gripper_width_history.append(float(gripper_width))

        if self._current_stage_target is None:
            return False
        if self._current_stage_idx >= len(self._stages) - 1:
            return self._maybe_loop_back()

        dist = np.linalg.norm(ee_pos - self._current_stage_target)
        if dist >= self._stage_proximity_threshold:
            return False

        # Proximity satisfied. Apply the grasp-completion gate (skipped for
        # articulated-target tasks where the gripper contacts a handle).
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

    def _maybe_loop_back(self) -> bool:
        """Restart from stage 0 after N policy-iters at the last stage.

        The success oracle terminates the episode on actual success; we only
        count time spent at the last stage so a stuck policy gets re-driven
        through the full stage sequence instead of locking onto a goal it has
        already "reached" but the env hasn't validated.
        """
        if not self._loop_back_enabled or len(self._stages) < 2:
            return False
        self._steps_in_last_stage += 1
        if (
            self._steps_in_last_stage >= self._loop_back_dwell_steps
            and self._loop_back_count < self._max_loop_backs
        ):
            self._loop_back_count += 1
            self._steps_in_last_stage = 0
            logger.info(
                f"Last-stage dwell ≥ {self._loop_back_dwell_steps} "
                f"policy-iters without success — looping back to stage 0 "
                f"(loop {self._loop_back_count}/{self._max_loop_backs})"
            )
            self._activate_stage(0)
            return True
        return False

    def _is_grasp_complete(self) -> bool:
        """True when the gripper has settled around an object.

        Requires a full window of width samples that all fall in the open
        interval (`grasp_min_width`, `grasp_max_width`) and whose spread is
        under `grasp_stability_eps`. Latches once true for the stage.
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

    def refresh(
        self,
        robot_obs: np.ndarray,
        scene_obs: np.ndarray,
        *,
        fixture_positions: Optional[dict] = None,
        block_aabbs: Optional[dict] = None,
    ) -> None:
        """Re-eval the active stage's costmap with current scene state."""
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

        if self._stages[self._current_stage_idx].mode == "static":
            return
        self._activate_stage(self._current_stage_idx, is_refresh=True)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def current(self) -> StageActivation:
        """Build a `StageActivation` snapshot for the guidance branches."""
        active_obj: Optional[str] = None
        if self._stages and 0 <= self._current_stage_idx < len(self._stages):
            active_obj = self._stages[self._current_stage_idx].object
        return StageActivation(
            value_map=self._value_map,
            gradient_field=self._gradient_field,
            stage_target_world=self._current_stage_target,
            rotation_target_6d=self._current_stage_target_rotation,
            primitive=self._current_primitive_name,
            object=active_obj,
            stage_idx=self._current_stage_idx,
            num_stages=len(self._stages),
            steps_in_stage=self._steps_in_stage,
        )

    def snapshot(self, ee_pos: np.ndarray) -> Optional[dict]:
        """Snapshot used by the live costmap viewer.

        Returns None when there's no active value map. The orchestrator
        merges in the episode-level `step` field, which lives outside the
        StageManager's responsibility.
        """
        if self._value_map is None:
            return None
        objects = (
            self._lmp_interface.get_all_detections() if self._lmp_interface else None
        )
        target_rot = (
            self._current_stage_target_rotation.detach().cpu().numpy()
            if self._current_stage_target_rotation is not None
            else None
        )
        # Current stage's target object (added in Task 5 iter 3 for the
        # OBJECT label in `LiveCostmapTkRenderer`). May be None on legacy
        # 2-tuple LLM emissions that omit the object slot.
        current_obj: Optional[str] = None
        if self._stages and 0 <= self._current_stage_idx < len(self._stages):
            current_obj = self._stages[self._current_stage_idx].object

        return {
            "value_map": self._value_map,
            "ee_pos": ee_pos,
            "target": self._current_stage_target,
            "target_rotation": target_rot,
            "objects": objects,
            "stage_idx": self._current_stage_idx,
            "num_stages": len(self._stages),
            "instruction": getattr(self._value_map, "instruction", "") or "",
            "primitive": self._current_primitive_name,
            "object": current_obj,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_map(map_fn: Any) -> Optional[np.ndarray]:
        """Evaluate a voxel map, handling callables and VoxelIndexingWrapper."""
        if map_fn is None:
            return None
        try:
            if callable(map_fn):
                result = map_fn()
            else:
                result = map_fn
            if hasattr(result, "array"):
                return result.array
            return np.asarray(result)
        except ObjectResolutionError:
            # Hard-fail per Phase 3b.0.5 spec: parse_query_obj / detect couldn't
            # resolve the composer's emitted object name. Propagate to the
            # runner instead of returning a silent-fallback workspace-center
            # affordance that misleads the policy.
            raise
        except Exception as e:
            logger.warning(f"Failed to evaluate map: {e}")
            return None
