"""VoxPoser value-map-based steering for diffusion policies — orchestrator.

This module is the public face of VoxPoser steering. It implements the
`BaseSteering` interface and wires the focused submodules together:

  * `StageManager`           — stage list, transitions, grasp gate, refresh,
                               LMP composer lifecycle, primitive/object callbacks.
  * `PositionTransform`      — model ↔ world coords + voxel-gradient lookup.
  * `PositionFieldGuidance`  — trajectory-position branch (value-map gradient → ε).
  * `RotationFieldGuidance`  — SO(3) branch (per-horizon SLERP target → ε).
  * `scalers.*`              — adaptive multipliers shared between branches.
  * `diffusion_utils.*`      — Tweedie / ᾱ / ε-vs-DPS coefficients.

Two guidance modes are supported (toggled via `guidance_mode` config):
  - 'epsilon': convert world-space gradient → ε space via Jacobian and add
    to ε before the scheduler step. No denoising-loop changes required.
  - 'dps':     return gradient correction in model trajectory space; the
    denoising loop applies it to x_{t-1} after the scheduler step.

Steering is skipped entirely for timesteps above `start_guidance_timestep`
to prevent guidance from acting on pure-noise early steps where Tweedie x₀
predictions are meaningless.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import torch

from core.steering import BaseSteering
from steering.coordinates import PositionTransform
from steering.diffusion_utils import get_alpha_bar
from steering.position_field import PositionFieldGuidance
from steering.rotation_field import RotationFieldGuidance
from steering.scalers import (
    DistanceScaler,
    RotationAlignmentScaler,
    ScalerContext,
    StepScaler,
    TimestepScaler,
)
from steering.stage_manager import StageManager

logger = logging.getLogger(__name__)


class VoxPoserSteering(BaseSteering):
    """Steer diffusion policies using LLM-generated spatial value maps.

    At episode start, the LLM composer decomposes the task instruction into
    affordance/avoidance maps over a 3D voxel grid (one per stage). During
    denoising, Tweedie's formula predicts x₀ from the current noisy
    trajectory, positions are denormalized to world space, the precomputed
    gradient field is queried, and the result is converted to either
    epsilon-space or DPS-style guidance. An optional per-stage rotation
    target adds a parallel branch in 6D rotation space.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.guidance_strength: float = cfg.get("guidance_strength", 1.0)
        self.horizon: int = cfg.get("horizon", 16)
        # Must be 'epsilon' for DiffuserActor (which uses epsilon prediction).
        self.prediction_type: str = cfg.get("prediction_type", "epsilon")
        self.device: str = cfg.get("device", "cuda")
        self.map_size: int = cfg.get("map_size", 100)
        self.guidance_mode: str = cfg.get("guidance_mode", "epsilon")
        self.start_guidance_timestep: int = cfg.get("start_guidance_timestep", 50)

        workspace_min = np.array(
            cfg.get("workspace_bounds_min", [-0.35, -0.40, 0.40]),
            dtype=np.float32,
        )
        workspace_max = np.array(
            cfg.get("workspace_bounds_max", [0.35, 0.15, 0.85]),
            dtype=np.float32,
        )

        # Model↔world coordinate transform (held by the position branch +
        # used by `set_current_gripper_pos` to update the gripper pose).
        self._coords = PositionTransform(
            gripper_loc_bounds=cfg.get("gripper_loc_bounds", None),
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            is_relative=cfg.get("relative", True),
            device=self.device,
        )

        # Schedulers — set later by the policy wrapper before inference.
        self.position_scheduler: Any = None
        self.rotation_scheduler: Any = None

        # LMP config dict — held by reference and shared with StageManager.
        self._lmp_config: Any = cfg

        # Stage management: composer, transitions, refresh, callbacks.
        # (Task 5 iter 3 dropped the `visualize` kwarg — HTML output is now
        # owned by `visualization.VisualizationManager` via the Renderer
        # Protocol, not the steering module.)
        self._stage_manager = StageManager(
            cfg,
            device=self.device,
            map_size=self.map_size,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
        )

        # Adaptive scalers — shared across branches where applicable.
        timestep_scaler = TimestepScaler(
            enabled=cfg.get("use_timestep_scaling", True),
            min_scale=cfg.get("min_timestep_scale", 0.1),
        )
        distance_scaler = DistanceScaler(
            enabled=cfg.get("use_distance_scaling", False),
            full=cfg.get("distance_full", 0.12),
            near=cfg.get("distance_near", 0.04),
            floor=cfg.get("distance_floor", 0.05),
        )
        step_scaler = StepScaler(
            enabled=cfg.get("use_step_scaling", False),
            full_steps=cfg.get("step_full", 0),
            decay_steps=cfg.get("step_decay", 80),
            floor=cfg.get("step_floor", 0.05),
        )
        rot_alignment_scaler = RotationAlignmentScaler(
            enabled=cfg.get("use_rot_alignment_scaling", False),
            full=cfg.get("rot_align_full", 0.5),
            near=cfg.get("rot_align_near", 0.1),
            floor=cfg.get("rot_align_floor", 0.05),
        )

        # Position branch.
        self._position_field = PositionFieldGuidance(
            horizon=self.horizon,
            guidance_strength=self.guidance_strength,
            prediction_type=self.prediction_type,
            guidance_mode=self.guidance_mode,
            start_guidance_timestep=self.start_guidance_timestep,
            coordinates=self._coords,
            map_size=self.map_size,
            timestep_scaler=timestep_scaler,
            distance_scaler=distance_scaler,
            step_scaler=step_scaler,
        )

        # Rotation branch — independent strength + gate threshold.
        self._rotation_field = RotationFieldGuidance(
            horizon=self.horizon,
            guidance_strength=cfg.get("guidance_strength_rot", self.guidance_strength),
            guidance_mode=self.guidance_mode,
            start_guidance_timestep=cfg.get(
                "start_guidance_timestep_rot", self.start_guidance_timestep
            ),
            rot_horizon_floor=cfg.get("rot_horizon_floor", 0.0),
            rot_horizon_alpha_max=cfg.get("rot_horizon_alpha_max", 0.3),
            timestep_scaler=timestep_scaler,
            alignment_scaler=rot_alignment_scaler,
        )

        # Episode-level step counter (drives log throttling).
        self._episode_step: int = 0
        # Forward-compat plumbing for a future align-decay on real EE
        # rotation (v1 uses the Tweedie x₀ prediction instead).
        self._current_gripper_rotation: Optional[torch.Tensor] = None

        logger.info(
            f"VoxPoserSteering: mode={self.guidance_mode}, "
            f"strength={self.guidance_strength}, "
            f"start_t={self.start_guidance_timestep}, "
            f"map_size={self.map_size}, prediction_type={self.prediction_type}"
        )

    # ------------------------------------------------------------------
    # Scheduler setters
    # ------------------------------------------------------------------

    def set_position_scheduler(self, scheduler: Any) -> None:
        self.position_scheduler = scheduler

    def set_rotation_scheduler(self, scheduler: Any) -> None:
        self.rotation_scheduler = scheduler

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def set_primitive_callback(self, fn: Callable[[int], None]) -> None:
        """Register a callback invoked at each stage transition with the primitive id.

        Wire with `voxposer.set_primitive_callback(policy.set_primitive)` when
        using the primitive-id-conditioned Diffuser Actor. Leaving this unset
        is a no-op (e.g. for CLIP or no-language policies).
        """
        self._stage_manager.set_primitive_callback(fn)

    def set_object_callback(self, fn: Callable[[int], None]) -> None:
        """Register a callback invoked at each stage transition with the object id.

        Wire with `voxposer.set_object_callback(policy.set_object)` when using
        the primitive+object-conditioned Diffuser Actor. Requires the LLM
        composer to emit an `object` slot in every stage tuple.
        """
        self._stage_manager.set_object_callback(fn)

    # ------------------------------------------------------------------
    # Per-step state setters (called by the policy wrapper)
    # ------------------------------------------------------------------

    def set_current_gripper_pos(self, gripper_pos: np.ndarray) -> None:
        """Set the current absolute gripper position used to denormalize x₀."""
        self._coords.set_gripper_pos(gripper_pos)

    def set_current_gripper_rotation(self, gripper_rot: Any) -> None:
        """Set current EE rotation. Stored for forward compatibility.

        v1 rotation guidance computes alignment-decay from the Tweedie x₀
        prediction, so this state isn't read inside `get_guidance`. Kept
        plumbed so a v2 decay variant on real EE rotation requires no wiring
        changes upstream.

        Args:
            gripper_rot: (3,) Euler XYZ, (4,) wxyz quaternion, or (3,3) matrix.
        """
        rot = np.asarray(gripper_rot, dtype=np.float32)
        if rot.shape == (3,):
            # Euler XYZ → 3x3, intrinsic XYZ (Rx @ Ry @ Rz) — must match
            # Diffuser Actor's convert_rotation.
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
            mat = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float32,
            )
        elif rot.shape == (3, 3):
            mat = rot.astype(np.float32)
        else:
            logger.warning(
                f"set_current_gripper_rotation: unexpected shape {rot.shape}"
            )
            return
        self._current_gripper_rotation = torch.from_numpy(mat).to(self.device)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def setup_episode(
        self,
        task_name: str,
        instruction: Optional[str] = None,
        robot_obs: Optional[np.ndarray] = None,
        scene_obs: Optional[np.ndarray] = None,
        fixture_positions: Optional[dict] = None,
        block_aabbs: Optional[dict] = None,
    ) -> tuple[None, None]:
        """Generate value maps for a new episode via the LLM composer.

        Returns (None, None) for compatibility with TweedieSteering's
        (robot_obs, scene_obs) two-tuple return.
        """
        self._episode_step = 0
        self._stage_manager.setup_episode(
            task_name,
            instruction=instruction,
            robot_obs=robot_obs,
            scene_obs=scene_obs,
            fixture_positions=fixture_positions,
            block_aabbs=block_aabbs,
        )
        return None, None

    def check_stage_transition(self, ee_pos: np.ndarray, gripper_width: float) -> bool:
        return self._stage_manager.check_transition(ee_pos, gripper_width)

    def refresh_costmap(
        self,
        robot_obs: np.ndarray,
        scene_obs: np.ndarray,
        fixture_positions: Optional[dict] = None,
        block_aabbs: Optional[dict] = None,
    ) -> None:
        self._stage_manager.refresh(
            robot_obs,
            scene_obs,
            fixture_positions=fixture_positions,
            block_aabbs=block_aabbs,
        )

    def increment_step(self) -> None:
        """Advance episode + stage step counters."""
        self._episode_step += 1
        self._stage_manager.increment_step()

    def get_costmap_state(self, ee_pos: np.ndarray) -> Optional[dict]:
        """Live-viewer snapshot. Returns None before composer succeeds."""
        snap = self._stage_manager.snapshot(ee_pos)
        if snap is None:
            return None
        snap["step"] = self._episode_step
        return snap

    @property
    def _value_map(self) -> Any:
        """Backwards-compat shim — external code (run_evaluation, run_experiment)
        reads `steering._value_map` to verify the composer succeeded."""
        return self._stage_manager.current().value_map

    @property
    def current_episode_step(self) -> int:
        return self._episode_step

    # ------------------------------------------------------------------
    # Core guidance computation
    # ------------------------------------------------------------------

    def get_guidance(
        self,
        current_sample: torch.Tensor,
        timestep: int,
        obs_embedding: Any,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value-map gradient guidance.

        Two independent branches both write into the returned guidance tensor:
          - Position branch (dims [0:3]) — active when the stage has an
            affordance map.
          - Rotation branch (dims [3:9]) — active only when the stage
            carries a rot_target.

        The openness slot ([9:10] for DiffuserActor) is never written.
        """
        guidance = torch.zeros_like(model_output)

        stage = self._stage_manager.current()
        t = (
            int(timestep.item())
            if isinstance(timestep, torch.Tensor)
            else int(timestep)
        )
        num_train_t = (
            self.position_scheduler.config.num_train_timesteps
            if self.position_scheduler is not None
            else None
        )

        ctx = ScalerContext(
            timestep=t,
            num_train_timesteps=num_train_t,
            ee_pos=self._coords.current_gripper_pos,
            stage_target=stage.stage_target_world,
            steps_in_stage=stage.steps_in_stage,
            rot_target_6d=stage.rotation_target_6d,
        )

        alpha_bar_pos = get_alpha_bar(
            self.position_scheduler,
            timestep,
            device=self.device,
        )
        alpha_bar_rot = get_alpha_bar(
            self.rotation_scheduler,
            timestep,
            device=self.device,
        )

        pos_delta = self._position_field.compute(
            x_t=current_sample,
            eps=model_output,
            timestep=timestep,
            alpha_bar=alpha_bar_pos,
            stage=stage,
            ctx=ctx,
            episode_step=self._episode_step,
        )
        if pos_delta is not None:
            H = pos_delta.shape[1]
            guidance[:, :H, :3] = pos_delta

        rot_delta = self._rotation_field.compute(
            x_t_rot=current_sample[..., 3:9],
            eps_rot=model_output[..., 3:9],
            timestep=timestep,
            alpha_bar_rot=alpha_bar_rot,
            stage=stage,
            ctx=ctx,
            episode_step=self._episode_step,
        )
        if rot_delta is not None:
            H = rot_delta.shape[1]
            guidance[:, :H, 3:9] = rot_delta

        return guidance.detach()
