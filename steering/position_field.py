"""Trajectory-position guidance branch — value-map gradient → ε correction.

Self-contained position branch of `VoxPoserSteering.get_guidance`. Given a
stage with a precomputed value-map gradient field, this:

  1. recovers x₀ from x_t via Tweedie's formula (`predict_x0`),
  2. denormalizes x₀'s position slice to absolute world coordinates,
  3. samples the gradient field at the predicted world positions,
  4. chains the result back into model space via the position transform,
  5. multiplies by the timestep/distance/step scalers and the ε-or-DPS
     coefficient, returning a (B, H, 3) delta.

The branch is a no-op (returns None) when the stage has no value map or the
current timestep is above the gating threshold.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from steering.coordinates import PositionTransform
from steering.diffusion_utils import dps_coeff, epsilon_coeff, predict_x0
from steering.scalers import (
    DistanceScaler,
    ScalerContext,
    StepScaler,
    TimestepScaler,
)

if TYPE_CHECKING:
    from steering.stage_manager import StageActivation

logger = logging.getLogger(__name__)


class PositionFieldGuidance:
    """Compute ε-space guidance from a stage's value-map gradient field."""

    def __init__(
        self,
        *,
        horizon: int,
        guidance_strength: float,
        prediction_type: str,
        guidance_mode: str,
        start_guidance_timestep: int,
        coordinates: PositionTransform,
        map_size: int,
        timestep_scaler: TimestepScaler,
        distance_scaler: DistanceScaler,
        step_scaler: StepScaler,
    ) -> None:
        self.horizon = horizon
        self.guidance_strength = guidance_strength
        self.prediction_type = prediction_type
        self.guidance_mode = guidance_mode
        self.start_guidance_timestep = start_guidance_timestep
        self._coordinates = coordinates
        self._map_size = map_size
        self._timestep_scaler = timestep_scaler
        self._distance_scaler = distance_scaler
        self._step_scaler = step_scaler

    def compute(
        self,
        *,
        x_t: torch.Tensor,
        eps: torch.Tensor,
        timestep: int,
        alpha_bar: torch.Tensor,
        stage: "StageActivation",
        ctx: ScalerContext,
        episode_step: int,
    ) -> Optional[torch.Tensor]:
        """Return (B, H, 3) ε-space delta, or None when the branch is gated off."""
        t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        if t > self.start_guidance_timestep:
            return None
        if stage.value_map is None or stage.gradient_field is None:
            return None

        B, L, _ = x_t.shape
        H = min(self.horizon, L)

        # model_output may carry openness in a trailing slot — match the slice
        # to x_t's last dim before Tweedie.
        D = x_t.shape[-1]
        x0 = predict_x0(
            x_t,
            eps[..., :D],
            alpha_bar,
            prediction_type=self.prediction_type,
        )
        model_pos = x0[:, :H, :3]
        world_pos = self._coordinates.model_to_world(model_pos)
        grad_world = self._coordinates.lookup_voxel_gradient(
            world_pos,
            stage.gradient_field,
            self._map_size,
        )
        grad_model = self._coordinates.world_gradient_to_model(grad_world)

        scale = self._timestep_scaler.compute(ctx)
        distance_scale = self._distance_scaler.compute(ctx)
        step_scale = self._step_scaler.compute(ctx)
        adaptive_scale = distance_scale * step_scale

        coeff = (
            epsilon_coeff(alpha_bar)
            if self.guidance_mode == "epsilon"
            else dps_coeff(alpha_bar)
        )
        delta = self.guidance_strength * scale * adaptive_scale * coeff * grad_model

        if episode_step % 10 == 0:
            logger.debug(
                f"[VoxPoser/{self.guidance_mode}/pos] step={episode_step}, "
                f"t={t}: norm={torch.norm(delta).item():.4f}, "
                f"coeff={coeff.item():.4f}, scale={scale:.4f}, "
                f"dist_scale={distance_scale:.4f}, step_scale={step_scale:.4f}, "
                f"alpha_bar={alpha_bar.item():.4f}"
            )
        return delta
