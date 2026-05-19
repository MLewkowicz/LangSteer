"""Adaptive guidance scalers — strategy pattern for fade/decay multipliers.

A scaler returns a scalar in [floor, 1.0] that the guidance term is multiplied
by before being added to the ε prediction. The four currently used variants
ramp on different signals:

  - `TimestepScaler`       — diffusion-step ramp (early high-noise steps get
                              less guidance, later low-noise steps get full).
  - `DistanceScaler`       — EE-to-stage-target distance ramp (back off once
                              the EE is in the basin so the primitive policy
                              can drive through).
  - `StepScaler`           — env-steps-in-stage ramp (catches the misfire of
                              distance scaling on offset basins).
  - `RotationAlignmentScaler` — Frobenius distance ||R_pred − R_target||_F
                              ramp (back off rotation guidance once aligned).

All inputs they need are bundled into a `ScalerContext` built once per
`get_guidance` call. Adding a new scaler is a one-class change.

Public surface:
    ScalerContext, BaseScaler, TimestepScaler, DistanceScaler, StepScaler,
    RotationAlignmentScaler, compose(scalers, ctx).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch

from policies.diffuser_actor_components.rotation_utils import (
    compute_rotation_matrix_from_ortho6d,
)


@dataclass
class ScalerContext:
    """Per-guidance-call inputs any scaler might consult.

    Fields not relevant to a given scaler are left as None — each scaler
    checks the ones it cares about and returns 1.0 (no-op) otherwise.
    """

    timestep: int
    num_train_timesteps: Optional[int]
    ee_pos: Optional[torch.Tensor]  # (3,) absolute world
    stage_target: Optional[np.ndarray]  # (3,) absolute world
    steps_in_stage: int
    rot_pred_6d: Optional[torch.Tensor] = (
        None  # (B, H, 6) — populated by the rotation branch
    )
    rot_target_6d: Optional[torch.Tensor] = None  # (6,) on device


class BaseScaler:
    """Common contract: `enabled` flag + `compute(ctx) -> float`."""

    enabled: bool = True

    def compute(self, ctx: ScalerContext) -> float:  # pragma: no cover — abstract
        raise NotImplementedError


class TimestepScaler(BaseScaler):
    """Linear ramp from `min_scale` at t=0 to 1.0 at t=num_train_timesteps."""

    def __init__(self, *, enabled: bool, min_scale: float) -> None:
        self.enabled = enabled
        self.min_scale = min_scale

    def compute(self, ctx: ScalerContext) -> float:
        if not self.enabled:
            return 1.0
        if ctx.num_train_timesteps is None:
            return 1.0
        normalized_t = ctx.timestep / ctx.num_train_timesteps
        return self.min_scale + (1.0 - self.min_scale) * normalized_t


class DistanceScaler(BaseScaler):
    """Linear ramp on ||ee_pos − stage_target||.

    1.0 when far (d ≥ `full`), `floor` when at-target (d ≤ `near`), linear
    between. Returns 1.0 (no-op) when the stage target or EE position is
    missing, or when `full <= near` (malformed config).
    """

    def __init__(
        self, *, enabled: bool, full: float, near: float, floor: float
    ) -> None:
        self.enabled = enabled
        self.full = full
        self.near = near
        self.floor = floor

    def compute(self, ctx: ScalerContext) -> float:
        if not self.enabled:
            return 1.0
        if ctx.stage_target is None or ctx.ee_pos is None:
            return 1.0
        target = torch.tensor(
            ctx.stage_target,
            dtype=ctx.ee_pos.dtype,
            device=ctx.ee_pos.device,
        )
        d = torch.norm(ctx.ee_pos - target).item()
        if self.full <= self.near:
            return 1.0
        if d >= self.full:
            return 1.0
        if d <= self.near:
            return self.floor
        return self.floor + (1.0 - self.floor) * (d - self.near) / (
            self.full - self.near
        )


class StepScaler(BaseScaler):
    """Linear ramp on env-steps spent in the current stage."""

    def __init__(
        self, *, enabled: bool, full_steps: int, decay_steps: int, floor: float
    ) -> None:
        self.enabled = enabled
        self.full_steps = full_steps
        self.decay_steps = decay_steps
        self.floor = floor

    def compute(self, ctx: ScalerContext) -> float:
        if not self.enabled:
            return 1.0
        s = ctx.steps_in_stage
        if self.decay_steps <= 0:
            return 1.0
        if s <= self.full_steps:
            return 1.0
        if s >= self.full_steps + self.decay_steps:
            return self.floor
        return 1.0 - (1.0 - self.floor) * (s - self.full_steps) / self.decay_steps


class RotationAlignmentScaler(BaseScaler):
    """Linear ramp on the chordal distance ||R_pred − R_target||_F.

    Probe trajectory midpoint as a cheap representative; chordal distance
    at that index correlates well with whole-trajectory alignment. Returns
    1.0 when either the predicted or the target rotation is unavailable.
    """

    def __init__(
        self, *, enabled: bool, full: float, near: float, floor: float
    ) -> None:
        self.enabled = enabled
        self.full = full
        self.near = near
        self.floor = floor

    def compute(self, ctx: ScalerContext) -> float:
        if not self.enabled:
            return 1.0
        if ctx.rot_target_6d is None or ctx.rot_pred_6d is None:
            return 1.0

        rot_target_6d = ctx.rot_target_6d.unsqueeze(0)  # (1, 6)
        R_target = compute_rotation_matrix_from_ortho6d(rot_target_6d)  # (1, 3, 3)
        probe = ctx.rot_pred_6d[0, ctx.rot_pred_6d.shape[1] // 2].unsqueeze(0)  # (1, 6)
        R_pred = compute_rotation_matrix_from_ortho6d(probe)  # (1, 3, 3)

        d = torch.norm(R_pred - R_target).item()
        if self.full <= self.near:
            return 1.0
        if d >= self.full:
            return 1.0
        if d <= self.near:
            return self.floor
        return self.floor + (1.0 - self.floor) * (d - self.near) / (
            self.full - self.near
        )


def compose(scalers: Iterable[BaseScaler], ctx: ScalerContext) -> float:
    """Multiply enabled scalers' outputs. Returns 1.0 when nothing's enabled."""
    result = 1.0
    for s in scalers:
        result *= s.compute(ctx)
    return result
