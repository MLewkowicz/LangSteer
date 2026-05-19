"""SO(3) guidance branch — per-horizon SLERP target → ε correction.

Companion to `position_field.py`. The rotation branch's only target source
today is a single 6D rotation pinned per stage (`StageActivation.rotation_target_6d`),
but the `compute(stage, ctx)` interface is generic: a future stage type that
emits a per-position rotation value map can plug in without changing this
module's contract.

Pipeline:
  1. recover x₀_rot from x_t_rot via Tweedie (with the *rotation* scheduler's
     ᾱ — squaredcos diverges from position's scaled_linear),
  2. build per-horizon SLERP targets on the SO(3) geodesic so the predicted
     trajectory rotates along the great circle instead of cutting through R⁶,
  3. compute Δ6D = (x₀_rot − target_per_h) and fold it into ε via the
     timestep + alignment scalers and the ε-or-DPS coefficient.

Returns None when the stage carries no rotation target or the current
timestep is above the gating threshold.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Optional

import torch

from policies.diffuser_actor_components.rotation_utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from steering.diffusion_utils import dps_coeff, epsilon_coeff, predict_x0
from steering.scalers import (
    RotationAlignmentScaler,
    ScalerContext,
    TimestepScaler,
)

if TYPE_CHECKING:
    from steering.stage_manager import StageActivation

logger = logging.getLogger(__name__)


class RotationFieldGuidance:
    """Compute ε-space guidance toward a stage's rotation target."""

    def __init__(
        self,
        *,
        horizon: int,
        guidance_strength: float,
        guidance_mode: str,
        start_guidance_timestep: int,
        rot_horizon_floor: float,
        rot_horizon_alpha_max: float,
        timestep_scaler: TimestepScaler,
        alignment_scaler: RotationAlignmentScaler,
    ) -> None:
        self.horizon = horizon
        self.guidance_strength = guidance_strength
        self.guidance_mode = guidance_mode
        self.start_guidance_timestep = start_guidance_timestep
        self.rot_horizon_floor = rot_horizon_floor
        self.rot_horizon_alpha_max = rot_horizon_alpha_max
        self._timestep_scaler = timestep_scaler
        self._alignment_scaler = alignment_scaler

    def compute(
        self,
        *,
        x_t_rot: torch.Tensor,
        eps_rot: torch.Tensor,
        timestep: int,
        alpha_bar_rot: torch.Tensor,
        stage: "StageActivation",
        ctx: ScalerContext,
        episode_step: int,
    ) -> Optional[torch.Tensor]:
        """Return (B, H, 6) ε-space delta, or None when gated off."""
        t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        if t > self.start_guidance_timestep:
            return None
        if stage.rotation_target_6d is None:
            return None

        B, L, _ = x_t_rot.shape
        H = min(self.horizon, L)
        x_t_h = x_t_rot[:, :H, :]
        eps_h = eps_rot[:, :H, :]

        # Position scheduler's ᾱ would yield a wrong x₀_rot — the squaredcos
        # rotation schedule diverges from position's scaled_linear.
        x0_rot_pred = predict_x0(
            x_t_h,
            eps_h,
            alpha_bar_rot,
            prediction_type="epsilon",
        )

        target_6d_single = stage.rotation_target_6d.to(x0_rot_pred.device)  # (6,)
        h_idx = torch.arange(H, device=x0_rot_pred.device, dtype=x0_rot_pred.dtype)
        # Quadratic ramp from `floor` at h=0 to `alpha_max` at h=H-1. alpha_max < 1
        # nudges only part-way along the SO(3) geodesic per call so convergence
        # accumulates across env steps at training-distribution speed.
        alpha_span = self.rot_horizon_alpha_max - self.rot_horizon_floor
        alphas = self.rot_horizon_floor + alpha_span * (h_idx / max(H - 1, 1)) ** 2
        target_6d_per_h = self.slerp_targets_per_horizon(
            x0_rot_pred,
            target_6d_single,
            alphas,
        )

        # Sign convention (matches position branch): cost gradient points AWAY
        # from the per-horizon target; Tweedie's −dx₀/dε flips it back.
        grad_6d = x0_rot_pred - target_6d_per_h  # (B, H, 6)

        # Alignment scaler needs x₀_rot — build a per-call ctx that adds it
        # without mutating the orchestrator's shared base context.
        ctx_rot = dataclasses.replace(ctx, rot_pred_6d=x0_rot_pred)
        ts_scale = self._timestep_scaler.compute(ctx_rot)
        align_scale = self._alignment_scaler.compute(ctx_rot)

        coeff_rot = (
            epsilon_coeff(alpha_bar_rot)
            if self.guidance_mode == "epsilon"
            else dps_coeff(alpha_bar_rot)
        )
        delta_rot = (
            self.guidance_strength * ts_scale * align_scale * coeff_rot * grad_6d
        )

        if episode_step % 10 == 0:
            R_target = compute_rotation_matrix_from_ortho6d(
                target_6d_single.unsqueeze(0)
            )
            R_pred = compute_rotation_matrix_from_ortho6d(
                x0_rot_pred[0, x0_rot_pred.shape[1] // 2].unsqueeze(0)
            )
            chord_d = torch.norm(R_pred - R_target).item()
            logger.info(
                f"[VoxPoser/{self.guidance_mode}/rot] step={episode_step}, "
                f"t={t}: norm={torch.norm(delta_rot).item():.4f}, "
                f"coeff_rot={coeff_rot.item():.4f}, "
                f"align_scale={align_scale:.4f}, "
                f"alpha_bar_rot={alpha_bar_rot.item():.4f}, "
                f"chord_d(R_pred,R_target)={chord_d:.4f}"
            )
        return delta_rot

    @staticmethod
    def slerp_targets_per_horizon(
        x0_rot_pred: torch.Tensor,
        target_6d_single: torch.Tensor,
        alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Per-horizon SLERP targets along the SO(3) geodesic.

        For each horizon step h, computes target_h = SLERP(R_pred_h, R_target,
        alphas[h]). The interpolation travels along the great-circle geodesic
        so the gradient (target − pred) points along the manifold rather than
        cutting through R⁶.

        Quaternion SLERP is well-defined even at the 180° antipode (where
        chord_d hits 2√2): orthogonal quaternions, sin(π/2)=1, no
        singularity. Hemisphere fix ensures we always go the short way.

        Args:
            x0_rot_pred: (B, H, 6) predicted rotations in 6D.
            target_6d_single: (6,) stage's final target rotation.
            alphas: (H,) per-horizon SLERP fractions in [0, 1]. 0 = stay at
                R_pred, 1 = reach R_target.

        Returns:
            (B, H, 6) per-horizon SLERP targets in 6D.
        """
        B, H, _ = x0_rot_pred.shape

        R_pred = compute_rotation_matrix_from_ortho6d(
            x0_rot_pred.reshape(-1, 6)
        )  # (B*H, 3, 3)
        q_pred = matrix_to_quaternion(R_pred)  # (B*H, 4)

        R_target = compute_rotation_matrix_from_ortho6d(
            target_6d_single.unsqueeze(0)
        )  # (1, 3, 3)
        q_target = matrix_to_quaternion(R_target).squeeze(0)  # (4,)
        q_target = q_target.unsqueeze(0).expand(B * H, 4).contiguous()

        # Hemisphere fix: q and −q are the same rotation. Pick the sign that
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
        sin_theta_safe = torch.where(use_slerp, sin_theta, torch.ones_like(sin_theta))
        s1 = torch.sin((1.0 - alphas_flat) * theta) / sin_theta_safe
        s2 = torch.sin(alphas_flat * theta) / sin_theta_safe
        q_slerp = s1 * q_pred + s2 * q_target
        q_lerp = (1.0 - alphas_flat) * q_pred + alphas_flat * q_target
        q_interp = torch.where(use_slerp, q_slerp, q_lerp)
        q_interp = q_interp / q_interp.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        R_interp = quaternion_to_matrix(q_interp)  # (B*H, 3, 3)
        target_6d_per_h = get_ortho6d_from_rotation_matrix(R_interp)  # (B*H, 6)
        return target_6d_per_h.reshape(B, H, 6)
