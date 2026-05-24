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


# ---------------------------------------------------------------------------
# SO(3) ↔ so(3) helpers (Rodrigues' rotation formula and its inverse).
# Used by the "tangent" guidance method, which bypasses the quaternion
# hemisphere ambiguity by working in axis-angle (Lie algebra) space.
# ---------------------------------------------------------------------------

def _log_map_so3(R: torch.Tensor) -> torch.Tensor:
    """Matrix log: SO(3) → so(3) as axis-angle (3-vector).

    For each rotation matrix R, returns ω where ||ω|| ∈ [0, π] is the
    rotation angle and ω/||ω|| is the axis. The result is unique on (-π, π],
    which is the key property the "tangent" method needs to disambiguate
    the rotation direction without picking a quaternion hemisphere.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) axis-angle vectors.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # (...,)
    sin_theta = torch.sin(theta)

    # Skew-symmetric part recovers (sin θ) * ω̂; divide by 2 sin θ to get ω̂,
    # then multiply by θ for ω.
    omega_unscaled = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )  # (..., 3)

    # Small-angle: sin θ ≈ θ → scale = 1/2 (avoids 0/0).
    small = sin_theta.abs() < 1e-6
    scale = torch.where(
        small,
        torch.full_like(theta, 0.5),
        theta / (2.0 * sin_theta + 1e-12),
    )
    return omega_unscaled * scale.unsqueeze(-1)


def _exp_map_so3(omega: torch.Tensor) -> torch.Tensor:
    """Matrix exp: so(3) axis-angle (3-vector) → SO(3).

    Rodrigues' formula: R = I + (sin θ / θ) K + ((1 − cos θ) / θ²) K²
    with K = skew(ω), θ = ||ω||.

    Args:
        omega: (..., 3) axis-angle vectors (unnormalized).

    Returns:
        (..., 3, 3) rotation matrices.
    """
    theta = torch.norm(omega, dim=-1)  # (...,)
    theta_safe = theta.clamp(min=1e-12)
    omega_hat = omega / theta_safe.unsqueeze(-1)
    ox, oy, oz = omega_hat[..., 0], omega_hat[..., 1], omega_hat[..., 2]
    zeros = torch.zeros_like(ox)

    K = torch.stack(
        [
            torch.stack([zeros, -oz, oy], dim=-1),
            torch.stack([oz, zeros, -ox], dim=-1),
            torch.stack([-oy, ox, zeros], dim=-1),
        ],
        dim=-2,
    )  # (..., 3, 3)

    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype)
    eye = eye.expand(K.shape)
    K_sq = torch.matmul(K, K)
    R = (
        eye
        + sin_t.unsqueeze(-1).unsqueeze(-1) * K
        + (1.0 - cos_t).unsqueeze(-1).unsqueeze(-1) * K_sq
    )
    return R


# Valid values for `rot_guidance_method`.
_VALID_GUIDANCE_METHODS = {"slerp", "tangent", "inject"}


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
        hemisphere_fix: bool = True,
        guidance_method: str = "slerp",
        inject_below_timestep: int = 20,
    ) -> None:
        if guidance_method not in _VALID_GUIDANCE_METHODS:
            raise ValueError(
                f"rot_guidance_method must be one of {_VALID_GUIDANCE_METHODS}, "
                f"got {guidance_method!r}"
            )
        self.horizon = horizon
        self.guidance_strength = guidance_strength
        self.guidance_mode = guidance_mode
        self.start_guidance_timestep = start_guidance_timestep
        self.rot_horizon_floor = rot_horizon_floor
        self.rot_horizon_alpha_max = rot_horizon_alpha_max
        self._timestep_scaler = timestep_scaler
        self._alignment_scaler = alignment_scaler
        self.hemisphere_fix = hemisphere_fix
        self.guidance_method = guidance_method
        # Used only by the "inject" method: at t below this threshold, the
        # gradient is replaced with an analytical delta that drives the
        # Tweedie x₀ estimate all the way to the target instead of nudging it.
        self.inject_below_timestep = inject_below_timestep

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

        # ---- Method dispatch -------------------------------------------------
        # "slerp"   : current behavior — quaternion SLERP with hemisphere_fix.
        # "tangent" : log/exp map on SO(3). Removes the quaternion hemisphere
        #             ambiguity (sign of rotation direction is fixed by R_target
        #             alone, not by q_pred's noisy state).
        # "inject"  : at t below `inject_below_timestep`, return an analytical
        #             delta that drives x0_rot_pred all the way to target_6d
        #             instead of merely nudging it. Above that threshold, fall
        #             back to the tangent path.
        # ----------------------------------------------------------------------
        if self.guidance_method == "slerp":
            target_6d_per_h = self.slerp_targets_per_horizon(
                x0_rot_pred,
                target_6d_single,
                alphas,
                hemisphere_fix=self.hemisphere_fix,
            )
        else:
            target_6d_per_h = self.tangent_targets_per_horizon(
                x0_rot_pred,
                target_6d_single,
                alphas,
            )

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

        if self.guidance_method == "inject" and t <= self.inject_below_timestep:
            # Analytical injection: solve for the ε-delta that makes
            # x₀_new(ε + delta) = target_6d_per_h, given Tweedie's formula
            #     x₀ = (x_t − √(1−ᾱ) ε) / √ᾱ.
            # Rearranging:
            #     ε + delta = (x_t − target · √ᾱ) / √(1−ᾱ)
            #     delta     = (x_t − target · √ᾱ) / √(1−ᾱ) − ε
            sqrt_alpha = torch.sqrt(alpha_bar_rot)
            sqrt_one_minus_alpha = torch.sqrt(
                (1.0 - alpha_bar_rot).clamp(min=1e-8)
            )
            forced_eps = (x_t_h - target_6d_per_h * sqrt_alpha) / sqrt_one_minus_alpha
            delta_rot = forced_eps - eps_h
            # Re-scale only by guidance_strength so the user can still dial down
            # the injection strength. The coeff/scalers are skipped because the
            # analytical delta is already the closed-form solution.
            delta_rot = self.guidance_strength * delta_rot
        else:
            # Sign convention (matches position branch): cost gradient points
            # AWAY from the per-horizon target; Tweedie's −dx₀/dε flips it back.
            grad_6d = x0_rot_pred - target_6d_per_h  # (B, H, 6)
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
                f"[VoxPoser/{self.guidance_mode}/rot/{self.guidance_method}] "
                f"step={episode_step}, t={t}: norm={torch.norm(delta_rot).item():.4f}, "
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
        hemisphere_fix: bool = True,
    ) -> torch.Tensor:
        """Per-horizon SLERP targets along the SO(3) geodesic.

        For each horizon step h, computes target_h = SLERP(R_pred_h, R_target,
        alphas[h]). The interpolation travels along the great-circle geodesic
        so the gradient (target − pred) points along the manifold rather than
        cutting through R⁶.

        Quaternion SLERP is well-defined even at the 180° antipode (where
        chord_d hits 2√2): orthogonal quaternions, sin(π/2)=1, no
        singularity. When `hemisphere_fix` is True the SLERP takes the short
        way in quaternion space; when False the q_target sign is left as-is,
        which lets the steering pull the prediction across the antipodal
        boundary instead of locking it to its current hemisphere.

        Args:
            x0_rot_pred: (B, H, 6) predicted rotations in 6D.
            target_6d_single: (6,) stage's final target rotation.
            alphas: (H,) per-horizon SLERP fractions in [0, 1]. 0 = stay at
                R_pred, 1 = reach R_target.
            hemisphere_fix: when True (default), flip q_target sign to share
                a hemisphere with q_pred (short-way SLERP).

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

        # Hemisphere fix: q and −q are the same rotation. When enabled, pick
        # the sign that gives a positive dot with q_pred so SLERP takes the
        # short way. When disabled, leave the sign alone — useful to test
        # whether the policy's directional bias is being locked in by the
        # short-way constraint.
        dot = (q_pred * q_target).sum(dim=-1, keepdim=True)
        if hemisphere_fix:
            q_target = torch.where(dot < 0, -q_target, q_target)
            dot = dot.abs().clamp(min=-1.0, max=1.0)
        else:
            dot = dot.clamp(min=-1.0, max=1.0)

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

    @staticmethod
    def tangent_targets_per_horizon(
        x0_rot_pred: torch.Tensor,
        target_6d_single: torch.Tensor,
        alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Per-horizon targets via the SO(3) log/exp map (Option 1 / tangent).

        Mathematically equivalent to SLERP on SO(3) but uses the unique
        principal-value log map in place of the quaternion hemisphere choice,
        so the per-horizon target's rotation direction is fixed by R_target
        alone — not by the noisy q_pred. This is what the SLERP path can't
        give us: the gradient direction follows the composer's intent rather
        than the policy's current bias.

        For each horizon step h: R_target_h = exp(αₕ · log(R_target · R_pred_hᵀ)) · R_pred_h.

        Args:
            x0_rot_pred: (B, H, 6) predicted rotations in 6D.
            target_6d_single: (6,) stage's final target rotation.
            alphas: (H,) per-horizon interpolation fractions in [0, 1].

        Returns:
            (B, H, 6) per-horizon targets in 6D.
        """
        B, H, _ = x0_rot_pred.shape

        R_pred = compute_rotation_matrix_from_ortho6d(
            x0_rot_pred.reshape(-1, 6)
        )  # (B*H, 3, 3)
        R_target = compute_rotation_matrix_from_ortho6d(
            target_6d_single.unsqueeze(0)
        )  # (1, 3, 3)
        R_target = R_target.expand(B * H, 3, 3)

        # Relative rotation that takes R_pred to R_target (in world frame).
        # log_map gives the unique axis-angle representation in (-π, π].
        R_rel = torch.matmul(R_target, R_pred.transpose(-2, -1))  # (B*H, 3, 3)
        omega = _log_map_so3(R_rel)  # (B*H, 3) axis-angle from R_pred to R_target

        # Scale the rotation by the per-horizon α to walk a fraction of the way.
        alphas_flat = alphas.view(1, H, 1).expand(B, H, 1).reshape(-1, 1)  # (B*H, 1)
        omega_scaled = omega * alphas_flat

        # Apply the scaled relative rotation to R_pred to get the per-horizon target.
        R_step = _exp_map_so3(omega_scaled)  # (B*H, 3, 3)
        R_target_h = torch.matmul(R_step, R_pred)
        target_6d_per_h = get_ortho6d_from_rotation_matrix(R_target_h)  # (B*H, 6)
        return target_6d_per_h.reshape(B, H, 6)
