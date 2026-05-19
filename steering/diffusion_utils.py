"""Shared Tweedie / scheduler helpers used by both guidance branches.

Position and rotation guidance both need to (a) read ᾱ_t from a DDPM-style
scheduler and (b) recover x₀ from a noisy sample + ε prediction via Tweedie's
formula. They run on separate schedulers (position uses scaled_linear,
rotation uses squaredcos), so they call these helpers with their own scheduler
reference but otherwise share the math.

Public surface:
    get_alpha_bar(scheduler, timestep, *, device, clamp_min, clamp_max)
    predict_x0(x_t, eps, alpha_bar, *, prediction_type)
    epsilon_coeff(alpha_bar)
    dps_coeff(alpha_bar)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def get_alpha_bar(
    scheduler: Any,
    timestep: Any,
    *,
    device: str,
    clamp_min: float = 1e-6,
    clamp_max: float = 1.0 - 1e-6,
) -> torch.Tensor:
    """Look up cumulative noise schedule ᾱ_t on any DDPM-style scheduler.

    Returns a 1-d tensor (shape `(1,)` for a scalar timestep). When `scheduler`
    is None — pre-inference, before the policy wrapper has wired its schedulers
    in — falls back to 0.5 so callers don't crash; downstream guidance is
    effectively no-op in that regime.
    """
    if scheduler is None:
        return torch.tensor(0.5, device=device)
    if isinstance(timestep, torch.Tensor):
        t_idx = timestep.long()
    else:
        t_idx = torch.tensor([timestep], device=device, dtype=torch.long)
    alpha_bar = scheduler.alphas_cumprod[t_idx]
    return torch.clamp(alpha_bar, min=clamp_min, max=clamp_max)


def predict_x0(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    alpha_bar: torch.Tensor,
    *,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    """Tweedie's formula — recover clean x₀ from noisy x_t and ε prediction.

    Supports the three DDPM prediction parameterizations:
      - 'epsilon':       x₀ = (x_t − √(1−ᾱ)·ε) / √ᾱ
      - 'v_prediction':  x₀ = √ᾱ·x_t − √(1−ᾱ)·ε
      - 'sample':        x₀ = ε   (model output already IS x₀)

    `alpha_bar` is reshaped to be broadcast-compatible with x_t / eps.
    """
    if alpha_bar.dim() < 3:
        alpha_bar = alpha_bar.view(-1, 1, 1)
    alpha_bar = torch.clamp(alpha_bar, min=1e-6)

    if prediction_type == "sample":
        return eps
    if prediction_type == "epsilon":
        return (x_t - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
    if prediction_type == "v_prediction":
        return torch.sqrt(alpha_bar) * x_t - torch.sqrt(1.0 - alpha_bar) * eps
    raise ValueError(f"Unknown prediction_type: {prediction_type}")


def epsilon_coeff(alpha_bar: torch.Tensor) -> torch.Tensor:
    """√ᾱ / √(1−ᾱ) — the chain-rule factor that maps Δx₀ → Δε.

    Lets a guidance term computed in x₀ space (e.g. a value-map gradient) be
    added directly to the model's ε prediction before the scheduler step.
    """
    return torch.sqrt(alpha_bar) / torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-6))


def dps_coeff(alpha_bar: torch.Tensor) -> torch.Tensor:
    """1 / √ᾱ — the multiplier used in DPS-style post-step corrections.

    Applied to x_{t-1} after the scheduler step rather than to ε.
    """
    return 1.0 / torch.sqrt(torch.clamp(alpha_bar, min=1e-6))
