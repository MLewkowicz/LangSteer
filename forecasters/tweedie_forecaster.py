"""Tweedie forecaster for analytical trajectory prediction.

Uses Tweedie's formula with the diffusion model's noise prediction to estimate
clean trajectories from noisy ones.
"""

from typing import Optional, Any
import torch

from forecasters.base_forecaster import BaseForecaster


class TweedieForecaster(BaseForecaster):
    """
    Analytical forecaster using Tweedie's formula.

    Formula: x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)

    This is a pure prediction module with no learnable parameters, cost functions,
    or steering logic. It uses the diffusion model's noise prediction to estimate
    the clean trajectory.

    Args:
        policy: DP3 policy object containing model and noise_scheduler
        use_ema: Whether to use EMA model if available (default: False)

    Example:
        >>> forecaster = TweedieForecaster(policy)
        >>> x_0_hat = forecaster.forecast(x_t, timestep, obs_encoding)
    """

    def __init__(self, policy, use_ema: bool = False):
        """
        Initialize Tweedie forecaster.

        Args:
            policy: DP3 policy object with .model and .noise_scheduler
            use_ema: Whether to use EMA model if available
        """
        self.policy = policy
        self.use_ema = use_ema

        # Access model (use EMA if requested and available)
        if use_ema and hasattr(policy, 'ema_model'):
            self.model = policy.ema_model
        else:
            self.model = policy.model

        self.noise_scheduler = policy.noise_scheduler

        # Cache scheduler attributes
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod  # alpha_bar_t values
        self.prediction_type = self.noise_scheduler.config.prediction_type

    def forecast(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        obs_encoding: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forecast clean trajectory using Tweedie's formula.

        Args:
            noisy_trajectory: (B, horizon, action_dim) noisy trajectory x_t
            timestep: (B,) or int, diffusion timestep t in [0, num_train_timesteps)
            obs_encoding: (B, obs_dim) observation features (required)
            **kwargs: Additional arguments (unused)

        Returns:
            clean_trajectory: (B, horizon, action_dim) forecasted x_0

        Note:
            - All trajectories are in normalized space
            - Handles prediction_type: 'epsilon', 'sample', 'v_prediction'
            - For 'sample' type, model already predicts x_0 directly
        """
        # Ensure timestep is a tensor
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep], device=noisy_trajectory.device)

        # Handle scalar timestep (broadcast to batch)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        # Get model prediction
        with torch.no_grad():
            model_output = self.model(
                sample=noisy_trajectory,
                timestep=timestep,
                local_cond=None,
                global_cond=obs_encoding
            )

        # Get alpha_bar_t for the timesteps
        alpha_bar_t = self.alphas_cumprod[timestep].to(noisy_trajectory.device)  # (B,)

        # Reshape for broadcasting with (B, H, D) trajectories
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1)  # (B, 1, 1)

        # Compute sqrt values
        alpha_t = torch.sqrt(alpha_bar_t)  # sqrt(alpha_bar_t)
        sigma_t = torch.sqrt(1.0 - alpha_bar_t)  # sqrt(1 - alpha_bar_t)

        # Apply Tweedie formula based on prediction type
        if self.prediction_type == "sample":
            # Model already predicts x_0 directly
            x_0_hat = model_output

        elif self.prediction_type == "epsilon":
            # Model predicts noise, apply Tweedie formula
            # x_0 = (x_t - sigma_t * epsilon) / alpha_t
            epsilon = model_output
            x_0_hat = (noisy_trajectory - sigma_t * epsilon) / alpha_t

        elif self.prediction_type == "v_prediction":
            # Model predicts velocity, convert to epsilon first
            # v = alpha_t * epsilon - sigma_t * x_0
            # => epsilon = (v + sigma_t * x_0) / alpha_t
            # But we don't have x_0 yet, so rearrange:
            # v = alpha_t * epsilon - sigma_t * x_0
            # x_t = alpha_t * x_0 + sigma_t * epsilon
            # Solve for x_0: x_0 = alpha_t * x_t - sigma_t * v
            v = model_output
            x_0_hat = alpha_t * noisy_trajectory - sigma_t * v

        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return x_0_hat

    def __repr__(self):
        return (
            f"TweedieForecaster("
            f"prediction_type={self.prediction_type}, "
            f"use_ema={self.use_ema})"
        )
