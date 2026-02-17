"""Abstract interface for trajectory forecasters.

Forecasters are pure prediction modules that predict clean trajectories (x_0)
from noisy trajectories (x_t). They have no cost functions or steering logic.

Steering modules (separate) will use forecasters to get x_0_hat, then compute
gradients of cost functions C(x_0_hat) for guidance.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import torch


class BaseForecaster(ABC):
    """
    Abstract interface for trajectory forecasters.

    Forecasters predict clean trajectories (x_0) from noisy trajectories (x_t).
    They are pure prediction modules with no cost functions or steering logic.

    Example usage:
        >>> forecaster = TweedieForecaster(policy)
        >>> x_0_hat = forecaster.forecast(x_t, timestep, obs_encoding)
        >>> # Later, in a steering module:
        >>> cost = cost_fn(x_0_hat)
        >>> guidance = torch.autograd.grad(cost, x_t)[0]
    """

    @abstractmethod
    def forecast(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        obs_encoding: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forecast clean trajectory from noisy trajectory.

        This is a pure prediction method with no cost functions or steering logic.

        Args:
            noisy_trajectory: (B, horizon, action_dim) noisy trajectory x_t
            timestep: (B,) or int, diffusion timestep t in [0, num_train_timesteps)
            obs_encoding: (B, obs_dim) observation features (optional, required by some forecasters)
            **kwargs: Additional forecaster-specific arguments

        Returns:
            clean_trajectory: (B, horizon, action_dim) forecasted x_0

        Note:
            All trajectories are in normalized space (via policy's normalizer).
            Forecaster predicts normalized clean trajectory.
        """
        pass

    def __call__(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        obs_encoding: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Convenience method to call forecast()."""
        return self.forecast(noisy_trajectory, timestep, obs_encoding, **kwargs)
