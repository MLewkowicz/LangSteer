"""Learned neural network forecaster for trajectory prediction.

Trains a neural network to predict clean trajectories from noisy ones, conditioned
on observations and diffusion timesteps.
"""

from typing import Optional, List, Any
import torch
import torch.nn as nn

from forecasters.base_forecaster import BaseForecaster
from policies.dp3_components.positional_embed import SinusoidalPosEmb
from policies.dp3_components.encoder import DP3Encoder


class TrajectoryForecaster(BaseForecaster, nn.Module):
    """
    Learned forecaster that predicts clean trajectories from noisy ones.

    This is a pure prediction module with no cost functions or steering logic.
    It learns to map (noisy_trajectory, timestep, obs_encoding) -> clean_trajectory.

    Architecture:
        1. Time embedding: Sinusoidal positional embedding for timestep
        2. Trajectory encoder: MLP or Conv1D to process noisy trajectory
        3. Feature fusion: Concatenate obs_encoding, time_embed, traj_features
        4. Decoder: MLP to predict clean trajectory

    Args:
        obs_encoding_dim: Dimension of observation encoding (e.g., 320 for DP3)
        trajectory_dim: Action dimension (e.g., 7 for CALVIN)
        horizon: Prediction horizon (e.g., 16)
        time_embed_dim: Timestep embedding dimension (default: 128)
        traj_encoder_type: "mlp" or "conv1d" (default: "mlp")
        hidden_dims: List of hidden layer dimensions for decoder (default: [512, 512, 256])
        encoder: Optional DP3Encoder for observation encoding (default: None, expects pre-computed encodings)
        use_layernorm: Whether to use LayerNorm in MLPs (default: False)

    Example:
        >>> forecaster = TrajectoryForecaster(obs_encoding_dim=320, trajectory_dim=7, horizon=16)
        >>> forecaster.train()  # Training mode
        >>> x_0_hat = forecaster(x_t, timestep, obs_encoding)
        >>>
        >>> # Later, in a steering module:
        >>> x_0_hat = forecaster.forecast(x_t, timestep, obs_encoding)
        >>> cost = cost_fn(x_0_hat)
        >>> guidance = torch.autograd.grad(cost, x_t)[0]
    """

    def __init__(
        self,
        obs_encoding_dim: int,
        trajectory_dim: int,
        horizon: int,
        time_embed_dim: int = 128,
        traj_encoder_type: str = "mlp",
        hidden_dims: Optional[List[int]] = None,
        encoder: Optional[DP3Encoder] = None,
        use_layernorm: bool = False,
    ):
        """
        Initialize trajectory forecaster.

        Args:
            obs_encoding_dim: Observation encoding dimension (e.g., 320)
            trajectory_dim: Action dimension (e.g., 7)
            horizon: Prediction horizon (e.g., 16)
            time_embed_dim: Timestep embedding dimension
            traj_encoder_type: "mlp" or "conv1d"
            hidden_dims: Decoder hidden dimensions
            encoder: Optional observation encoder (DP3Encoder)
            use_layernorm: Whether to use LayerNorm
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 512, 256]

        self.obs_encoding_dim = obs_encoding_dim
        self.trajectory_dim = trajectory_dim
        self.horizon = horizon
        self.time_embed_dim = time_embed_dim
        self.traj_encoder_type = traj_encoder_type
        self.hidden_dims = hidden_dims
        self.use_layernorm = use_layernorm

        # Optional observation encoder
        self.encoder = encoder

        # 1. Time embedding (sinusoidal positional embedding)
        self.time_embed = SinusoidalPosEmb(time_embed_dim)

        # 2. Trajectory encoder
        traj_flat_dim = horizon * trajectory_dim  # e.g., 16 * 7 = 112

        if traj_encoder_type == "mlp":
            # MLP: flatten trajectory and process
            traj_feat_dim = 256
            traj_encoder_layers = [
                nn.Linear(traj_flat_dim, 256),
                nn.LayerNorm(256) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(256, traj_feat_dim),
                nn.LayerNorm(traj_feat_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
            ]
            self.traj_encoder = nn.Sequential(*traj_encoder_layers)
            self.traj_feat_dim = traj_feat_dim

        elif traj_encoder_type == "conv1d":
            # Conv1D: process temporal sequence
            # Input: (B, trajectory_dim, horizon)
            traj_feat_dim = 256
            self.traj_encoder = nn.Sequential(
                nn.Conv1d(trajectory_dim, 64, kernel_size=3, padding=1),
                nn.LayerNorm([64, horizon]) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.LayerNorm([128, horizon]) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.LayerNorm([256, horizon]) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # Global pooling: (B, 256, 1)
            )
            self.traj_feat_dim = traj_feat_dim

        else:
            raise ValueError(f"Unknown traj_encoder_type: {traj_encoder_type}")

        # 3. Feature fusion dimension
        fusion_dim = obs_encoding_dim + time_embed_dim + self.traj_feat_dim

        # 4. Decoder MLP
        decoder_layers = []
        prev_dim = fusion_dim
        for hidden_dim in hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                decoder_layers.append(nn.LayerNorm(hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, traj_flat_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode_observation(self, obs_dict: dict) -> torch.Tensor:
        """
        Encode observation using internal encoder.

        Args:
            obs_dict: Dictionary with 'point_cloud', 'agent_pos', etc.

        Returns:
            obs_encoding: (B, obs_encoding_dim) observation features

        Raises:
            ValueError: If encoder is not provided
        """
        if self.encoder is None:
            raise ValueError("Encoder not provided. Pass encoder during initialization or provide pre-computed obs_encoding.")

        return self.encoder(obs_dict)

    def forecast(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        obs_encoding: Optional[torch.Tensor] = None,
        obs_dict: Optional[dict] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forecast clean trajectory from noisy trajectory.

        Args:
            noisy_trajectory: (B, horizon, trajectory_dim) noisy trajectory x_t
            timestep: (B,) or int, diffusion timestep t in [0, num_train_timesteps)
            obs_encoding: (B, obs_encoding_dim) observation features (if pre-computed)
            obs_dict: Dictionary of observations (if obs_encoding not provided)
            **kwargs: Additional arguments (unused)

        Returns:
            clean_trajectory: (B, horizon, trajectory_dim) forecasted x_0

        Note:
            Either obs_encoding or obs_dict must be provided.
            All trajectories are in normalized space.
        """
        B = noisy_trajectory.shape[0]

        # 1. Get observation encoding
        if obs_encoding is None:
            if obs_dict is None:
                raise ValueError("Either obs_encoding or obs_dict must be provided")
            obs_encoding = self.encode_observation(obs_dict)

        # 2. Encode timestep
        # Ensure timestep is a tensor
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep] * B, device=noisy_trajectory.device)
        elif timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).expand(B)
        elif timestep.shape[0] == 1 and B > 1:
            timestep = timestep.expand(B)

        time_embed = self.time_embed(timestep)  # (B, time_embed_dim)

        # 3. Encode trajectory
        if self.traj_encoder_type == "mlp":
            # Flatten trajectory: (B, H, D) -> (B, H*D)
            traj_flat = noisy_trajectory.flatten(start_dim=1)  # (B, H*D)
            traj_features = self.traj_encoder(traj_flat)  # (B, traj_feat_dim)

        elif self.traj_encoder_type == "conv1d":
            # Transpose for Conv1D: (B, H, D) -> (B, D, H)
            traj_transposed = noisy_trajectory.transpose(1, 2)  # (B, D, H)
            traj_features = self.traj_encoder(traj_transposed)  # (B, traj_feat_dim, 1)
            traj_features = traj_features.squeeze(-1)  # (B, traj_feat_dim)

        # 4. Fuse features
        fused_features = torch.cat([obs_encoding, time_embed, traj_features], dim=-1)  # (B, fusion_dim)

        # 5. Decode to clean trajectory
        clean_flat = self.decoder(fused_features)  # (B, H*D)

        # 6. Reshape to trajectory
        clean_trajectory = clean_flat.view(B, self.horizon, self.trajectory_dim)  # (B, H, D)

        return clean_trajectory

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        obs_encoding: Optional[torch.Tensor] = None,
        obs_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        """Forward pass (alias for forecast)."""
        return self.forecast(noisy_trajectory, timestep, obs_encoding, obs_dict)

    def __repr__(self):
        return (
            f"TrajectoryForecaster("
            f"obs_encoding_dim={self.obs_encoding_dim}, "
            f"trajectory_dim={self.trajectory_dim}, "
            f"horizon={self.horizon}, "
            f"traj_encoder_type={self.traj_encoder_type}, "
            f"has_encoder={self.encoder is not None})"
        )
