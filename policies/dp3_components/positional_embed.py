"""Positional embedding for diffusion timesteps.

Extracted from:
- diffusion_policy_3d/model/diffusion/positional_embedding.py
"""

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for timesteps.
    Encodes diffusion timesteps as continuous embeddings.
    """

    def __init__(self, dim):
        """
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Encode timesteps as sinusoidal embeddings.

        Args:
            x: (B,) tensor of timestep values

        Returns:
            (B, dim) embedded timesteps
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
