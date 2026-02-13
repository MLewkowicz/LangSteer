"""1D convolution building blocks for diffusion U-Net.

Extracted from:
- diffusion_policy_3d/model/diffusion/conv1d_components.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample1d(nn.Module):
    """Downsampling layer for 1D sequences."""

    def __init__(self, dim):
        """
        Args:
            dim: Number of channels
        """
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        """
        Args:
            x: (B, C, T) input tensor

        Returns:
            (B, C, T//2) downsampled tensor
        """
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsampling layer for 1D sequences."""

    def __init__(self, dim):
        """
        Args:
            dim: Number of channels
        """
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        """
        Args:
            x: (B, C, T) input tensor

        Returns:
            (B, C, T*2) upsampled tensor
        """
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Basic 1D convolutional block: Conv1d -> GroupNorm -> Mish.
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        """
        Args:
            inp_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, inp_channels, T) input tensor

        Returns:
            (B, out_channels, T) output tensor
        """
        return self.block(x)
