"""PointNet-based encoder for DP3 policy.

Extracted from:
- diffusion_policy_3d/model/vision/pointnet_extractor.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    Args:
        input_dim: Dimension of the input vector
        output_dim: Output dimension
        net_arch: Architecture of the neural net
            It represents the number of units per layer.
            The length of this list is the number of layers.
        activation_fn: The activation function to use after each layer.
        squash_output: Whether to squash the output using a Tanh
            activation function

    Returns:
        List of nn.Module layers
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """PointNet encoder for point clouds with XYZ + RGB channels."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs):
        """
        Args:
            in_channels: feature size of input (typically 6 for XYZ+RGB)
            out_channels: output feature dimension
            use_layernorm: whether to use LayerNorm in MLP
            final_norm: normalization for final projection ('layernorm' or 'none')
            use_projection: whether to use final projection layer
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x):
        """
        Args:
            x: (B, N, C) point cloud tensor

        Returns:
            (B, out_channels) global feature vector
        """
        x = self.mlp(x)
        x = torch.max(x, 1)[0]  # Global max pooling
        x = self.final_projection(x)
        return x


class PointNetEncoderXYZ(nn.Module):
    """PointNet encoder for point clouds with XYZ coordinates only."""

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs):
        """
        Args:
            in_channels: feature size of input (must be 3 for XYZ)
            out_channels: output feature dimension
            use_layernorm: whether to use LayerNorm in MLP
            final_norm: normalization for final projection ('layernorm' or 'none')
            use_projection: whether to use final projection layer
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud tensor with XYZ coordinates

        Returns:
            (B, out_channels) global feature vector
        """
        x = self.mlp(x)
        x = torch.max(x, 1)[0]  # Global max pooling
        x = self.final_projection(x)
        return x


class DP3Encoder(nn.Module):
    """
    Main encoder for DP3 policy.
    Combines point cloud encoding (via PointNet) with agent state processing.
    """

    def __init__(self,
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        """
        Args:
            observation_space: Dictionary defining observation shapes
                - 'point_cloud': [num_points, channels]
                - 'agent_pos': [state_dim]
                - 'imagin_robot': [num_points, channels] (optional)
            img_crop_shape: Unused, kept for compatibility
            out_channel: Total output dimension (PointNet output + state MLP output)
            state_mlp_size: Architecture for agent state MLP
            state_mlp_activation_fn: Activation function for state MLP
            pointcloud_encoder_cfg: Config dict for PointNet encoder
            use_pc_color: Whether to use RGB colors (6 channels) or just XYZ (3 channels)
            pointnet_type: Type of PointNet encoder ('pointnet' only supported)
        """
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        # Initialize PointNet encoder
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg['in_channels'] = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg['in_channels'] = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        # Initialize state MLP
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(
            self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        """
        Encode observations to feature vector.

        Args:
            observations: Dictionary containing:
                - 'point_cloud': (B, N, C) point cloud
                - 'agent_pos': (B, state_dim) agent state
                - 'imagin_robot': (B, N_img, C) optional imagined robot points

        Returns:
            (B, n_output_channels) concatenated feature vector
        """
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")

        # Concatenate imagined robot points if available
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]]  # align the last dim
            points = torch.concat([points, img_points], dim=1)

        # Encode point cloud
        pn_feat = self.extractor(points)  # (B, out_channel)

        # Encode agent state
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # (B, state_mlp_out_dim)

        # Concatenate features
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat

    def output_shape(self):
        """Get output feature dimension."""
        return self.n_output_channels
