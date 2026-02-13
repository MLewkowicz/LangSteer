"""DP3 Components - Minimal extraction from 3D-Diffusion-Policy.

This package contains the core components needed for DP3 inference:
- dp3_policy: Main DP3 policy class with diffusion sampling
- encoder: PointNet-based observation encoder
- unet: Conditional 1D U-Net for diffusion denoising
- normalizer: Linear normalization for observations and actions
- utils: Utility functions (FPS, dict operations, module helpers)
- conv_components: 1D convolution building blocks
- positional_embed: Sinusoidal positional embedding for timesteps
"""

from policies.dp3_components.dp3_policy import DP3
from policies.dp3_components.encoder import DP3Encoder, PointNetEncoderXYZ, PointNetEncoderXYZRGB
from policies.dp3_components.unet import ConditionalUnet1D, ConditionalResidualBlock1D
from policies.dp3_components.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from policies.dp3_components.utils import (
    sample_farthest_points,
    dict_apply,
    optimizer_to,
    ModuleAttrMixin
)
from policies.dp3_components.conv_components import Downsample1d, Upsample1d, Conv1dBlock
from policies.dp3_components.positional_embed import SinusoidalPosEmb

__all__ = [
    'DP3',
    'DP3Encoder',
    'PointNetEncoderXYZ',
    'PointNetEncoderXYZRGB',
    'ConditionalUnet1D',
    'ConditionalResidualBlock1D',
    'LinearNormalizer',
    'SingleFieldLinearNormalizer',
    'sample_farthest_points',
    'dict_apply',
    'optimizer_to',
    'ModuleAttrMixin',
    'Downsample1d',
    'Upsample1d',
    'Conv1dBlock',
    'SinusoidalPosEmb',
]
