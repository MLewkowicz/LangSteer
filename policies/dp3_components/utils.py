"""Utility functions for DP3 components.

Merged from:
- diffusion_policy_3d/utils/fps_ops.py
- diffusion_policy_3d/common/pytorch_util.py
- diffusion_policy_3d/model/common/module_attr_mixin.py
"""

from typing import Dict, Callable, List
import collections
import torch
import torch.nn as nn


# ==============================================================================
# Farthest Point Sampling (from fps_ops.py)
# ==============================================================================

def sample_farthest_points(points: torch.Tensor, K=None, **kwargs) -> tuple:
    """
    Memory-Optimized Farthest Point Sampling (FPS) in pure PyTorch.
    Compatible with pytorch3d.ops.sample_farthest_points API.
    Sampling indices are computed under torch.no_grad() to avoid building
    a huge computation graph when gradients are enabled (e.g. during rollout).
    Gradients flow only through the gathered point *values*, not the sampling.

    Args:
        points: (B, N, D) tensor of point coordinates
        K: Number of points to sample. Can be:
           - int
           - torch.Tensor (scalar or 1-element)
           - list with one element [num_points]
        **kwargs: Can contain 'K' as alternative way to pass number of points

    Returns:
        sampled_points: (B, K, D) tensor of sampled points
        indices: (B, K) tensor of indices into original points
    """
    device = points.device
    dtype = points.dtype
    B, N, D = points.shape

    # Handle K from kwargs if not provided directly
    if K is None:
        K = kwargs.get("K", None)
        if K is None:
            raise ValueError("K must be provided either as argument or in kwargs")

    if isinstance(K, torch.Tensor):
        num_points = K.item() if K.numel() == 1 else int(K[0].item())
    elif isinstance(K, (list, tuple)):
        num_points = int(K[0])
    else:
        num_points = int(K)

    if N <= num_points:
        indices = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
        padding = torch.full((B, num_points - N), N - 1, device=device, dtype=torch.long)
        indices = torch.cat([indices, padding], dim=1)
        sampled_points = points[torch.arange(B, device=device).unsqueeze(1), indices]
        return sampled_points, indices

    # --- MEMORY OPTIMIZED SAMPLING ---
    # Disable gradients strictly for index computation to avoid a gigantic graph.
    with torch.no_grad():
        centroids = torch.zeros((B, num_points), dtype=torch.long, device=device)
        distance = torch.ones((B, N), dtype=dtype, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), device=device, dtype=torch.long)
        batch_indices = torch.arange(B, device=device, dtype=torch.long)

        for i in range(num_points):
            centroids[:, i] = farthest
            centroid = points[batch_indices, farthest, :].view(B, 1, D)
            dist = torch.sum((points - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=-1)[1]

    # Gather points: gradients flow through the *values* of the points only.
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_points)
    sampled_points = points[batch_indices, centroids]

    return sampled_points, centroids


# ==============================================================================
# PyTorch Utilities (from pytorch_util.py)
# ==============================================================================

def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    """Apply function to all tensors in a (nested) dictionary."""
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def pad_remaining_dims(x, target):
    """Pad tensor x to match the number of dimensions of target."""
    assert x.shape == target.shape[:len(x.shape)]
    return x.reshape(x.shape + (1,)*(len(target.shape) - len(x.shape)))


def dict_apply_split(
        x: Dict[str, torch.Tensor],
        split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
    """Apply split function to dictionary values and reorganize results."""
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results


def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    """Apply reduce function to list of dictionaries with same keys."""
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def optimizer_to(optimizer, device):
    """Move optimizer state to specified device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


# ==============================================================================
# Module Attribute Mixin (from module_attr_mixin.py)
# ==============================================================================

class ModuleAttrMixin(nn.Module):
    """Mixin to provide device and dtype properties for modules."""

    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        """Get device of module parameters."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        """Get dtype of module parameters."""
        return next(iter(self.parameters())).dtype
