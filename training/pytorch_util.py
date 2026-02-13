"""PyTorch utilities for training.

Minimal utilities extracted from 3D-Diffusion-Policy for dict operations.
"""

from typing import Dict, Callable
import torch


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    """Recursively apply a function to all values in a nested dictionary.

    Args:
        x: Dictionary with potentially nested dictionaries
        func: Function to apply to each tensor value

    Returns:
        Dictionary with same structure, but func applied to all tensor values
    """
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
