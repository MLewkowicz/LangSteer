"""Unified visualization system for LangSteer experiments.

This module consolidates all visualization functionality previously spread across
multiple scripts (rollout_reference.py, visualize_cameras.py, visualize_trajectories.py,
visualize_reference.py) into a single config-driven system.
"""

from .config import VisualizationConfig
from .manager import VisualizationManager

__all__ = [
    'VisualizationConfig',
    'VisualizationManager',
]
