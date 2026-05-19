"""Unified visualization system for LangSteer experiments.

Renderer Protocol + Manager + per-renderer config. Iter 1 of Task 5 introduced
the Protocol; iter 2 strips dead infrastructure; iter 3 renames + extends
the live tk window with the OBJECT label; iter 4 wires the system into
`run_evaluation.py`.

Public API:
    Renderer            — Protocol describing the renderer contract.
    VisualizationManager — fan-out dispatcher over the enabled renderers.
    VisualizationConfig  — master config with toggles + sub-configs.
"""

from .base import Renderer
from .config import VisualizationConfig
from .manager import VisualizationManager

__all__ = [
    "Renderer",
    "VisualizationConfig",
    "VisualizationManager",
]
