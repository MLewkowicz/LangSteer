"""Concrete environment adapters."""

from envs.calvin import CalvinEnvironment
from envs.rlbench import RLBenchEnvironment

# Isaac Sim is an optional dependency — import only when available
try:
    from envs.isaac_sim import IsaacSimEnvironment
except ImportError:
    IsaacSimEnvironment = None  # type: ignore[assignment, misc]

__all__ = ["CalvinEnvironment", "RLBenchEnvironment", "IsaacSimEnvironment"]
