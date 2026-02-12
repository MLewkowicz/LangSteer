"""CALVIN environment adapter."""

from typing import Tuple, Dict, Any
import numpy as np
from core.env import BaseEnvironment
from core.types import Observation, Action


class CalvinEnvironment(BaseEnvironment):
    """
    Adapter for CALVIN manipulation environment.
    Handles conversion between CALVIN's native format and standardized Observation/Action.
    """
    
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)
        # TODO: Initialize CALVIN environment
        # Use lazy imports to avoid dependency issues:
        # from calvin_utils import ...
        self._env = None
        self._current_task = cfg.get("task", "open_drawer")
    
    def reset(self) -> Observation:
        """
        Resets the environment to an initial state and returns the first observation.
        """
        # TODO: Implement CALVIN reset logic
        # Convert CALVIN observation format to standardized Observation
        raise NotImplementedError("CalvinEnvironment.reset() not yet implemented")
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Executes an action in the environment.
        """
        # TODO: Convert Action to CALVIN format and step
        # Convert CALVIN observation back to standardized Observation
        raise NotImplementedError("CalvinEnvironment.step() not yet implemented")
    