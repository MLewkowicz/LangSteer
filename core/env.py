"""Abstract interface for environments."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from core.types import Observation, Action


class BaseEnvironment(ABC):
    """
    Abstract wrapper for environments.
    Handles loading simulators, task specification, and standardized I/O.
    """
    
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    @abstractmethod
    def reset(self) -> Observation:
        """
        Resets the environment to an initial state and returns the first observation.
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Executes an action in the environment.
        
        Args:
            action: Standardized Action object.
            
        Returns:
            observation (Observation): New state.
            reward (float): Step reward.
            done (bool): Whether the episode/task is finished.
            info (Dict): Debug info (e.g., success metrics).
        """
        pass
