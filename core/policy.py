"""Abstract interface for Policy models."""

from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING
from core.types import Observation, Action

if TYPE_CHECKING:
    from core.steering import BaseSteering


class BasePolicy(ABC):
    """
    Abstract interface for Policy models.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Loads model weights."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clears internal temporal buffers (e.g. for Recurrent/Transformer history)."""
        pass

    @abstractmethod
    def forward(self, obs: Observation, steering: Optional["BaseSteering"] = None) -> Action:
        """
        Predicts an action given an observation, optionally guided by a steering module.
        
        Args:
            obs: Standardized Observation.
            steering: Optional steering module to influence generation (e.g., during denoising).
            
        Returns:
            Action: The predicted action trajectory.
        """
        pass
