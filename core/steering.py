"""Abstract interface for inference-time steering"""

from abc import ABC, abstractmethod
from typing import Any
import torch


class BaseSteering(ABC):
    """
    Abstract interface for Inference-time steering (guidance).
    Used to inject gradients or bias into the diffusion denoising loop.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def get_guidance(self, current_sample: torch.Tensor, timestep: int, obs_embedding: Any) -> torch.Tensor:
        """
        Calculates the guidance signal (e.g., gradient of energy function) 
        to be added to the noise prediction.
        
        Args:
            current_sample: The noisy latents/trajectory at current diffusion step.
            timestep: The current diffusion timestep.
            obs_embedding: Context from the observation (e.g., condition features).
            
        Returns:
            torch.Tensor: The guidance gradient or offset vector.
        """
        pass
