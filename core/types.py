"""Standardized data transfer objects (DTOs) for the repository."""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np
import torch


@dataclass
class Observation:
    """
    Standardized observation container passed from Env to Policy.
    
    Attributes:
        rgb (Dict[str, np.ndarray]): Dictionary of camera views (e.g., 'front', 'wrist'). 
                                     Format: (H, W, C), uint8.
        depth (Dict[str, np.ndarray]): Optional depth maps. Format: (H, W), float32.
        pcd (Optional[np.ndarray]): fused point cloud. Shape (N, 3).
        proprio (np.ndarray): Robot joint states (pos + vel). Shape (D,).
        ee_pose (np.ndarray): End-effector pose (pos + quat). Shape (7,).
        instruction (str): The natural language task description.
    """
    rgb: Dict[str, np.ndarray]
    proprio: np.ndarray
    ee_pose: np.ndarray
    instruction: str
    depth: Optional[Dict[str, np.ndarray]] = None
    pcd: Optional[np.ndarray] = None


@dataclass
class Action:
    """
    Standardized action container passed from Policy to Env.
    
    Attributes:
        trajectory (np.ndarray): Sequence of predicted end-effector poses. 
                                 Shape (H, 7) or (H, 6) depending on relative/absolute.
                                 Base policies may predict a sequence, but Envs might only execute step 0.
        gripper (float): Gripper open/close state. Range [-1, 1] or [0, 1].
    """
    trajectory: np.ndarray
    gripper: float
