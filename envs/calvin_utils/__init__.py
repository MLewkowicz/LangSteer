"""CALVIN utilities for environment integration.

This package provides utilities for working with the CALVIN dataset and environment:
- observation: RGB-D processing and point cloud generation
- language_ann: Language annotation loading
- gym_wrapper: Gym-compatible CALVIN environment wrapper
"""

from envs.calvin_utils.observation import process_calvin_obs, deproject_depth, sample_farthest_points_numpy
from envs.calvin_utils.language_ann import load_language_annotations, get_instruction_for_task
from envs.calvin_utils.gym_wrapper import CalvinGymWrapper

__all__ = [
    'process_calvin_obs',
    'deproject_depth',
    'sample_farthest_points_numpy',
    'load_language_annotations',
    'get_instruction_for_task',
    'CalvinGymWrapper',
]
