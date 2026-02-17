"""Training infrastructure for LangSteer.

Organized by model type:
- training/common/ - Shared utilities (EMA, checkpointing, sampling, etc.)
- training/policies/dp3/ - DP3 policy training
- training/forecasters/trajectory/ - Trajectory forecaster training

Future models should follow this pattern:
- training/policies/<model_name>/
- training/forecasters/<forecaster_type>/
"""

# Re-export commonly used components for backward compatibility
from .common import (
    EMAModel,
    TopKCheckpointManager,
    dict_apply,
    SequenceSampler
)

from .policies.dp3 import (
    DP3TrainingWorkspace,
    CalvinDataset,
    ReplayBuffer
)

from .forecasters.trajectory import ForecasterTrainingWorkspace

__all__ = [
    # Common utilities
    'EMAModel',
    'TopKCheckpointManager',
    'dict_apply',
    'SequenceSampler',
    # DP3 training
    'DP3TrainingWorkspace',
    'CalvinDataset',
    'ReplayBuffer',
    # Forecaster training
    'ForecasterTrainingWorkspace',
]
