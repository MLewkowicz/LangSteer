"""Training infrastructure for LangSteer.

Organized by model type:
- training/common/ - Shared utilities (EMA, checkpointing, sampling, replay buffer)
- training/policies/dp3/ - DP3 policy training
- training/forecasters/trajectory/ - Trajectory forecaster training

Future models should follow this pattern:
- training/policies/<model_name>/
- training/forecasters/<forecaster_type>/
"""
