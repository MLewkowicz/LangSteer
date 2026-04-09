"""Training infrastructure for LangSteer.

Organized by model type:
- training/common/ - Shared utilities (EMA, checkpointing, sampling, replay buffer)
- training/policies/diffuser_actor/ - Diffuser Actor policy training
- training/forecasters/trajectory/ - Trajectory forecaster training
"""
