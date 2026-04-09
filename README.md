# LangSteer: Language-Conditioned Diffusion Policy Steering

A research framework for language-conditioned diffusion policy steering on robotic manipulation benchmarks. Modifies the diffusion denoising process at inference time using external guidance without retraining the base policy.

## Overview

LangSteer enables **steering** — modifying the diffusion denoising process at inference time using external guidance (e.g., reference trajectories, spatial value maps) without retraining the base policy. The framework supports:

- **Policy**: 3D Diffuser Actor (extensible to new architectures)
- **Environments**: CALVIN, RLBench, Isaac Sim
- **Steering Methods**: Tweedie guidance, VoxPoser (LLM-generated value maps)

## Quick Start

```bash
# Setup (see SETUP.md for details)
uv sync

# Run experiment with default settings
uv run python scripts/run_experiment.py

# Run with steering guidance
uv run python scripts/run_experiment.py steering=tweedie env.task=open_drawer

# Run with VoxPoser steering
uv run python scripts/run_experiment.py steering=voxposer env.task=open_drawer

# Train Diffuser Actor
uv run python scripts/train_diffuser_actor.py training=diffuser_actor_calvin
```

## Repository Structure

```
LangSteer/
├── conf/                          # Hydra configuration
│   ├── config.yaml                # Main entry point
│   ├── env/                       # Environment configs
│   ├── policy/                    # Policy configs (diffuser_actor)
│   ├── steering/                  # Steering configs (tweedie, voxposer)
│   └── training/                  # Training configurations
│
├── core/                          # Abstract interfaces & types
│   ├── types.py                   # Observation, Action DTOs
│   ├── policy.py                  # BasePolicy
│   ├── env.py                     # BaseEnvironment
│   └── steering.py                # BaseSteering
│
├── envs/                          # Environment adapters
│   ├── calvin.py                  # CALVIN adapter
│   ├── rlbench.py                 # RLBench adapter
│   └── isaac_sim.py               # Isaac Sim adapter
│
├── policies/                      # Policy implementations
│   ├── diffuser_actor.py          # Diffuser Actor wrapper
│   └── diffuser_actor_components/ # Model architecture
│
├── steering/                      # Steering methods
│   ├── tweedie.py                 # Tweedie analytical guidance
│   └── voxposer_steering.py       # VoxPoser value-map guidance
│
├── voxposer/                      # VoxPoser LLM value map generation
│
├── forecasters/                   # Trajectory forecaster models
│
├── training/                      # Training infrastructure
│   ├── common/                    # Shared utilities
│   ├── policies/diffuser_actor/   # Diffuser Actor training
│   └── forecasters/               # Forecaster training
│
├── scripts/                       # Entry points
│   ├── run_experiment.py          # Main evaluation loop
│   └── train_diffuser_actor.py    # Training entry point
│
├── SETUP.md                       # Installation guide
└── README.md                      # This file
```

## Core Components

### Data Transfer Objects ([core/types.py](core/types.py))

- **`Observation`**: RGB images, depth, point clouds, proprioception, end-effector pose, language instruction
- **`Action`**: Trajectory (sequence of poses) and gripper state

### Abstract Interfaces

- **`BaseEnvironment`** ([core/env.py](core/env.py)): Environment adapter interface
- **`BasePolicy`** ([core/policy.py](core/policy.py)): Policy interface with optional steering
- **`BaseSteering`** ([core/steering.py](core/steering.py)): Guidance mechanism interface

## Configuration

All experiments configured via Hydra YAML files in [conf/](conf/):

```bash
uv run python scripts/run_experiment.py \
    env=calvin \
    steering=tweedie \
    env.task=open_drawer \
    steering.guidance_strength=2.0
```

## Adding New Components

**New Policy:** Implement `BasePolicy`, add wrapper in `policies/`, config in `conf/policy/`

**New Environment:** Implement `BaseEnvironment`, add adapter in `envs/`, config in `conf/env/`

**New Steering Method:** Implement `BaseSteering`, add module in `steering/`, config in `conf/steering/`
