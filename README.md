# LangSteer: Language-Conditioned Diffusion Policy Steering

A modular, scalable research framework for evaluating language-conditioned diffusion policies with inference-time steering guidance on robotic manipulation benchmarks.

## Overview

LangSteer enables **steering** — modifying the diffusion denoising process at inference time using external guidance (e.g., reference trajectories, dynamics models) without retraining the base policy. The framework supports:

- **Policies**: DP3, DiffuserActor (extensible to new models)
- **Environments**: CALVIN, RLBench (extensible to new benchmarks)
- **Steering Methods**: Tweedie guidance, DynaGuide (extensible to new methods)

## Quick Start

```bash
# Setup (see SETUP.md for details)
uv sync

# Run experiment with default settings
python scripts/run_experiment.py

# Run with steering guidance
python scripts/run_experiment.py steering=tweedie env.task=open_drawer

# Visualize rollout in PyBullet
python scripts/run_experiment.py visualization.render=true

# Train DP3 policy
python scripts/train_dp3.py training=training/policies/dp3/calvin
```

## Architecture

LangSteer follows three core principles:

1. **Interface Segregation**: Core logic relies on abstract interfaces ([core/](core/)), not concrete implementations
2. **Composition over Inheritance**: Experiments composed dynamically via Hydra configs
3. **Standardization**: All components communicate via strict `Observation` and `Action` DTOs

## Repository Structure

```
LangSteer/
├── conf/                          # Hydra configuration
│   ├── config.yaml                # Main entry point
│   ├── env/                       # Environment configs (CALVIN, RLBench)
│   ├── policy/                    # Policy configs (DP3, etc.)
│   ├── steering/                  # Steering configs (Tweedie, DynaGuide)
│   ├── visualization/             # Visualization settings
│   └── training/                  # Training configurations
│       ├── policies/dp3/          # DP3 training configs
│       └── forecasters/           # Forecaster training configs
│
├── core/                          # Abstract interfaces & types
│   ├── types.py                   # Observation, Action DTOs
│   ├── policy.py                  # BasePolicy
│   ├── env.py                     # BaseEnvironment
│   └── steering.py                # BaseSteering
│
├── envs/                          # Environment adapters
│   ├── calvin.py                  # CALVIN adapter
│   └── calvin_utils/              # CALVIN-specific utilities
│
├── policies/                      # Policy implementations
│   ├── dp3.py                     # DP3 policy wrapper
│   └── dp3_components/            # DP3 architecture components
│
├── steering/                      # Steering methods
│   ├── tweedie.py                 # Tweedie guidance
│   └── dynaguide.py               # Dynamics-based guidance
│
├── forecasters/                   # Trajectory forecaster models
│   ├── trajectory_forecaster.py   # Neural forecaster
│   └── tweedie_forecaster.py      # Analytical forecaster
│
├── training/                      # Training infrastructure
│   ├── common/                    # Shared utilities (EMA, checkpointing)
│   ├── policies/dp3/              # DP3 training
│   │   ├── trainer.py             # Training workspace
│   │   ├── dataset.py             # CALVIN dataset loader
│   │   └── preprocessing/         # Data preprocessing
│   └── forecasters/trajectory/    # Forecaster training
│
├── visualization/                 # Unified visualization system
│   ├── manager.py                 # VisualizationManager
│   ├── config.py                  # Visualization configs
│   └── renderers/                 # Camera, PyBullet, Plotly renderers
│
├── utils/                         # Utilities
│   ├── rollout/                   # Episode runner & data collection
│   ├── state_management/          # Environment snapshots
│   └── reference_trajectory_loader.py
│
├── scripts/                       # Entry points
│   ├── run_experiment.py          # Main evaluation loop
│   ├── train_dp3.py               # DP3 training entry point
│   └── slurm_train_dp3.sh         # SLURM job submission
│
├── docs/                          # Documentation
│   ├── guides/                    # How-to guides
│   │   ├── experiments.md         # Running experiments
│   │   ├── training.md            # Training models
│   │   └── visualization.md       # Visualization system
│   └── reference/                 # Reference docs
│       ├── forecasters.md         # Forecaster documentation
│       └── rollout.md             # Rollout system
│
├── SETUP.md                       # Installation guide
└── README.md                      # This file
```

## Key Features

### Unified Visualization System

Replace multiple visualization scripts with a single config-driven interface:

```bash
# Camera feeds
python scripts/run_experiment.py visualization.cameras=true

# PyBullet GUI rendering
python scripts/run_experiment.py visualization.render=true

# 3D trajectory analysis
python scripts/run_experiment.py visualization.trajectory_3d=true

# Combine multiple modes
python scripts/run_experiment.py \
    visualization.render=true \
    visualization.cameras=true
```

See [docs/guides/visualization.md](docs/guides/visualization.md) for details.

### Scalable Training Structure

Organized by model type for easy expansion:

```bash
# Train DP3
python scripts/train_dp3.py

# Train forecaster
python -m training.forecasters.trajectory.trainer

# Add new model → training/policies/<model_name>/
```

See [docs/guides/training.md](docs/guides/training.md) for details.

### Steering Methods

Modify policy behavior at inference time without retraining:

```bash
# Tweedie guidance (analytical, no training)
python scripts/run_experiment.py steering=tweedie

# Custom guidance strength
python scripts/run_experiment.py \
    steering=tweedie \
    steering.guidance_strength=2.0
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
# Override via command line
python scripts/run_experiment.py \
    env=calvin \
    policy=dp3 \
    steering=tweedie \
    env.task=open_drawer

# Or create custom experiment configs
python scripts/run_experiment.py experiment=my_experiment
```

## Documentation

- **[Setup Guide](SETUP.md)** - Installation and environment setup
- **[Experiments Guide](docs/guides/experiments.md)** - Running experiments
- **[Training Guide](docs/guides/training.md)** - Training policies and forecasters
- **[Visualization Guide](docs/guides/visualization.md)** - Visualization system
- **[Forecasters Reference](docs/reference/forecasters.md)** - Forecaster models

## Development

### Dependencies

Managed via `uv`:
```bash
uv sync
```

### Code Quality

```bash
# Type checking
mypy .

# Formatting
ruff format .
ruff check .
```

### Adding New Components

**New Policy:**
1. Implement `BasePolicy` interface
2. Add policy wrapper in `policies/`
3. Create config in `conf/policy/`

**New Environment:**
1. Implement `BaseEnvironment` interface
2. Add environment adapter in `envs/`
3. Create config in `conf/env/`

**New Steering Method:**
1. Implement `BaseSteering` interface
2. Add steering module in `steering/`
3. Create config in `conf/steering/`

## Citation

```bibtex
@misc{langsteer2025,
  title={LangSteer: Language-Conditioned Diffusion Policy Steering},
  author={Your Name},
  year={2025}
}
```

## License

[Add license information]
