# LangSteer: Language-Conditioned Value-Map Guidance for Diffusion Policies

A modular, scalable research repository for evaluating language-conditioned diffusion policies (e.g., DP3, DiffuserActor) on robotic manipulation benchmarks (CALVIN, RLBench). The system supports **Steering** — modifying the diffusion denoising process at inference time using external value maps (e.g., DynaGuide, Tweedie) derived from language instructions, without retraining the base policy.

## Architecture

The repository follows three core principles:

1. **Interface Segregation:** Core logic relies on abstract interfaces (`core/`), not concrete implementations.
2. **Composition over Inheritance:** Experiments are composed dynamically using Hydra configurations.
3. **Standardization:** All policies and environments communicate via strict `Observation` and `Action` dataclasses.

## Repository Structure

```
project_root/
├── conf/                     # Hydra Configuration Root
│   ├── config.yaml           # Main entry point (defaults)
│   ├── experiment/           # Composition of env + policy + steering
│   ├── env/                  # Environment parameters (task, seed, render_mode)
│   ├── policy/               # Policy architecture & checkpoints
│   └── steering/             # Steering parameters (guidance strength, horizon)
├── core/                     # THE CONTRACT (Abstract Base Classes & Types)
│   ├── __init__.py
│   ├── types.py              # Data Transfer Objects (DTOs)
│   ├── policy.py             # Base Policy Interface
│   ├── env.py                # Base Environment Interface
│   └── steering.py           # Base Steering Interface
├── envs/                     # Concrete Environments (Adapters)
│   ├── __init__.py
│   ├── calvin.py             # CALVIN Adapter
│   └── rlbench.py            # RLBench Adapter
├── policies/                 # Concrete Policies (Adapters)
│   ├── __init__.py
│   ├── dp3.py                # DP3 Wrapper
│   └── diffuser_actor.py     # DiffuserActor Wrapper
├── steering/                 # Concrete Steering Logic
│   ├── __init__.py
│   ├── dynaguide.py          # Dynamics-based guidance
│   └── analytical.py         # Analytical/Tweedie guidance
├── scripts/                  # Entry Points
│   ├── run_experiment.py     # Main evaluation loop
│   └── visualize.py          # Rendering script
├── pyproject.toml            # Dependencies (managed by uv)
└── README.md
```

## Core Components

### Types (`core/types.py`)

Standardized data transfer objects:
- `Observation`: Container for RGB images, depth, point clouds, proprioception, end-effector pose, and language instruction
- `Action`: Container for trajectory (sequence of poses) and gripper state

### Interfaces

- `BaseEnvironment`: Abstract interface for robot manipulation environments
- `BasePolicy`: Abstract interface for diffusion policies with optional steering support
- `BaseSteering`: Abstract interface for inference-time guidance mechanisms

## Usage

### Running Experiments

```bash
# Run with default configuration
python scripts/run_experiment.py

# Override configuration via command line
python scripts/run_experiment.py env=rlbench policy=diffuser_actor steering=dynaguide

# Visualize policy behavior
python scripts/visualize.py env=calvin policy=dp3
```

### Configuration

All experiments are configured via Hydra YAML files in `conf/`. The main entry point is `conf/config.yaml`, which composes:
- Environment configuration (`conf/env/`)
- Policy configuration (`conf/policy/`)
- Steering configuration (`conf/steering/`)

## Development

### Dependencies

Managed via `uv`:
```bash
uv sync
```

### Type Checking

```bash
mypy .
```

### Formatting

```bash
ruff format .
ruff check .
```

## License

[Add license information]
