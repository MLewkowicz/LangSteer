# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangSteer is a research framework for language-conditioned diffusion policy steering on robotic manipulation benchmarks. It modifies the diffusion denoising process at inference time using external guidance (reference trajectories, LLM-generated value maps) without retraining the base policy. Currently targets the CALVIN benchmark with 3D Diffuser Actor policies.

## Common Commands

```bash
# Install all dependencies (uses uv package manager)
uv sync

# Run inference experiment
uv run python scripts/run_experiment.py
uv run python scripts/run_experiment.py steering=tweedie env.task=open_drawer

# Run with VoxPoser steering
uv run python scripts/run_experiment.py steering=voxposer env.task=open_drawer

# Train Diffuser Actor
uv run python scripts/train_diffuser_actor.py training=diffuser_actor_calvin

# Code quality
ruff format .
ruff check .
mypy .
```

## Architecture

### Core Abstractions (`core/`)

All components communicate through two DTOs defined in `core/types.py`:
- **`Observation`**: rgb dict, proprio, ee_pose, instruction, optional depth/pcd
- **`Action`**: trajectory array `(H, 7)` + gripper float

Three abstract interfaces in `core/`:
- **`BasePolicy`** — takes Observation, returns Action (with optional steering hook)
- **`BaseEnvironment`** — reset/step interface producing Observations
- **`BaseSteering`** — guidance mechanism injected into the diffusion denoising loop

### Component Wiring

Experiments are composed via **Hydra configs** in `conf/`. The main config (`conf/config.yaml`) has defaults for env, policy, steering, rollout, and visualization groups. Override any parameter via CLI: `python scripts/run_experiment.py steering=tweedie steering.guidance_strength=2.0`.

Entry points:
- `scripts/run_experiment.py` — inference/evaluation loop
- `scripts/train_diffuser_actor.py` — Diffuser Actor training entry point

### Policy: 3D Diffuser Actor (`policies/`)

`policies/diffuser_actor.py` wraps the model components in `policies/diffuser_actor_components/`. Key details:
- Uses a deque-based gripper history buffer (length = `nhist`)
- Dual noise schedulers: position (scaled_linear) + rotation (squaredcos_cap_v2), both epsilon prediction
- Converts CALVIN observations to model format: RGB crops, per-pixel PCD, gripper history (quat wxyz), CLIP text embeddings
- Output: 20-step trajectories converted from 6D rotation + relative coords to euler + absolute poses
- `gripper_loc_bounds` are normalization bounds for gripper-centric coords, NOT absolute workspace bounds

### Environment: CALVIN (`envs/`)

`envs/calvin.py` adapts the CALVIN benchmark to the `BaseEnvironment` interface. Runtime monkey-patches in `envs/calvin_utils/gym_wrapper.py`:
- Removes tactile sensors from CALVIN config (NumPy 2.0 `np.float` deprecation)
- Patches PyBullet URDF search paths
- Converts continuous gripper actions to binary (-1/1) for CALVIN

`envs/calvin_utils/task_configs.py` defines initial conditions for all 34 CALVIN tasks.

### Steering (`steering/`)

`steering/tweedie.py` implements Tweedie guidance — uses Tweedie's formula for x_0 prediction from x_t, then computes analytical gradient guidance toward reference trajectories in epsilon space. Operates with dual schedulers (position/rotation).

`steering/voxposer_steering.py` implements VoxPoser guidance — uses LLM-generated 3D spatial value maps (affordance + avoidance) to guide the denoising process via precomputed gradient fields.

### Training (`training/`)

`training/policies/diffuser_actor/` — Diffuser Actor training with dataset loading, preprocessing, and SLURM support.

### Forecasters (`forecasters/`)

Trajectory forecasting for steering guidance. `TrajectoryForecaster` (neural) and `TweedieForecaster` (analytical) both implement `BaseForecaster`.

## Environment Variables

- `CALVIN_DATASET_PATH` — path to CALVIN dataset (task_D_D split)

## Key Conventions

- Python 3.10+ required (< 3.13)
- Hydra outputs go to `outputs/YYYY-MM-DD/HH-MM-SS/`
- New policies: implement `BasePolicy`, add wrapper in `policies/`, config in `conf/policy/`
- New environments: implement `BaseEnvironment`, add adapter in `envs/`, config in `conf/env/`
- New steering methods: implement `BaseSteering`, add module in `steering/`, config in `conf/steering/`
