# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangSteer is a research framework for language-conditioned diffusion policy steering on robotic manipulation benchmarks. It modifies the diffusion denoising process at inference time using external guidance (reference trajectories, dynamics models) without retraining the base policy. Currently targets the CALVIN benchmark with DP3 diffusion policies.

## Common Commands

```bash
# Install all dependencies (uses uv package manager)
uv sync

# Run inference experiment
uv run python scripts/run_experiment.py
uv run python scripts/run_experiment.py steering=tweedie env.task=open_drawer

# Train DP3 policy
uv run python scripts/train_dp3.py training=dp3_calvin

# Train with debug config (10 epochs, small batch)
uv run python scripts/train_dp3.py training=dp3_debug

# SLURM distributed training
sbatch scripts/slurm_train_dp3.sh

# Convert CALVIN dataset to Zarr format
uv run python scripts/convert_calvin_to_zarr.py

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
- `scripts/train_dp3.py` — DP3 training entry point

### Policy: DP3 (`policies/`)

`policies/dp3.py` wraps the diffusion model components in `policies/dp3_components/`. Key details:
- Uses a deque-based observation history buffer (length = `obs_horizon`)
- `dp3_components/dp3_policy.py` contains the core diffusion architecture (conditional denoising with U-Net)
- `dp3_components/encoder.py` has PointNet-based point cloud encoders
- `dp3_components/normalizer.py` handles input/output normalization with learnable stats
- Checkpoint loading supports multiple legacy formats with fallbacks

### Environment: CALVIN (`envs/`)

`envs/calvin.py` adapts the CALVIN benchmark to the `BaseEnvironment` interface. Runtime monkey-patches in `envs/calvin_utils/gym_wrapper.py`:
- Removes tactile sensors from CALVIN config (NumPy 2.0 `np.float` deprecation)
- Patches PyBullet URDF search paths
- Converts continuous gripper actions to binary (-1/1) for CALVIN

`envs/calvin_utils/task_configs.py` defines initial conditions for all 34 CALVIN tasks.

### Steering (`steering/`)

`steering/tweedie.py` implements Tweedie guidance — uses Tweedie's formula for x₀ prediction from xₜ, then computes gradient guidance toward reference trajectories. Supports timestep-scaled guidance.

### Training (`training/`)

`training/policies/dp3/trainer.py` — DP3TrainingWorkspace with multi-GPU DDP, checkpoint management (top-K by validation loss), EMA support, and WandB logging. Dataset loading and Zarr preprocessing live alongside the trainer.

### Forecasters (`forecasters/`)

Trajectory forecasting for steering guidance. `TrajectoryForecaster` (neural) and `TweedieForecaster` (analytical) both implement `BaseForecaster`.

## Environment Variables

- `CALVIN_DATASET_PATH` — path to CALVIN dataset (task_D_D split)
- `DP3_CHECKPOINT_PATH` — path to trained DP3 checkpoint
- `CALVIN_ZARR_PATH` — path to Zarr-converted dataset for training

## Key Conventions

- Python 3.10+ required (< 3.13)
- Hydra outputs go to `outputs/YYYY-MM-DD/HH-MM-SS/`
- New policies: implement `BasePolicy`, add wrapper in `policies/`, config in `conf/policy/`
- New environments: implement `BaseEnvironment`, add adapter in `envs/`, config in `conf/env/`
- New steering methods: implement `BaseSteering`, add module in `steering/`, config in `conf/steering/`
