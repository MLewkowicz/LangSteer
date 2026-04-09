# LangSteer Setup Guide

Complete setup instructions for running Diffuser Actor inference on CALVIN.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU
- [uv](https://github.com/astral-sh/uv) package manager

## Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url> LangSteer
cd LangSteer
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Setup CALVIN Environment

```bash
bash scripts/setup_calvin.sh
```

This handles URDF files and PyBullet data paths. The monkey-patch in `envs/calvin_utils/gym_wrapper.py` handles PyBullet search paths automatically at runtime.

### 4. Download CALVIN Dataset

```bash
wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
unzip task_D_D.zip -d /path/to/calvin/dataset/
export CALVIN_DATASET_PATH=/path/to/calvin/dataset/task_D_D
```

### 5. Download Diffuser Actor Checkpoint

```bash
# Set path to your trained checkpoint
export DA_CHECKPOINT_PATH=/path/to/checkpoint.pth
```

### 6. Run Inference

```bash
# Run with default settings
uv run python scripts/run_experiment.py \
  policy.ckpt_path=$DA_CHECKPOINT_PATH

# Run specific task
uv run python scripts/run_experiment.py \
  policy.ckpt_path=$DA_CHECKPOINT_PATH \
  env.task=open_drawer

# Run with steering
uv run python scripts/run_experiment.py \
  policy.ckpt_path=$DA_CHECKPOINT_PATH \
  steering=tweedie
```

## Configuration

### Environment Variables

```bash
export CALVIN_DATASET_PATH=/path/to/calvin/dataset/task_D_D
```

### Config Files

- `conf/env/calvin.yaml` — CALVIN environment settings
- `conf/policy/diffuser_actor.yaml` — Diffuser Actor policy
- `conf/steering/tweedie.yaml` — Tweedie steering
- `conf/steering/voxposer.yaml` — VoxPoser steering

## Automatic Runtime Patches

No manual config edits required:

1. **Tactile camera removal** (`envs/calvin_utils/gym_wrapper.py`) — removes tactile sensors for NumPy 2.0 compatibility
2. **PyBullet URDF search path** — monkey-patches `Robot.load()` to find URDFs
3. **Binary gripper actions** (`envs/calvin.py`) — converts continuous gripper to binary (-1/1)

## Troubleshooting

- **`ModuleNotFoundError: No module named 'calvin_env'`** — Run `uv pip install git+https://github.com/mees/calvin_env.git`
- **`URDF file not found`** — Run `bash scripts/setup_calvin.sh`
- **NumPy 2.0 `np.float` deprecation** — Handled automatically at runtime

## Available CALVIN Tasks

All 34 CALVIN tasks are supported. See `envs/calvin_utils/task_configs.py` for the complete list.

**Examples:** `open_drawer`, `close_drawer`, `move_slider_left`, `lift_red_block_table`, `turn_on_lightbulb`, `stack_block`
