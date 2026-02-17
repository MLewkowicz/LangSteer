# LangSteer Setup Guide

Complete setup instructions for running DP3 inference on CALVIN.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (for GPU inference)
- [uv](https://github.com/astral-sh/uv) package manager

## Quick Start (New Machine)

### 1. Clone Repository

```bash
git clone <your-repo-url> LangSteer
cd LangSteer
```

### 2. Install Dependencies

```bash
# Install all dependencies including CALVIN environment
uv sync

# Install the package in editable mode
uv pip install -e .

# Install DP3 dependencies
uv pip install diffusers dill einops termcolor
```

### 3. Setup CALVIN Environment

The CALVIN environment requires some additional setup for PyBullet assets:

```bash
# Run the setup script (handles URDF files and data paths)
bash scripts/setup_calvin.sh
```

**What this script does:**
- Copies Franka Panda URDF files to PyBullet data directory
- Creates symlinks for CALVIN data assets
- Sets up required directory structure

**Note:** The monkey-patch in `envs/calvin_utils/gym_wrapper.py` handles PyBullet search paths automatically, so URDF files will be found at runtime.

### 4. Download CALVIN Dataset

```bash
# Download the CALVIN dataset (task_D_D split recommended)
wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
unzip task_D_D.zip -d /path/to/calvin/dataset/

# Set environment variable
export CALVIN_DATASET_PATH=/path/to/calvin/dataset/task_D_D
```

**Important:** The code automatically patches CALVIN configs to remove tactile sensors (NumPy 2.0 compatibility), so **no manual config edits are needed**.

### 5. Download DP3 Checkpoint

```bash
# Download your trained DP3 checkpoint
# Set environment variable
export DP3_CHECKPOINT_PATH=/path/to/checkpoint.ckpt
```

### 6. Run Inference

```bash
# Run a single episode with visualization
python scripts/run_experiment.py experiment=dp3_calvin_inference

# Run headless evaluation (10 episodes)
python scripts/run_experiment.py experiment=dp3_calvin_debug \
  env.use_task_initial_condition=true \
  experiment.num_episodes=10

# Run specific task
python scripts/run_experiment.py experiment=dp3_calvin_inference \
  env.task=move_slider_left
```

## Configuration

### Environment Variables

```bash
export CALVIN_DATASET_PATH=/path/to/calvin/dataset/task_D_D
export DP3_CHECKPOINT_PATH=/path/to/dp3_checkpoint.ckpt
```

### Config Files

- `conf/env/calvin.yaml` - CALVIN environment settings
- `conf/policy/dp3.yaml` - DP3 policy architecture
- `conf/experiment/dp3_calvin_inference.yaml` - Inference experiment config
- `conf/experiment/dp3_calvin_debug.yaml` - Debug config (shorter episodes)

### Key Config Parameters

```yaml
# Environment
env:
  task: "open_drawer"  # CALVIN task name
  use_gui: false  # Enable PyBullet GUI rendering
  use_task_initial_condition: false  # Task-specific scene reset
  max_steps: 360  # Maximum steps per episode

# Policy
policy:
  ckpt_path: "${oc.env:DP3_CHECKPOINT_PATH}"
  device: "cuda"
  obs_horizon: 2
  pred_horizon: 16
  action_horizon: 8

# Experiment
experiment:
  num_episodes: 10
  enable_gui: false
  log_trajectory: false
```

## Automatic Configuration Patches

The following modifications are applied **automatically at runtime** (no manual edits needed):

### 1. Tactile Camera Removal
**Location:** `envs/calvin_utils/gym_wrapper.py:66-84`

Removes tactile sensors from CALVIN config to avoid NumPy 2.0 compatibility issues (`np.float` deprecation).

**Original CALVIN config:**
```yaml
cameras:
  static: {...}
  gripper: {...}
  tactile: {...}  # ❌ Causes np.float error
```

**Patched at runtime:**
```yaml
cameras:
  static: {...}
  gripper: {...}
  # tactile removed ✅
```

### 2. PyBullet URDF Search Path
**Location:** `envs/calvin_utils/gym_wrapper.py:52-61`

Monkey-patches `Robot.load()` to add PyBullet data path before loading URDFs.

### 3. Binary Gripper Action Conversion
**Location:** `envs/calvin.py:115-118`

Converts continuous gripper actions (DP3 output) to binary values (-1 or 1) required by CALVIN.

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'calvin_env'`
**Solution:** Install CALVIN package:
```bash
uv pip install git+https://github.com/mees/calvin_env.git
```

### Issue: `URDF file not found`
**Solution:** Run setup script:
```bash
bash scripts/setup_calvin.sh
```

### Issue: `AssertionError: gripper_action in (-1, 1)`
**Solution:** This is handled automatically by gripper action conversion in `envs/calvin.py`. If you still see this, check that you're using the latest code.

### Issue: `Weights only load failed` (PyTorch checkpoint)
**Solution:** Already handled - code uses `weights_only=False` for checkpoint loading.

### Issue: NumPy 2.0 `np.float` deprecation
**Solution:** Tactile camera is automatically removed from config at runtime.

## Self-Contained Setup Summary

✅ **No manual config edits required** - All CALVIN config patches are programmatic
✅ **No dataset modifications needed** - Tactile removal happens at runtime
✅ **Reproducible on new machines** - `uv sync` + `setup_calvin.sh` handles everything
✅ **Version controlled** - All code changes are in the repo

### Dependencies Tracked

- **Python packages:** `pyproject.toml`
- **CALVIN environment:** Git submodule + pip install
- **DP3 components:** Included in `policies/dp3_components/`
- **URDF files:** Copied by `setup_calvin.sh`
- **Config patches:** Applied programmatically at runtime

### What's NOT Tracked (External Data)

- CALVIN dataset (must be downloaded separately)
- DP3 checkpoint (must be provided by user)
- PyBullet data directory (setup script handles this)

## File Structure

```
LangSteer/
├── core/              # Core abstractions (BasePolicy, BaseEnvironment, etc.)
├── envs/              # Environment implementations
│   ├── calvin.py      # CALVIN environment wrapper
│   └── calvin_utils/  # CALVIN-specific utilities
│       ├── gym_wrapper.py       # Low-level CALVIN gym wrapper
│       ├── observation.py       # Observation processing
│       ├── language_ann.py      # Language annotation loading
│       └── task_configs.py      # Task-specific scene states
├── policies/          # Policy implementations
│   ├── dp3.py         # DP3 policy wrapper
│   └── dp3_components/  # Extracted DP3 components
├── steering/          # Steering modules (guidance functions)
├── scripts/           # Execution scripts
│   ├── run_experiment.py  # Main inference script
│   └── setup_calvin.sh    # CALVIN setup script
├── conf/              # Hydra configuration files
│   ├── env/           # Environment configs
│   ├── policy/        # Policy configs
│   └── experiment/    # Experiment configs
├── pyproject.toml     # Python dependencies
└── SETUP.md          # This file
```

## Available CALVIN Tasks

All 34 CALVIN tasks are supported with task-specific initial conditions:

**Drawer tasks:** `open_drawer`, `close_drawer`
**Slider tasks:** `move_slider_left`, `move_slider_right`
**Block manipulation:** `lift_red_block_table`, `lift_blue_block_table`, `lift_pink_block_table`
**Block placement:** `place_in_drawer`, `place_in_slider`, `push_into_drawer`
**Stacking:** `stack_block`, `unstack_block`
**Rotation:** `rotate_red_block_left`, `rotate_blue_block_right`, etc.
**Lighting:** `turn_on_lightbulb`, `turn_off_lightbulb`, `turn_on_led`, `turn_off_led`

See `envs/calvin_utils/task_configs.py` for complete list and initial conditions.


