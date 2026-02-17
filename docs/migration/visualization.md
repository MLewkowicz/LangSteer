# Visualization System Migration Guide

## Overview

The LangSteer visualization system has been consolidated from 4 separate scripts into a unified config-driven system. This guide explains how to migrate from the old scripts to the new system.

## What Changed?

### Before (Old System)
```bash
# 4 separate scripts for different visualization modes
python scripts/rollout_reference.py      # PyBullet GUI rendering
python scripts/visualize_cameras.py       # Camera views
python scripts/visualize_trajectories.py  # 3D trajectory analysis
python scripts/visualize_reference.py     # Reference plots
```

### After (New System)
```bash
# Single unified interface with config flags
python scripts/run_experiment.py visualization.render=true
python scripts/run_experiment.py visualization.cameras=true
python scripts/run_experiment.py visualization.trajectory_3d=true
python scripts/run_experiment.py visualization.reference_plot=true
```

## Migration Examples

### 1. PyBullet Rendering (rollout_reference.py → visualization.render)

**Old:**
```bash
python scripts/rollout_reference.py
```

**New:**
```bash
python scripts/run_experiment.py visualization.render=true
```

**With options:**
```bash
python scripts/run_experiment.py \
    visualization.render=true \
    visualization.rollout.frame_skip=2 \
    visualization.rollout.playback_speed=0.5
```

### 2. Camera Visualization (visualize_cameras.py → visualization.cameras)

**Old:**
```bash
python scripts/visualize_cameras.py
```

**New:**
```bash
python scripts/run_experiment.py visualization.cameras=true
```

**With options:**
```bash
python scripts/run_experiment.py \
    visualization.cameras=true \
    visualization.camera.save_images=true \
    visualization.camera.save_dir="outputs/my_cameras"
```

### 3. 3D Trajectory Analysis (visualize_trajectories.py → visualization.trajectory_3d)

**Old:**
```bash
python scripts/visualize_trajectories.py
```

**New:**
```bash
python scripts/run_experiment.py \
    visualization.trajectory_3d=true \
    num_episodes=1
```

**With options:**
```bash
python scripts/run_experiment.py \
    visualization.trajectory_3d=true \
    visualization.trajectory.num_rollouts=20 \
    visualization.trajectory.modes='[scatter,lines,success_comparison]'
```

### 4. Reference Trajectory Plots (visualize_reference.py → visualization.reference_plot)

**Old:**
```bash
python scripts/visualize_reference.py
```

**New:**
```bash
python scripts/run_experiment.py \
    visualization.reference_plot=true \
    steering=tweedie
```

## Combining Multiple Visualization Modes

One of the key benefits of the new system is the ability to combine multiple visualization modes:

```bash
# Render in PyBullet GUI + save camera views
python scripts/run_experiment.py \
    visualization.render=true \
    visualization.cameras=true

# Full visualization suite
python scripts/run_experiment.py \
    visualization.render=true \
    visualization.cameras=true \
    visualization.reference_plot=true \
    steering=tweedie
```

## Configuration via Config Files

You can also create custom config files for common visualization setups:

**conf/experiment/dp3_full_viz.yaml:**
```yaml
# @package _global_

defaults:
  - /policy: dp3
  - /env: calvin
  - /steering: tweedie
  - /visualization: base

# Override visualization settings
visualization:
  render: true
  cameras: true
  reference_plot: true

  camera:
    save_images: true
    show_live: false

  rollout:
    playback_speed: 1.0

num_episodes: 5
```

**Usage:**
```bash
python scripts/run_experiment.py experiment=dp3_full_viz
```

## Available Configuration Options

### Main Toggles
- `visualization.render`: PyBullet GUI rendering
- `visualization.cameras`: Camera feed visualization
- `visualization.trajectory_3d`: Multi-rollout 3D analysis
- `visualization.reference_plot`: Reference trajectory plots

### Camera Options
- `visualization.camera.save_dir`: Output directory
- `visualization.camera.display_mode`: "side_by_side" or "overlay"
- `visualization.camera.save_images`: Save PNG files
- `visualization.camera.show_live`: Display live matplotlib window

### Trajectory 3D Options
- `visualization.trajectory.num_rollouts`: Number of rollouts to collect
- `visualization.trajectory.max_steps`: Max steps per rollout
- `visualization.trajectory.modes`: List of plot types (scatter, lines, combined, success_comparison)
- `visualization.trajectory.plot_options.colormap`: Colormap for visualization

### Reference Plot Options
- `visualization.reference.save_dir`: Output directory
- `visualization.reference.plot_full_trajectory`: Generate full trajectory plot
- `visualization.reference.plot_sliding_windows`: Generate sliding window plots
- `visualization.reference.window_timesteps`: Timesteps for sliding windows

### Rollout Options
- `visualization.rollout.frame_skip`: Play every Nth frame
- `visualization.rollout.playback_speed`: Speed multiplier (1.0 = normal)
- `visualization.rollout.keep_gui_open`: Keep GUI open after rollout

## Programmatic Usage

You can also use the visualization system programmatically:

```python
from visualization import VisualizationManager, VisualizationConfig

# Create config
config = VisualizationConfig(
    render=True,
    cameras=True
)

# Initialize manager
viz_manager = VisualizationManager(config)

# Use during experiment
for episode in episodes:
    viz_manager.reset()
    result = run_episode(...)
    viz_manager.visualize_episode(env, result)
```

## Deprecation Timeline

- **v0.1.0** (Current): Old scripts work with deprecation warnings
- **v0.2.0** (Next release): Old scripts removed

Please migrate to the new system before the next major release.

## Benefits of New System

1. **Unified interface**: One command instead of 4 separate scripts
2. **Composable**: Combine multiple visualization modes
3. **Config-driven**: Easy to save and share visualization setups
4. **Less code**: ~50% reduction in visualization code
5. **Better organized**: Clear separation of concerns

## Need Help?

- See [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) for detailed usage examples
- Check [conf/visualization/base.yaml](conf/visualization/base.yaml) for all available options
- Report issues at: https://github.com/your-repo/issues
