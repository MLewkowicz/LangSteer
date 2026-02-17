# Visualization Guide

Complete guide for visualizing experiments in LangSteer.

## Quick Start

Enable visualization modes via config flags:

```bash
# Camera feeds
python scripts/run_experiment.py visualization.cameras=true

# PyBullet GUI rendering
python scripts/run_experiment.py visualization.render=true

# 3D trajectory analysis
python scripts/run_experiment.py visualization.trajectory_3d=true

# Reference trajectory plots
python scripts/run_experiment.py visualization.reference_plot=true steering=tweedie

# Combine multiple modes
python scripts/run_experiment.py \
    visualization.render=true \
    visualization.cameras=true
```

## Visualization Modes

### 1. Camera Feeds (`visualization.cameras=true`)

Displays and saves RGB images from CALVIN's cameras:
- **Static camera**: Overhead view of workspace
- **Gripper camera**: Robot wrist-mounted view

**Configuration:**
```yaml
visualization:
  cameras: true
  camera:
    save_dir: "outputs/cameras"
    save_images: true
    show_live: false  # Display matplotlib window
    display_mode: "side_by_side"  # or "overlay"
```

**Output:**
- PNG images: `outputs/cameras/step_NNNN_static.png`, `step_NNNN_gripper.png`

### 2. PyBullet Rendering (`visualization.render=true`)

Real-time 3D visualization in PyBullet GUI.

**Configuration:**
```yaml
visualization:
  render: true
  rollout:
    frame_skip: 1         # Play every Nth frame
    playback_speed: 1.0   # Speed multiplier
    keep_gui_open: true   # Keep window open after completion
```

**Controls:**
- Mouse: Rotate/pan camera
- Scroll: Zoom
- Ctrl+C: Exit

### 3. Multi-Rollout 3D Analysis (`visualization.trajectory_3d=true`)

Generates interactive Plotly visualizations of end-effector trajectories from multiple rollouts.

**Configuration:**
```yaml
visualization:
  trajectory_3d: true
  trajectory:
    num_rollouts: 10
    max_steps: 50
    modes:
      - scatter       # 3D scatter plot
      - lines         # Connected line paths
      - combined      # Scatter + lines
      - success_comparison  # Success (green) vs failure (red)
    plot_options:
      point_size: 3
      line_width: 2
      opacity: 0.7
      colormap: "tab10"
    snapshot_auto_save: true  # Save initial state for reproducibility
```

**Output:**
- Interactive HTML files in `outputs/trajectory_viz/`
- Snapshot file: `.npz` format for reproducible rollouts

**Example:**
```bash
python scripts/run_experiment.py \
    visualization.trajectory_3d=true \
    visualization.trajectory.num_rollouts=20 \
    visualization.trajectory.modes='[scatter,lines,success_comparison]' \
    num_episodes=1  # Only need 1 episode to collect N rollouts
```

### 4. Reference Trajectory Plots (`visualization.reference_plot=true`)

Matplotlib visualizations of reference trajectories (requires steering to be active).

**Configuration:**
```yaml
visualization:
  reference_plot: true
  reference:
    save_dir: "outputs/reference_viz"
    plot_full_trajectory: true
    plot_sliding_windows: true
    window_timesteps: [0, 10, 20, 30]
```

**Output:**
- Full trajectory 3D plot with position/orientation components
- Sliding window plots showing prediction horizons
- PNG files in `outputs/reference_viz/`

**Example:**
```bash
python scripts/run_experiment.py \
    visualization.reference_plot=true \
    steering=tweedie \
    env.task=open_drawer
```

## Advanced Usage

### Custom Visualization Config

Create a custom config file for common visualization setups:

**conf/experiment/full_viz.yaml:**
```yaml
# @package _global_

defaults:
  - /policy: dp3
  - /env: calvin
  - /steering: tweedie
  - /visualization: base

visualization:
  render: true
  cameras: true
  reference_plot: true

  camera:
    save_images: true

  rollout:
    playback_speed: 0.5  # Slow motion

num_episodes: 3
```

**Usage:**
```bash
python scripts/run_experiment.py experiment=full_viz
```

### Programmatic Usage

```python
from visualization import VisualizationManager, VisualizationConfig

# Create config
config = VisualizationConfig(
    cameras=True,
    render=True,
    camera=CameraVisualizationConfig(save_images=True)
)

# Initialize manager
viz_manager = VisualizationManager(config)

# Use during experiment
for episode in range(num_episodes):
    viz_manager.reset()
    result = runner.run_episode(...)

    # Visualize step-by-step
    for step, obs in enumerate(trajectory):
        viz_manager.visualize_step(env, action, calvin_obs, step)
```

### Environment Snapshots for Reproducible Rollouts

When using `visualization.trajectory_3d=true`, snapshots ensure all rollouts start from identical state:

```bash
# Save a snapshot
python scripts/run_experiment.py \
    visualization.trajectory_3d=true \
    visualization.trajectory.snapshot_auto_save=true \
    visualization.trajectory.snapshot_save_path="my_snapshot.npz"

# Load existing snapshot
python scripts/run_experiment.py \
    visualization.trajectory_3d=true \
    visualization.trajectory.snapshot_load_path="my_snapshot.npz"
```

## Tips & Best Practices

1. **Performance**: Disable visualization during training/large evaluations
2. **Combining modes**: Camera + render works well together
3. **Multi-rollout**: Use `num_episodes=1` with `trajectory_3d` to avoid confusion
4. **Storage**: Camera images can accumulate quickly—set `save_images=false` if not needed
5. **PyBullet GUI**: Use `keep_gui_open=false` for automated runs

## Troubleshooting

**"No display" error with PyBullet:**
```bash
# Use headless mode or disable rendering
python scripts/run_experiment.py visualization.render=false
```

**Matplotlib not showing windows:**
```bash
# Set show_live=false and view saved PNG files instead
visualization.camera.show_live=false
```

**Plotly visualizations too large:**
```bash
# Reduce rollouts or max_steps
visualization.trajectory.num_rollouts=5
visualization.trajectory.max_steps=25
```

## Configuration Reference

See [conf/visualization/base.yaml](../../conf/visualization/base.yaml) for all available options.

## Migration from Old Scripts

The old visualization scripts have been removed. See [docs/migration/visualization.md](../migration/visualization.md) for migration instructions.
