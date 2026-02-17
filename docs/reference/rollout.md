# Reference Trajectory Rollout in PyBullet

This guide explains how to execute and visualize reference trajectories directly in the PyBullet physics simulator.

## Quick Start

Execute a reference trajectory in PyBullet with GUI visualization:

```bash
PYTHONPATH=/home/mlewkowicz/LangSteer:$PYTHONPATH uv run python scripts/rollout_reference.py \
    env.task=close_drawer \
    env.use_gui=true
```

## What It Does

The rollout script:
1. **Loads** the reference trajectory for the specified task from the CALVIN dataset
2. **Resets** the PyBullet environment to match the reference starting state
3. **Executes** each action from the reference trajectory step-by-step
4. **Displays** the robot following the trajectory in the GUI (if enabled)

This is useful for:
- **Verifying reference quality** - See if the reference trajectory successfully completes the task
- **Debugging steering** - Understand what the reference trajectory looks like before using it for guidance
- **Environment validation** - Confirm that environment state matches reference initial conditions
- **Task understanding** - Visualize successful demonstrations for each task

## Command-Line Options

### Basic Usage

```bash
# Rollout with GUI
uv run python scripts/rollout_reference.py env.task=close_drawer env.use_gui=true

# Rollout without GUI (faster)
uv run python scripts/rollout_reference.py env.task=close_drawer env.use_gui=false
```

### Playback Control

**Frame Skip** - Play every Nth frame (useful for faster preview):
```bash
# Play every 2nd frame
uv run python scripts/rollout_reference.py env.task=close_drawer rollout.frame_skip=2

# Play every 5th frame (very fast preview)
uv run python scripts/rollout_reference.py env.task=close_drawer rollout.frame_skip=5
```

**Playback Speed** - Adjust visualization speed:
```bash
# 2x speed
uv run python scripts/rollout_reference.py env.task=close_drawer rollout.playback_speed=2.0

# 0.5x speed (slow motion)
uv run python scripts/rollout_reference.py env.task=close_drawer rollout.playback_speed=0.5
```

**Combine Options**:
```bash
# Fast preview: skip frames + increase speed
uv run python scripts/rollout_reference.py \
    env.task=close_drawer \
    env.use_gui=true \
    rollout.frame_skip=3 \
    rollout.playback_speed=2.0
```

### Task Selection

```bash
# Different tasks
uv run python scripts/rollout_reference.py env.task=open_drawer
uv run python scripts/rollout_reference.py env.task=turn_on_lightbulb
uv run python scripts/rollout_reference.py env.task=move_slider_left
uv run python scripts/rollout_reference.py env.task=lift_red_block_table
```

## Configuration

### Rollout Config File

The default rollout configuration is in [`conf/rollout/default.yaml`](conf/rollout/default.yaml):

```yaml
# Playback settings
frame_skip: 1  # Play every Nth frame
playback_speed: 1.0  # Speed multiplier

# Video recording (optional)
record_video: false
video_path: "outputs/rollouts"
video_fps: 30
```

### Environment Config

Environment settings are inherited from [`conf/env/calvin.yaml`](conf/env/calvin.yaml):

```yaml
dataset_path: "/path/to/calvin/dataset"
split: "validation"
task: "close_drawer"
use_gui: true
action_repeat: 1
```

## Output

The script prints detailed progress information:

```
============================================================
Reference Trajectory Rollout in PyBullet
============================================================
Task: close_drawer
Dataset: /path/to/calvin/dataset
Split: validation
GUI: True
Frame skip: 1
Playback speed: 1.0x
============================================================

Loading reference trajectory for task: close_drawer

Loaded reference trajectory:
  Frames: 50
  Action shape: (50, 7)
  Action dims: [x, y, z, roll, pitch, yaw, gripper]
  Robot obs init: (15,)
  Scene obs init: (24,)

Initializing CALVIN environment...
Resetting environment to reference starting state...
  Initial observation shape: (39,)

============================================================
Starting rollout (press Ctrl+C to stop early)...
============================================================

Step 0/49 (frame 0/49) - Action: [0.010, -0.005, 0.020, ...]
Step 10/49 (frame 10/49) - Action: [0.012, -0.003, 0.018, ...]
...
Step 49/49 (frame 49/49) - Action: [0.001, 0.000, -0.001, ...]

============================================================
Rollout completed successfully!
============================================================
  Total frames played: 50
  Original trajectory length: 50

GUI is open. Press Ctrl+C to exit...
```

## Examples

### Preview Multiple Tasks

```bash
# Loop through several tasks
for task in close_drawer open_drawer move_slider_left turn_on_lightbulb; do
    echo "Previewing task: $task"
    uv run python scripts/rollout_reference.py \
        env.task=$task \
        env.use_gui=true \
        rollout.frame_skip=2 \
        rollout.playback_speed=1.5
done
```

### Fast Quality Check

```bash
# Quickly check if reference trajectory is reasonable
uv run python scripts/rollout_reference.py \
    env.task=close_drawer \
    env.use_gui=true \
    rollout.frame_skip=5 \
    rollout.playback_speed=3.0
```

### Detailed Inspection

```bash
# Slow-motion playback for detailed analysis
uv run python scripts/rollout_reference.py \
    env.task=close_drawer \
    env.use_gui=true \
    rollout.frame_skip=1 \
    rollout.playback_speed=0.3
```

## Programmatic Usage

You can also use the rollout functionality in your own Python scripts:

```python
from utils.reference_trajectory_loader import ReferenceTrajectoryLoader
from envs.calvin import CalvinEnvironment
import numpy as np

# Load reference trajectory
loader = ReferenceTrajectoryLoader(
    dataset_path="/path/to/calvin/dataset",
    split="validation",
    lang_ann_path="/path/to/auto_lang_ann.npy"
)

trajectory_data = loader.load_trajectory_for_task("close_drawer")
actions = trajectory_data['actions']  # (T, 7) numpy array
robot_obs_init = trajectory_data['robot_obs_init']
scene_obs_init = trajectory_data['scene_obs_init']

# Initialize environment
env_cfg = {
    'dataset_path': '/path/to/calvin/dataset',
    'task': 'close_drawer',
    'use_gui': True,
    'action_repeat': 1
}
env = CalvinEnvironment(env_cfg)

# Reset to reference starting state
obs = env.reset(robot_obs=robot_obs_init, scene_obs=scene_obs_init)

# Execute reference trajectory
for i, action in enumerate(actions):
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {i}/{len(actions)-1}")

    if done:
        print("Episode terminated early")
        break

env.close()
```

## Troubleshooting

### Issue: "No trajectory found for task"

**Cause**: Task name doesn't match any tasks in the language annotations.

**Solution**:
1. Check available tasks:
   ```python
   from utils.reference_trajectory_loader import ReferenceTrajectoryLoader

   loader = ReferenceTrajectoryLoader(
       dataset_path="/path/to/calvin",
       split="validation",
       lang_ann_path="/path/to/auto_lang_ann.npy"
   )

   print("Available tasks:", loader.get_available_tasks())
   ```

2. Use exact task name from the list

### Issue: Robot doesn't complete task

**Cause**: Reference trajectory may be incomplete or noisy.

**Possible solutions**:
- Try a different task episode (modify loader to load different episode)
- The reference trajectory is from real demonstrations which may not always succeed
- This is expected - not all demonstrations in CALVIN are perfect

### Issue: Environment state mismatch

**Cause**: Environment reset doesn't perfectly match reference starting state.

**Solution**:
- Verify that `robot_obs_init` and `scene_obs_init` are being passed correctly
- Check that CALVIN environment accepts these parameters
- See [envs/calvin.py](envs/calvin.py) `reset()` method

### Issue: Rollout is too slow

**Solutions**:
```bash
# Increase frame skip
uv run python scripts/rollout_reference.py env.task=close_drawer rollout.frame_skip=5

# Increase playback speed
uv run python scripts/rollout_reference.py env.task=close_drawer rollout.playback_speed=3.0

# Disable GUI (if visualization not needed)
uv run python scripts/rollout_reference.py env.task=close_drawer env.use_gui=false
```

## Related Documentation

- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - Plot and compare trajectories
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Tweedie steering implementation
- [TRAJECTORY_VISUALIZATION.md](TRAJECTORY_VISUALIZATION.md) - Trajectory data visualization

## Common Tasks

Here are some common CALVIN tasks you can try:

### Drawer Tasks
- `open_drawer`
- `close_drawer`

### Slider Tasks
- `move_slider_left`
- `move_slider_right`

### Light Tasks
- `turn_on_lightbulb`
- `turn_off_lightbulb`
- `turn_on_led`
- `turn_off_led`

### Block Manipulation
- `lift_red_block_table`
- `lift_blue_block_table`
- `push_red_block_left`
- `push_blue_block_right`
- `rotate_red_block_left`
- `rotate_blue_block_right`

## Technical Details

### Action Space

Reference trajectories contain 7-dimensional actions:
- `[x, y, z, roll, pitch, yaw, gripper]`
- Position: Relative end-effector displacement
- Orientation: Relative rotation (Euler angles)
- Gripper: Gripper state (-1 = open, 1 = closed)

### Trajectory Length

- Reference trajectories typically contain 30-50 frames
- Each frame represents one environment step
- Total episode time ≈ frames × action_repeat × timestep

### Initial State Alignment

The script ensures the environment starts from the exact same state as the reference:
1. Loads `robot_obs_init` (15D robot proprioception)
2. Loads `scene_obs_init` (24D scene state)
3. Passes these to `env.reset(robot_obs=..., scene_obs=...)`

This alignment is critical for Tweedie steering to work correctly.
